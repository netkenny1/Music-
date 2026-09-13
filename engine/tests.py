"""
Validation suite for the DSP engine.

These are correctness checks, not taste checks. They verify the properties the
mix relies on: that oscillators do not alias, that the limiter cannot overshoot
its ceiling, that filters actually attenuate where they claim to, and that the
loudness meter agrees with the published reference values in ITU-R BS.1770.

Run with:  python3 engine/tests.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scipy.signal import welch

from dsp.core import (SR, sine, saw, square, noise, stereo, to_db, adsr,
                      perc_env, pan)
from dsp import filters as F
from dsp import dynamics as D
from dsp import space as S
import analysis as A

FAILURES = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f"   {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


# --------------------------------------------------------------------------

def test_oscillator_aliasing():
    """
    A band-limited saw at a high fundamental should put almost no energy at
    frequencies that are not harmonics. Aliased partials fold to arbitrary
    non-harmonic positions, so we measure how much energy lands off-harmonic.
    """
    print("\noscillators")
    f0 = 2500.0
    naive_phase = (np.cumsum(np.full(SR, f0 / SR))) % 1.0
    naive = 2.0 * naive_phase - 1.0
    blep = saw(f0, SR)

    def off_harmonic_ratio(x):
        fr, p = welch(x, SR, nperseg=16384)
        harmonics = np.arange(1, int(0.5 * SR / f0) + 1) * f0
        mask = np.ones(len(fr), dtype=bool)
        for h in harmonics:
            mask &= np.abs(fr - h) > 60.0
        mask &= fr > 100.0
        return np.trapezoid(p[mask], fr[mask]) / np.trapezoid(p, fr)

    r_naive = off_harmonic_ratio(naive)
    r_blep = off_harmonic_ratio(blep)
    check("PolyBLEP saw aliases less than naive saw",
          r_blep < r_naive * 0.5,
          f"naive {r_naive:.5f} -> blep {r_blep:.5f}")
    check("PolyBLEP off-harmonic energy under 1%", r_blep < 0.01,
          f"{r_blep * 100:.3f}%")

    for name, fn in [("sine", sine), ("saw", saw), ("square", square)]:
        x = fn(440.0, SR)
        check(f"{name} is finite and bounded",
              np.all(np.isfinite(x)) and np.max(np.abs(x)) < 1.6)


def test_envelopes():
    print("\nenvelopes")
    e = adsr(SR, SR, a=0.01, d=0.1, s=0.5, r=0.2)
    check("ADSR length exact", len(e) == SR)
    check("ADSR starts at zero", e[0] < 1e-6)
    check("ADSR peaks at one", abs(e.max() - 1.0) < 1e-6)
    check("ADSR ends at zero", e[-1] < 1e-3, f"{e[-1]:.2e}")
    p = perc_env(SR, SR, decay=0.2)
    check("percussive envelope decays monotonically after attack",
          np.all(np.diff(p[200:]) <= 1e-9))


def test_filters():
    print("\nfilters")
    nyq = SR / 2

    def response_db(coeffs, freq):
        x = sine(freq, SR)
        y = F.apply(x, coeffs)
        return to_db(np.sqrt(np.mean(y[SR // 4:] ** 2)) /
                     np.sqrt(np.mean(x[SR // 4:] ** 2)))

    check("lowpass passband flat at 100 Hz",
          abs(response_db(F.lowpass(1000.0), 100.0)) < 0.5,
          f"{response_db(F.lowpass(1000.0), 100.0):+.2f} dB")
    check("lowpass -3 dB at cutoff",
          abs(response_db(F.lowpass(1000.0), 1000.0) + 3.0) < 1.0,
          f"{response_db(F.lowpass(1000.0), 1000.0):+.2f} dB")
    check("lowpass 12 dB/oct stopband slope",
          response_db(F.lowpass(1000.0), 4000.0) < -20.0,
          f"{response_db(F.lowpass(1000.0), 4000.0):+.2f} dB at 2 oct")
    check("highpass rejects below cutoff",
          response_db(F.highpass(1000.0), 100.0) < -30.0,
          f"{response_db(F.highpass(1000.0), 100.0):+.2f} dB")

    x = sine(1000.0, SR)
    y = F.lp24(x, 1000.0)
    g24 = to_db(np.sqrt(np.mean(y[SR // 4:] ** 2)) /
                np.sqrt(np.mean(x[SR // 4:] ** 2)))
    y4 = F.lp24(sine(4000.0, SR), 1000.0)
    g24b = to_db(np.sqrt(np.mean(y4[SR // 4:] ** 2)) /
                 np.sqrt(np.mean(sine(4000.0, SR)[SR // 4:] ** 2)))
    check("24 dB/oct steeper than 12 dB/oct", g24b < -40.0,
          f"{g24b:+.1f} dB two octaves up")

    peak_g = None
    x = sine(3000.0, SR)
    y = F.apply(x, F.peaking(3000.0, 6.0, 1.0))
    peak_g = to_db(np.sqrt(np.mean(y[SR // 4:] ** 2)) /
                   np.sqrt(np.mean(x[SR // 4:] ** 2)))
    check("peaking EQ delivers requested gain", abs(peak_g - 6.0) < 0.3,
          f"asked +6.0, got {peak_g:+.2f} dB")

    # all filters must stay stable
    for fc in [20.0, 200.0, 5000.0, 20000.0, 23000.0]:
        y = F.lp24(saw(110.0, SR), fc, 4.0)
        check(f"lp24 stable at fc={fc:g}", np.all(np.isfinite(y)) and
              np.max(np.abs(y)) < 50.0)


def test_limiter():
    print("\nlimiter")
    for ceil in [-0.3, -1.0, -3.0]:
        hot = stereo(saw(110.0, SR * 2) * 6.0, saw(110.7, SR * 2) * 6.0)
        out = D.limit(hot, ceiling_db=ceil)
        pk = A.sample_peak_db(out)
        check(f"never exceeds ceiling {ceil:+.1f} dBFS", pk <= ceil + 0.01,
              f"peak {pk:+.3f} dBFS")

    quiet = stereo(sine(220.0, SR) * 0.05, sine(220.0, SR) * 0.05)
    out = D.limit(quiet, ceiling_db=-1.0)
    check("transparent below ceiling", np.allclose(out, quiet, atol=1e-12),
          f"max deviation {np.max(np.abs(out - quiet)):.2e}")

    spike = np.zeros(SR)
    spike[:SR // 2] = sine(80.0, SR // 2) * 0.3
    spike[SR // 2] = 9.0
    out = D.limit(stereo(spike, spike), ceiling_db=-1.0)
    check("catches isolated transient", A.sample_peak_db(out) <= -0.99,
          f"peak {A.sample_peak_db(out):+.3f} dBFS")


def test_compressor():
    print("\ncompressor")
    loud = stereo(sine(440.0, SR * 2) * 0.9, sine(440.0, SR * 2) * 0.9)
    out = D.compress(loud, threshold=-20.0, ratio=4.0, makeup=0.0)
    # Measure after the attack has settled. The first few milliseconds pass
    # through at full level by design -- that overshoot is what an attack time
    # *is* -- so a whole-buffer peak reading describes the transient, not the
    # compression.
    red = A.sample_peak_db(out[SR:]) - A.sample_peak_db(loud[SR:])
    check("reduces level above threshold", red < -8.0, f"{red:+.1f} dB steady state")

    # -20 dBFS in at 4:1 over a -20 dB threshold should give ~-14 dB
    expected = (1.0 / 4.0 - 1.0) * (A.sample_peak_db(loud) - (-20.0))
    check("reduction matches the ratio", abs(red - expected) < 1.5,
          f"predicted {expected:+.1f} dB, measured {red:+.1f} dB")

    soft = stereo(sine(440.0, SR * 2) * 0.02, sine(440.0, SR * 2) * 0.02)
    out = D.compress(soft, threshold=-20.0, ratio=4.0, makeup=0.0)
    red = A.sample_peak_db(out) - A.sample_peak_db(soft)
    check("leaves signal below threshold alone", abs(red) < 0.6,
          f"{red:+.2f} dB")

    check("stereo linking keeps channels balanced",
          abs(A.correlation(D.compress(
              stereo(sine(440.0, SR) * 0.9, sine(440.0, SR) * 0.3),
              threshold=-25.0, ratio=6.0)) - 1.0) < 1e-6)


def test_saturation():
    print("\nsaturation")
    # 7 kHz is chosen deliberately: tanh generates odd harmonics, and the 7th
    # (49 kHz) folds back to 1 kHz at a 48 kHz sample rate -- squarely inside
    # the measurement band and nowhere near a real harmonic. With a 6 kHz
    # fundamental every alias lands back on top of a harmonic and the test
    # cannot see the difference at all.
    x = sine(7000.0, SR) * 0.9
    plain = np.tanh(x * 4.0)
    os4 = D.saturate(x, 4.0, "tanh", oversample=4)

    def alias_energy(y):
        fr, p = welch(y, SR, nperseg=16384)
        mask = (fr > 500) & (fr < 3000)
        return np.trapezoid(p[mask], fr[mask]) / np.trapezoid(p, fr)

    a_plain, a_os = alias_energy(plain), alias_energy(os4)
    check("oversampling reduces distortion aliasing", a_os < a_plain * 0.5,
          f"{a_plain:.5f} -> {a_os:.5f}")
    check("saturation output bounded", np.max(np.abs(os4)) < 1.5)


def test_space():
    print("\nspace")
    ir = S.reverb_ir(rt60=2.0)
    check("reverb IR is stereo", ir.shape[1] == 2)
    check("reverb channels decorrelated", abs(A.correlation(ir)) < 0.3,
          f"correlation {A.correlation(ir):+.3f}")

    # measure actual decay against the requested RT60
    env = np.abs(ir[:, 0])
    w = int(0.02 * SR)
    sm = np.convolve(env, np.ones(w) / w, mode="same")
    pk = np.argmax(sm)
    tail = to_db(sm[pk:] / sm[pk])
    idx = np.argmax(tail < -40.0)
    measured = (idx / SR) * (60.0 / 40.0) if idx > 0 else 0.0
    check("reverb decay near requested RT60", 1.2 < measured < 3.2,
          f"asked 2.0 s, measured {measured:.2f} s")

    # Quadrature bass: the two channels are fully decorrelated (correlation 0)
    # but both carry real energy, so there is something for mono_below to
    # centre. Perfectly anti-phase bass is pure side signal, and mono_below
    # correctly annihilates it rather than centring it -- a valid result, but
    # a degenerate one that says nothing about the centring behaviour.
    quad = stereo(np.sin(2 * np.pi * 50.0 * np.arange(SR) / SR),
                  np.cos(2 * np.pi * 50.0 * np.arange(SR) / SR))
    check("decorrelated bass starts uncentred",
          abs(A.bass_correlation(quad)) < 0.2,
          f"correlation {A.bass_correlation(quad):+.3f}")
    centred = S.mono_below(quad, 120.0)
    check("mono_below centres the low end",
          A.bass_correlation(centred) > 0.99,
          f"correlation {A.bass_correlation(centred):+.3f}")
    # and it must leave the highs alone
    hi = stereo(sine(3000.0, SR), noise(SR, seed=9) * 0.3)
    check("mono_below leaves the top end untouched",
          abs(A.high_correlation(S.mono_below(hi, 120.0), fc=1000.0)
              - A.high_correlation(hi, fc=1000.0)) < 0.02)
    check("width=0 collapses to mono",
          A.correlation(S.width(stereo(sine(400.0, SR),
                                       sine(404.0, SR)), 0.0)) > 0.999)


def test_loudness():
    print("\nloudness metering (ITU-R BS.1770-4)")
    # The spec's reference: a 1 kHz sine at -20 dBFS in both channels reads
    # -20.0 LUFS (within the stated +/-0.1 tolerance).
    t = sine(1000.0, SR * 5) * 0.1
    m = A.lufs_integrated(stereo(t, t))
    check("reference tone reads -20 LUFS", abs(m + 20.0) < 0.15,
          f"{m:.3f} LUFS")

    louder = A.lufs_integrated(stereo(t * 2, t * 2))
    check("doubling amplitude adds 6 LU", abs((louder - m) - 6.02) < 0.1,
          f"{louder - m:+.2f} LU")

    check("true peak >= sample peak",
          A.true_peak_db(stereo(t, t)) >= A.sample_peak_db(stereo(t, t)) - 1e-6)


def test_panning():
    print("\npanning")
    x = np.ones(1000)
    c = pan(x, 0.0)
    power = np.mean(c[:, 0] ** 2 + c[:, 1] ** 2)
    for p in [-1.0, -0.5, 0.0, 0.5, 1.0]:
        c = pan(x, p)
        pw = np.mean(c[:, 0] ** 2 + c[:, 1] ** 2)
        check(f"constant power at pan={p:+.1f}", abs(pw - power) < 1e-9)


def main():
    print("=" * 64)
    print("DSP ENGINE VALIDATION")
    print("=" * 64)
    test_oscillator_aliasing()
    test_envelopes()
    test_filters()
    test_limiter()
    test_compressor()
    test_saturation()
    test_space()
    test_loudness()
    test_panning()

    print("\n" + "=" * 64)
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: " + ", ".join(FAILURES))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
