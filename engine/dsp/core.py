"""
Core synthesis primitives: oscillators, noise, envelopes, and helpers.

Design notes
------------
Every oscillator here is *band-limited* using PolyBLEP correction. A naive
sawtooth generated as `2*phase - 1` has infinite harmonics; once a harmonic
exceeds Nyquist (sr/2) it folds back down into the audible range as an
inharmonic "buzz" that no amount of mixing can remove. PolyBLEP replaces the
instantaneous jump at the waveform discontinuity with a short polynomial
transition, which cancels most of that foldback. This is the single biggest
difference between "coded music that sounds cheap" and "coded music that
sounds like a synth".
"""

import numpy as np

SR = 48_000  # master sample rate (Hz)


# --------------------------------------------------------------------------
# Level / pitch helpers
# --------------------------------------------------------------------------

def db(gain_db):
    """Decibels -> linear amplitude multiplier."""
    return 10.0 ** (np.asarray(gain_db, dtype=float) / 20.0)


def to_db(amplitude, floor=1e-9):
    """Linear amplitude -> decibels, with a floor so log(0) never happens."""
    return 20.0 * np.log10(np.maximum(np.abs(amplitude), floor))


def midi_to_hz(note):
    """MIDI note number -> frequency. 69 = A4 = 440 Hz."""
    return 440.0 * 2.0 ** ((np.asarray(note, dtype=float) - 69.0) / 12.0)


def cents(n):
    """Cents of detune -> frequency ratio multiplier."""
    return 2.0 ** (np.asarray(n, dtype=float) / 1200.0)


# --------------------------------------------------------------------------
# Phase generation
# --------------------------------------------------------------------------

def phasor(freq, n, sr=SR, phase0=0.0):
    """
    Accumulate phase in [0,1) for `n` samples.

    `freq` may be a scalar or a per-sample array, which is what lets us do
    pitch envelopes (kick drums) and vibrato without extra machinery.
    Returns (phase, dt) where dt is the per-sample phase increment -- PolyBLEP
    needs dt to know how wide to make its correction window.
    """
    dt = np.broadcast_to(np.asarray(freq, dtype=float) / sr, (n,)).astype(float)
    phase = (phase0 + np.cumsum(dt)) % 1.0
    return phase, dt


def _blep(t, dt):
    """
    PolyBLEP residual: a 2-sample-wide polynomial approximation of the
    difference between a band-limited step and an instantaneous one.
    Subtracting it at each discontinuity removes most aliasing.
    """
    y = np.zeros_like(t)

    rising = t < dt                      # just after the jump
    if np.any(rising):
        x = t[rising] / dt[rising]
        y[rising] = x + x - x * x - 1.0

    falling = t > (1.0 - dt)             # just before the jump
    if np.any(falling):
        x = (t[falling] - 1.0) / dt[falling]
        y[falling] = x * x + x + x + 1.0

    return y


# --------------------------------------------------------------------------
# Oscillators
# --------------------------------------------------------------------------

def sine(freq, n, sr=SR, phase0=0.0):
    """Pure sine. No aliasing possible -- it has exactly one harmonic."""
    phase, _ = phasor(freq, n, sr, phase0)
    return np.sin(2.0 * np.pi * phase)


def saw(freq, n, sr=SR, phase0=0.0):
    """Band-limited sawtooth. The workhorse of subtractive synthesis:
    every harmonic present, amplitude falling as 1/n."""
    phase, dt = phasor(freq, n, sr, phase0)
    return 2.0 * phase - 1.0 - _blep(phase, dt)


def square(freq, n, sr=SR, duty=0.5, phase0=0.0):
    """Band-limited pulse. Two discontinuities per cycle, so two BLEPs."""
    phase, dt = phasor(freq, n, sr, phase0)
    naive = np.where(phase < duty, 1.0, -1.0)
    naive += _blep(phase, dt)
    naive -= _blep((phase + (1.0 - duty)) % 1.0, dt)
    return naive


def triangle(freq, n, sr=SR, phase0=0.0):
    """Triangle via integrated square -- inherits the square's band-limiting.
    The leaky integrator removes the DC that exact integration would build up."""
    sq = square(freq, n, sr, 0.5, phase0)
    f = np.mean(np.broadcast_to(np.asarray(freq, dtype=float), (n,)))
    out = np.cumsum(sq) * (4.0 * max(f, 1.0) / sr)
    return out - _dc_block(out, sr)


def _dc_block(x, sr, fc=10.0):
    """Return the DC/sub-sonic component of x (a very slow moving average)."""
    a = np.exp(-2.0 * np.pi * fc / sr)
    from scipy.signal import lfilter
    return lfilter([1.0 - a], [1.0, -a], x)


def supersaw(freq, n, sr=SR, voices=7, detune_cents=14.0, spread=1.0, seed=0):
    """
    Detuned saw stack, returned as (left, right).

    Two things make this sound expensive rather than muddy:
      1. Each voice gets a *random* start phase. Identical phases would sum to
         one loud saw with a comb-filtered attack instead of a lush ensemble.
      2. Voices are distributed across the stereo field in mirrored pairs, so
         the stack is wide but its mono sum stays balanced.
    """
    rng = np.random.default_rng(seed)
    left = np.zeros(n)
    right = np.zeros(n)

    for i in range(voices):
        # -1..+1 position within the detune spread
        pos = 0.0 if voices == 1 else (2.0 * i / (voices - 1) - 1.0)
        f = np.asarray(freq, dtype=float) * cents(pos * detune_cents)
        v = saw(f, n, sr, phase0=rng.random())

        # centre voice stays centred; outer voices fan out
        pan = pos * spread
        gl = np.sqrt(0.5 * (1.0 - pan))
        gr = np.sqrt(0.5 * (1.0 + pan))
        # detuned voices sit slightly back so the centre pitch stays defined
        amp = 1.0 / np.sqrt(voices) * (1.0 - 0.25 * abs(pos))
        left += v * gl * amp
        right += v * gr * amp

    return left, right


def noise(n, seed=0, kind="white"):
    """White or pink noise. Pink (1/f) is closer to how real cymbals and
    room tone distribute energy, so it sits in a mix more naturally."""
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(n)
    if kind == "white":
        return w
    # Paul Kellet's economical pink filter: three cascaded one-poles
    from scipy.signal import lfilter
    b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
    a = [1.0, -2.494956002, 2.017265875, -0.522189400]
    return lfilter(b, a, w) * 3.0


# --------------------------------------------------------------------------
# Envelopes
# --------------------------------------------------------------------------

def _curve(n, start, end, curve):
    """
    Segment shaper. curve > 1 bends toward the start value (fast move then
    settle -- how physical decays actually behave); curve == 1 is linear.
    """
    if n <= 0:
        return np.zeros(0)
    t = np.linspace(0.0, 1.0, n, endpoint=False)
    return start + (end - start) * (t ** curve)


def adsr(n, sr=SR, a=0.005, d=0.1, s=0.6, r=0.2, curve=2.5, hold=None):
    """
    ADSR envelope of exactly `n` samples.

    `curve` applies to decay and release so they fall quickly then taper,
    matching how our ears expect plucked and struck sounds to behave. A linear
    decay sounds artificial and "digital" by comparison.
    """
    if hold is None:
        hold = max(0.0, n / sr - (a + d + r))

    na = max(1, int(a * sr))
    nd = max(1, int(d * sr))
    nh = max(0, int(hold * sr))
    nr = max(1, int(r * sr))

    seg_a = _curve(na, 0.0, 1.0, 1.0 / curve)   # attack bends the other way
    seg_d = _curve(nd, 1.0, s, curve)
    seg_h = np.full(nh, s)
    seg_r = _curve(nr, s, 0.0, curve)

    env = np.concatenate([seg_a, seg_d, seg_h, seg_r])
    if len(env) < n:
        env = np.concatenate([env, np.zeros(n - len(env))])
    return env[:n]


def perc_env(n, sr=SR, attack=0.002, decay=0.25, curve=3.0):
    """One-shot percussive envelope: near-instant attack, exponential-ish fall."""
    na = max(1, int(attack * sr))
    nd = max(1, n - na)
    seg_a = np.linspace(0.0, 1.0, na)
    t = np.linspace(0.0, 1.0, nd)
    seg_d = np.exp(-curve * t * (nd / sr) / max(decay, 1e-4))
    env = np.concatenate([seg_a, seg_d])
    return env[:n] if len(env) >= n else np.concatenate([env, np.zeros(n - len(env))])


def fade(x, sr=SR, fade_in=0.002, fade_out=0.004):
    """Apply short raised-cosine fades so a buffer never starts or ends on a
    non-zero sample (which would click)."""
    y = np.array(x, dtype=float, copy=True)
    ni = min(int(fade_in * sr), len(y) // 2)
    no = min(int(fade_out * sr), len(y) // 2)
    if ni > 0:
        y[:ni] *= 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, ni))
    if no > 0:
        y[-no:] *= 0.5 + 0.5 * np.cos(np.linspace(0, np.pi, no))
    return y


# --------------------------------------------------------------------------
# Buffer utilities
# --------------------------------------------------------------------------

def add_at(dest, src, pos):
    """Mix `src` into `dest` starting at sample `pos`, clipping at the edges.
    Works for mono (n,) and stereo (n,2) buffers."""
    if pos >= len(dest):
        return
    if pos < 0:
        src = src[-pos:]
        pos = 0
        if len(src) == 0:
            return
    end = min(pos + len(src), len(dest))
    dest[pos:end] += src[:end - pos]


def stereo(left, right):
    """Pack two mono arrays into an (n, 2) stereo buffer."""
    return np.stack([left, right], axis=-1)


def pan(x, position=0.0):
    """
    Constant-power pan, -1 (hard left) .. +1 (hard right).

    Constant *power* (sin/cos law) rather than constant amplitude: a linear
    pan law makes centred sounds drop ~3 dB relative to hard-panned ones,
    which is why naive panning makes a mix feel hollow in the middle.
    """
    position = float(np.clip(position, -1.0, 1.0))
    angle = (position + 1.0) * np.pi / 4.0
    return stereo(x * np.cos(angle), x * np.sin(angle))
