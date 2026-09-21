"""
The mix: channel strips, effect sends, and the master chain.

A mix is mostly about *allocation*. Two sounds competing for the same
frequency range and the same stereo position will always fight, no matter how
good each one sounds alone. So every channel here is assigned:

  * a frequency slot   -- via EQ, cutting what it does not need
  * a stereo position  -- via pan and mid/side width
  * a depth            -- via how much reverb it gets and how pre-delayed

Three shared reverbs plus one delay serve the whole track. Using sends rather
than per-channel reverbs is not just cheaper: putting several instruments in
the *same* simulated room is what makes them sound like one performance
instead of a stack of separate recordings.
"""

import multiprocessing as mp
import os

import numpy as np

from dsp.core import SR, db, stereo
from dsp import filters as F
from dsp import dynamics as D
from dsp import space as S


# --------------------------------------------------------------------------
# Channel strip
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# Parallel processing
# --------------------------------------------------------------------------
#
# Channel strips are independent of one another until the sum, and reverb
# buses are independent of one another until the sum, so both stages are
# embarrassingly parallel. The buffers are large (a 200 s stereo channel at
# 48 kHz is ~160 MB) and there are sixteen of them, so instead of pickling
# them out to workers the mixer is stashed in a module global and the pool is
# started with fork(): every child inherits the parent's memory copy-on-write
# and reads its channel for free. Only the processed result crosses back.
#
# Each worker also pins BLAS/OpenMP to one thread: the DSP here is
# element-wise numpy and scipy.signal, which do not benefit from threading,
# and four processes each spawning four threads would just thrash the cores.

_SHARED = {}


def _limit_threads():
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"


def _process_channel(name):
    mx, duck = _SHARED["mx"], _SHARED["duck"]
    ch = mx.channels[name]
    y = ch.process(duck)
    return name, y, ch.gr_db, ch.peak_in


def _process_bus(bus_name):
    mx, duck = _SHARED["mx"], _SHARED["duck"]
    cfg = mx.buses[bus_name]
    src = _SHARED["sends"][bus_name]
    wet = S.convolve(src, cfg["ir"])
    for coeffs in cfg["eq"]:
        wet = F.apply(wet, coeffs)
    if cfg["width"] != 1.0:
        wet = S.width(wet, cfg["width"])
    if cfg["duck"] > 0 and duck is not None:
        wet = wet * (1.0 - cfg["duck"] * (1.0 - duck))[:, None]
    return bus_name, wet * db(cfg["gain_db"])


def _pool(jobs):
    """Fork pool sized to the machine, or None when there is nothing to gain."""
    n = min(jobs, os.cpu_count() or 1)
    if n <= 1:
        return None
    return mp.get_context("fork").Pool(n, initializer=_limit_threads)


class Channel:
    """One instrument's signal path from raw buffer to the mix bus."""

    def __init__(self, name, n, sr=SR, gain_db=0.0, pan=0.0, width=1.0,
                 eq=None, comp=None, sends=None, duck=0.0, mono_below=None,
                 sat=None, hp=None, lp=None, filter_curve=None,
                 excite=None):
        self.name = name
        self.sr = sr
        self.buf = np.zeros((n, 2))
        self.gain_db = gain_db
        self.pan = pan
        self.width = width
        self.eq = eq or []                  # list of biquad coefficient tuples
        self.comp = comp                    # dict of compress() kwargs
        self.sends = sends or {}            # {bus_name: linear_send_level}
        self.duck = duck                    # 0..1 amount of sidechain pumping
        self.mono_below = mono_below        # Hz, or None
        self.sat = sat                      # dict of saturate() kwargs
        self.hp = hp                        # channel high-pass Hz
        self.lp = lp                        # channel low-pass Hz
        self.filter_curve = filter_curve    # per-sample cutoff automation
        self.excite = excite                # dict of exciter() kwargs
        self.peak_in = 0.0
        self.gr_db = 0.0

    def add(self, mono_or_stereo, pos, pan=None):
        """Mix a rendered note into this channel at sample `pos`."""
        x = mono_or_stereo
        if x.ndim == 1:
            p = self.pan if pan is None else pan
            angle = (float(np.clip(p, -1, 1)) + 1.0) * np.pi / 4.0
            x = np.stack([x * np.cos(angle), x * np.sin(angle)], axis=-1)
        if pos >= len(self.buf):
            return
        if pos < 0:
            x = x[-pos:]
            pos = 0
            if len(x) == 0:
                return
        end = min(pos + len(x), len(self.buf))
        self.buf[pos:end] += x[:end - pos]

    def process(self, duck_env=None):
        """Run the strip. Returns the post-fader stereo signal."""
        y = self.buf
        self.peak_in = float(np.max(np.abs(y))) if len(y) else 0.0
        if self.peak_in == 0.0:
            return y

        # 1. Housekeeping filters first -- never compress rumble you intend
        #    to throw away, it just wastes gain reduction.
        if self.hp:
            y = F.apply(y, F.highpass(self.hp, 0.707, self.sr))
        if self.lp:
            y = F.apply(y, F.lowpass(self.lp, 0.707, self.sr))

        # 2. Automated filter (section sweeps), before static EQ so the
        #    sweep acts on the raw instrument rather than on its EQ curve.
        if self.filter_curve is not None:
            y = np.stack([
                F.sweep_lowpass(y[:, c], self.filter_curve, 1.15, self.sr,
                                block=128, poles=4)
                for c in range(2)], axis=-1)

        # 3. Tone shaping
        for coeffs in self.eq:
            y = F.apply(y, coeffs)

        # 4. Presence. The exciter sits after the EQ so the tone shaping
        #    decides which band breeds the new harmonics, and before the
        #    compressor so its output is levelled with everything else.
        if self.excite:
            y = D.exciter(y, sr=self.sr, **self.excite)

        # 5. Colour
        if self.sat:
            y = D.saturate(y, sr=self.sr, **self.sat)

        # 6. Dynamics
        if self.comp:
            before = float(np.max(np.abs(y)))
            y = D.compress(y, sr=self.sr, **self.comp)
            after = float(np.max(np.abs(y)))
            self.gr_db = 20 * np.log10(max(after, 1e-9) / max(before, 1e-9))

        # 7. Sidechain pump, applied after compression so the compressor's own
        #    release does not fight the ducking curve.
        if self.duck > 0 and duck_env is not None:
            env = 1.0 - self.duck * (1.0 - duck_env)
            y = y * env[:, None]

        # 8. Stereo placement
        if self.width != 1.0:
            y = S.width(y, self.width)
        if self.mono_below:
            y = S.mono_below(y, self.mono_below, self.sr)

        return y * db(self.gain_db)


# --------------------------------------------------------------------------
# Mixer
# --------------------------------------------------------------------------

class Mixer:
    def __init__(self, n, sr=SR):
        self.n = n
        self.sr = sr
        self.channels = {}
        self.buses = {}
        self.bus_returns = {}

    def channel(self, name, **kw):
        ch = Channel(name, self.n, self.sr, **kw)
        self.channels[name] = ch
        return ch

    def bus(self, name, ir, gain_db=0.0, eq=None, width=1.0, duck=0.0):
        """Register an effect return fed by channel sends."""
        self.buses[name] = dict(ir=ir, gain_db=gain_db, eq=eq or [],
                                width=width, duck=duck)

    def render(self, duck_env=None, verbose=True, keep_stems=False,
               parallel=True):
        """Sum channels, run the sends, and return the pre-master mix.

        With `keep_stems` the processed output of every channel and bus is
        retained in `self.stems`. Diagnosing a spectral problem in the sum
        is guesswork; diagnosing it per channel is measurement.

        `parallel` runs the channel strips, then the reverb buses, across
        all cores. Results are identical to the serial path: every strip is
        a pure function of its own buffer and the shared duck envelope.
        """
        mix = np.zeros((self.n, 2))
        send_bufs = {k: np.zeros((self.n, 2)) for k in self.buses}
        if keep_stems:
            self.stems = {}

        # -- channel strips ---------------------------------------------
        names = [nm for nm, ch in self.channels.items() if np.any(ch.buf)]
        _SHARED.update(mx=self, duck=duck_env)
        pool = _pool(len(names)) if parallel else None
        try:
            if pool is None:
                results = map(_process_channel, names)
            else:
                results = pool.imap(_process_channel, names)
            for name, y, gr_db, peak_in in results:
                ch = self.channels[name]
                ch.gr_db, ch.peak_in = gr_db, peak_in
                mix += y
                for bus_name, level in ch.sends.items():
                    if bus_name in send_bufs and level > 0:
                        send_bufs[bus_name] += y * level
                if keep_stems:
                    self.stems[name] = y
                if verbose:
                    pk = 20 * np.log10(max(float(np.max(np.abs(y))), 1e-9))
                    print(f"    {name:12s} peak {pk:6.1f} dBFS"
                          f"{'  GR ' + format(ch.gr_db, '5.1f') + ' dB' if ch.comp else ''}",
                          flush=True)
        finally:
            if pool is not None:
                pool.close(); pool.join()

        # -- effect buses --------------------------------------------------
        live = [b for b in self.buses if np.any(send_bufs[b])]
        _SHARED.update(sends=send_bufs)
        pool = _pool(len(live)) if parallel else None
        try:
            if pool is None:
                results = map(_process_bus, live)
            else:
                results = pool.imap(_process_bus, live)
            for bus_name, wet in results:
                self.bus_returns[bus_name] = wet
                if keep_stems:
                    self.stems[f"[{bus_name}]"] = wet
                mix += wet
                if verbose:
                    pk = 20 * np.log10(max(float(np.max(np.abs(wet))), 1e-9))
                    print(f"    [{bus_name:10s}] peak {pk:6.1f} dBFS", flush=True)
        finally:
            if pool is not None:
                pool.close(); pool.join()
            _SHARED.clear()

        return mix


# --------------------------------------------------------------------------
# Master chain
# --------------------------------------------------------------------------

def master_chain(mix, sr=SR, target_lufs=-9.3, ceiling_db=-0.9, verbose=True):
    """
    Master processing, in the order that gives the cleanest result.

    1. **Sub-sonic filter.** Below ~25 Hz there is nothing musical, only
       energy that steals headroom from the limiter.
    2. **Broad tone EQ.** Wide, gentle moves only. Anything needing a narrow
       cut at this stage should have been fixed on the channel.
    3. **Glue compression.** Slow attack (30 ms) so transients pass through
       untouched, ~1-2 dB of reduction. The point is cohesion, not level.
    4. **Mid/side widening above 300 Hz.** Widening the whole spectrum would
       de-centre the bass; restricting it to the top keeps the low end solid
       while the air and reverb open up.
    5. **Saturation, blended.** Gentle tube drive rounds peaks and adds
       harmonics, so the limiter works less hard for the same loudness.
    6. **Loudness-targeted limiting.** Rather than guessing a drive amount,
       the chain measures its own LUFS, calculates the gain needed to hit the
       target, limits, re-measures and corrects. Limiting changes loudness
       non-linearly (it raises density while capping peaks), so a single
       calculated gain always lands short -- the second pass closes the gap.
    """
    import analysis as A     # imported here to keep the DSP layer standalone

    y = mix

    y = F.apply(y, F.highpass(24.0, 0.707, sr))

    y = F.chain(
        y,
        # Reference-matched: the drop's third-octave spectrum was measured
        # against the tilt a commercial house master sits on (about -3
        # dB/octave from 100 Hz to 1 kHz, -4.5 above). Before this pass it
        # was 3-6 dB light at 125-200 Hz -- the warmth -- and 5-11 dB hot
        # from 8 to 16 kHz, hat and shaker fizz left over from an earlier
        # "too dark" correction. The channels were fixed first (the organ
        # and pad high-passes were cutting the organ's 16' drawbar); the
        # master only tilts the remainder.
        F.lowshelf(80.0, -0.9, 0.8, sr),       # the mix is already bass-forward
        F.peaking(170.0, 2.4, 1.0, sr),        # warmth: bass harmonics, organ 16'
        F.peaking(250.0, 0.5, 0.9, sr),
        F.peaking(540.0, -0.8, 1.1, sr),       # the shared pile-up
        F.peaking(900.0, 1.2, 1.0, sr),
        F.peaking(1800.0, 2.6, 0.9, sr),       # the hole under presence
        F.peaking(4300.0, 3.2, 0.7, sr),
        F.highshelf(9000.0, -1.0, 0.7, sr),
        F.highshelf(14000.0, -5.5, 0.6, sr),   # the top octave, tamed
    )

    before = float(np.max(np.abs(y)))
    y = D.compress(y, sr=sr, threshold=-16.0, ratio=2.0, attack=0.030,
                   release=0.220, knee=8.0, makeup=0.0)
    if verbose:
        gr = 20 * np.log10(max(float(np.max(np.abs(y))), 1e-9) / max(before, 1e-9))
        print(f"    glue comp     {gr:+5.1f} dB")

    # width above 300 Hz only
    lo = F.apply(y, F.lowpass(300.0, 0.707, sr))
    hi = y - lo
    y = lo + S.width(hi, 1.30)
    y = S.mono_below(y, 110.0, sr)

    y = D.saturate(y, drive=1.5, mode="tube", sr=sr, oversample=4, mix=0.35)

    # --- loudness-targeted limiting ---------------------------------------
    # Solved rather than guessed. Loudness after limiting is a sub-linear
    # function of the drive applied: past a point, every extra dB of gain is
    # partly eaten by extra gain reduction, so a naive "add the shortfall"
    # correction always undershoots. A secant solver measures that local slope
    # from the last two attempts and steps by shortfall/slope instead, which
    # converges in two or three passes instead of crawling.
    measured = A.lufs_integrated(y, sr)
    if verbose:
        print(f"    pre-limiter   {measured:6.2f} LUFS")

    # Oversample for true-peak detection once; every pass below is the same
    # signal at a different scalar gain, so its block peaks are peak0 * gain.
    peak0 = D._true_block_peak(y, 16, sr)

    def run(gain_db):
        out = D.limit(y * db(gain_db), sr=sr, ceiling_db=ceiling_db,
                      lookahead=0.005, release=0.070,
                      peak=peak0 * db(gain_db))
        return out, A.lufs_integrated(out, sr)

    g_prev = target_lufs - measured
    out, l_prev = run(g_prev)
    best = (abs(target_lufs - l_prev), g_prev, out)

    g = g_prev + (target_lufs - l_prev)
    for _ in range(4):
        if abs(target_lufs - l_prev) < 0.15:
            break
        out, l = run(g)
        if abs(target_lufs - l) < best[0]:
            best = (abs(target_lufs - l), g, out)

        raw_slope = (l - l_prev) / (g - g_prev) if abs(g - g_prev) > 1e-6 else 1.0
        # Saturation guard: if extra drive has stopped buying loudness, the
        # material has hit the ceiling its own waveform allows. Pushing
        # further only flattens transients for nothing, so stop and keep the
        # least-driven result that got closest.
        if raw_slope < 0.15:
            if verbose:
                print(f"    limiter       loudness saturated at {l:.2f} LUFS"
                      f" -- holding drive rather than over-limiting")
            break

        slope = float(np.clip(raw_slope, 0.25, 1.0))
        g_prev, l_prev = g, l
        g = g + (target_lufs - l) / slope

    out = best[2]
    if verbose:
        print(f"    limiter       drive {best[1]:+.2f} dB -> "
              f"{A.lufs_integrated(out, sr):.2f} LUFS "
              f"(target {target_lufs:.1f}), "
              f"peak {20*np.log10(max(float(np.max(np.abs(out))),1e-9)):+.2f} dBFS")

    return out
