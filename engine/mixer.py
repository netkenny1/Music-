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

import numpy as np

from dsp.core import SR, db, stereo
from dsp import filters as F
from dsp import dynamics as D
from dsp import space as S


# --------------------------------------------------------------------------
# Channel strip
# --------------------------------------------------------------------------

class Channel:
    """One instrument's signal path from raw buffer to the mix bus."""

    def __init__(self, name, n, sr=SR, gain_db=0.0, pan=0.0, width=1.0,
                 eq=None, comp=None, sends=None, duck=0.0, mono_below=None,
                 sat=None, hp=None, lp=None, filter_curve=None):
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

        # 4. Colour
        if self.sat:
            y = D.saturate(y, sr=self.sr, **self.sat)

        # 5. Dynamics
        if self.comp:
            before = float(np.max(np.abs(y)))
            y = D.compress(y, sr=self.sr, **self.comp)
            after = float(np.max(np.abs(y)))
            self.gr_db = 20 * np.log10(max(after, 1e-9) / max(before, 1e-9))

        # 6. Sidechain pump, applied after compression so the compressor's own
        #    release does not fight the ducking curve.
        if self.duck > 0 and duck_env is not None:
            env = 1.0 - self.duck * (1.0 - duck_env)
            y = y * env[:, None]

        # 7. Stereo placement
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

    def render(self, duck_env=None, verbose=True):
        """Sum channels, run the sends, and return the pre-master mix."""
        mix = np.zeros((self.n, 2))
        send_bufs = {k: np.zeros((self.n, 2)) for k in self.buses}

        for name, ch in self.channels.items():
            y = ch.process(duck_env)
            if not np.any(y):
                continue
            mix += y
            for bus_name, level in ch.sends.items():
                if bus_name in send_bufs and level > 0:
                    send_bufs[bus_name] += y * level
            if verbose:
                pk = 20 * np.log10(max(float(np.max(np.abs(y))), 1e-9))
                print(f"    {name:12s} peak {pk:6.1f} dBFS"
                      f"{'  GR ' + format(ch.gr_db, '5.1f') + ' dB' if ch.comp else ''}")

        for bus_name, cfg in self.buses.items():
            src = send_bufs[bus_name]
            if not np.any(src):
                continue
            wet = S.convolve(src, cfg["ir"])
            for coeffs in cfg["eq"]:
                wet = F.apply(wet, coeffs)
            if cfg["width"] != 1.0:
                wet = S.width(wet, cfg["width"])
            # Ducking the reverb returns too keeps the tails from filling the
            # gaps the sidechain just carved open.
            if cfg["duck"] > 0 and duck_env is not None:
                wet = wet * (1.0 - cfg["duck"] * (1.0 - duck_env))[:, None]
            wet = wet * db(cfg["gain_db"])
            self.bus_returns[bus_name] = wet
            mix += wet
            if verbose:
                pk = 20 * np.log10(max(float(np.max(np.abs(wet))), 1e-9))
                print(f"    [{bus_name:10s}] peak {pk:6.1f} dBFS")

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
        F.lowshelf(80.0, -0.5, 0.8, sr),       # the mix is already bass-forward
        F.peaking(250.0, -1.6, 0.9, sr),       # keep the low-mids uncluttered
        F.peaking(2400.0, 1.5, 0.8, sr),       # lower presence
        F.peaking(4000.0, 3.2, 0.7, sr),       # presence -- the 2-6k cliff
        F.highshelf(8500.0, 4.0, 0.7, sr),     # air
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

    def run(gain_db):
        out = D.limit(y * db(gain_db), sr=sr, ceiling_db=ceiling_db,
                      lookahead=0.005, release=0.070)
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
