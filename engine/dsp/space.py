"""
Space: reverb, delay, and stereo-field control.

Reverb here is *convolution* reverb with a synthetic impulse response rather
than a recirculating delay network. Two reasons:

  - Quality. A real room's decay is frequency dependent: air and soft
    surfaces absorb treble far faster than bass, so a hall stays warm as it
    fades. We build that in explicitly by giving each frequency band its own
    decay time. Cheap algorithmic reverbs decay uniformly and sound metallic.

  - Speed. Convolution via overlap-add FFT is O(n log n) over the whole
    track, where a sample-by-sample delay network in Python would take
    minutes.

Placement in a mix is three separate cues, and we control each one:
  * left/right  -> panning
  * width       -> mid/side balance and decorrelation
  * distance    -> reverb amount, pre-delay, and high-frequency rolloff
"""

import numpy as np
from scipy.signal import oaconvolve

from .core import SR, db, stereo
from . import filters as F


# --------------------------------------------------------------------------
# Impulse response construction
# --------------------------------------------------------------------------

def reverb_ir(sr=SR, rt60=2.0, predelay=0.02, size=1.0, damping=0.55,
              width=1.0, er_level=0.5, seed=7):
    """
    Build a stereo impulse response.

    `rt60`    seconds for the tail to fall 60 dB (the standard decay metric)
    `damping` 0..1, how much faster treble dies than bass
    `er_level` level of the discrete early reflections that convey room size

    Left and right are generated from *independent* noise, which decorrelates
    the tail. Correlated tails collapse to the centre and sound like a mono
    reverb played through two speakers; decorrelated ones envelop the listener.
    """
    rng = np.random.default_rng(seed)
    n = int(sr * (rt60 * 1.4 + predelay + 0.05))
    t = np.arange(n) / sr

    # --- late tail: per-band decay ---------------------------------------
    # Bass rings longest, air dies first. `damping` sets how steep that is.
    bands = [(20, 180), (180, 700), (700, 2500), (2500, 7000), (7000, 16000)]
    factors = [1.15, 1.0, 0.82, 0.60, 0.38]

    tail = np.zeros((n, 2))
    for ch in range(2):
        nz = rng.standard_normal(n)
        acc = np.zeros(n)
        for (lo, hi), f in zip(bands, factors):
            band_rt = rt60 * (1.0 - damping * (1.0 - f)) * size
            band_rt = max(band_rt, 0.05)
            decay = np.exp(-6.907 * t / band_rt)     # ln(1000) = 6.907 -> -60 dB
            acc += F.bandlimit(nz, lo, min(hi, 0.45 * sr), sr) * decay
        tail[:, ch] = acc

    # Build-up: real rooms take a few ms to reach full diffusion instead of
    # starting at maximum density, which is what makes a tail sound "behind"
    # the source rather than layered on top of it.
    build = np.minimum(1.0, t / 0.012) ** 2
    tail *= build[:, None]

    # --- early reflections: sparse discrete echoes ------------------------
    er = np.zeros((n, 2))
    n_er = 18
    times = np.sort(rng.uniform(0.004, 0.075 * size, n_er))
    for k, tt in enumerate(times):
        idx = int(tt * sr)
        if idx >= n:
            continue
        g = (1.0 - k / n_er) ** 1.5 * rng.uniform(0.5, 1.0)
        # each reflection arrives at the two ears at slightly different times
        er[idx, 0] += g * rng.choice([-1.0, 1.0])
        off = idx + int(rng.uniform(0.0002, 0.0012) * sr)
        if off < n:
            er[off, 1] += g * rng.choice([-1.0, 1.0])
    er = np.stack([F.bandlimit(er[:, c], 200, 6500, sr) for c in range(2)], axis=-1)

    ir = tail + er * er_level

    # --- stereo width via mid/side scaling --------------------------------
    if width != 1.0:
        m = (ir[:, 0] + ir[:, 1]) * 0.5
        s = (ir[:, 0] - ir[:, 1]) * 0.5 * width
        ir = np.stack([m + s, m - s], axis=-1)

    # --- pre-delay ---------------------------------------------------------
    # A gap between the dry sound and its reverb keeps the source intelligible
    # and, psychoacoustically, reads as a bigger room.
    pd = int(predelay * sr)
    if pd > 0:
        ir = np.concatenate([np.zeros((pd, 2)), ir])[:n]

    # Normalise so that adding reverb never changes the send level.
    ir /= np.sqrt(np.sum(ir ** 2) / 2.0) + 1e-12
    return ir


def delay_ir(sr=SR, time=0.25, feedback=0.42, repeats=14, ping_pong=True,
             damping=0.55, width=1.0):
    """
    Delay rendered as an impulse response.

    A feedback delay line is mathematically a train of decaying taps, so we
    can just place the taps directly. That also lets us darken each successive
    repeat (as an analog delay's filtered feedback path would) without any
    recursion at all.
    """
    n = int(sr * time * (repeats + 1)) + 1024
    ir = np.zeros((n, 2))

    tap = np.zeros(max(64, int(sr * 0.01)))
    tap[0] = 1.0

    for k in range(1, repeats + 1):
        g = feedback ** k
        if g < 1e-4:
            break
        # each repeat loses more treble -- repeats recede into the distance
        cut = max(900.0, 13000.0 * (1.0 - damping * 0.55) ** k)
        shaped = F.apply(np.copy(tap), F.lowpass(cut, 0.707, sr))
        shaped = F.apply(shaped, F.highpass(180.0, 0.707, sr))  # keep echoes out of the bass

        idx = int(k * time * sr)
        if idx + len(shaped) >= n:
            break
        if ping_pong:
            # odd repeats left, even repeats right -> the echo bounces
            ch = 0 if k % 2 == 1 else 1
            ir[idx:idx + len(shaped), ch] += shaped * g
            ir[idx:idx + len(shaped), 1 - ch] += shaped * g * (1.0 - width) * 0.5
        else:
            ir[idx:idx + len(shaped), 0] += shaped * g
            ir[idx:idx + len(shaped), 1] += shaped * g

    return ir


def convolve(x, ir):
    """
    Convolve mono or stereo `x` with a stereo IR, truncated to the input
    length. Uses overlap-add FFT convolution.
    """
    n = len(x)
    if x.ndim == 1:
        left = oaconvolve(x, ir[:, 0])[:n]
        right = oaconvolve(x, ir[:, 1])[:n]
    else:
        left = oaconvolve(x[:, 0], ir[:, 0])[:n]
        right = oaconvolve(x[:, 1], ir[:, 1])[:n]
    return np.stack([left, right], axis=-1)


# --------------------------------------------------------------------------
# Stereo field
# --------------------------------------------------------------------------

def mid_side(x):
    """Stereo -> (mid, side). Mid is what a mono listener hears."""
    return (x[:, 0] + x[:, 1]) * 0.5, (x[:, 0] - x[:, 1]) * 0.5


def from_mid_side(m, s):
    return np.stack([m + s, m - s], axis=-1)


def width(x, amount=1.0):
    """
    Stereo width. 0 = mono, 1 = unchanged, >1 = wider.

    Widening is just turning up the side signal. Past about 1.6 the mix starts
    to hollow out in mono, because the side content cancels when summed -- so
    this is used sparingly and never on low frequencies.
    """
    m, s = mid_side(x)
    return from_mid_side(m, s * amount)


def mono_below(x, fc=120.0, sr=SR, poles=4):
    """
    Collapse everything below `fc` to mono, leave the rest untouched.

    This is the single most important spatial rule in club music. Low
    frequencies carry most of the energy; if they are out of phase between
    channels they partially cancel on a mono-summed club subwoofer, and the
    track loses its bottom end exactly where it matters most. Vinyl cutting
    lathes require it too.

    The side signal is removed with a 24 dB/oct filter by default, not 12.
    A gentle slope is the wrong tool here: an octave below a 120 Hz crossover,
    12 dB/oct still leaves about -14 dB of side content, so the very lowest
    octave -- the part that matters most -- stays partly stereo. Doubling the
    slope puts that residue around -30 dB, which is genuinely mono.
    """
    m, s = mid_side(x)
    s = F.hp24(s, fc, 0.707, sr) if poles >= 4 else \
        F.apply(s, F.highpass(fc, 0.707, sr))
    return from_mid_side(m, s)


def haas(x, ms=12.0, sr=SR, side=1):
    """
    Haas/precedence widening: delay one channel by a few milliseconds.

    Under ~30 ms the brain fuses the two arrivals into one event but reads it
    as wide. Cheap and effective, but it is *not* mono-safe -- the delay
    becomes comb filtering when summed -- so we reserve it for high-frequency
    ear candy where the comb notches fall outside the important range.
    """
    d = int(ms * sr / 1000.0)
    y = np.copy(x)
    if d > 0:
        y[:, side] = np.concatenate([np.zeros(d), x[:-d, side]])
    return y


def chorus(x, sr=SR, rate=0.28, depth_ms=5.5, base_ms=14.0, mix=0.42, voices=3, seed=3):
    """
    Modulated fractional-delay chorus.

    Each voice reads the input at a slowly wobbling delay, which detunes it
    marginally. Because the LFOs run at different rates and phases, the voices
    drift in and out of alignment and the result thickens and widens without
    the static comb filtering of a fixed delay. Fractional delay is done with
    linear interpolation via `np.interp`, so it stays fully vectorised.
    """
    rng = np.random.default_rng(seed)
    mono_in = x.ndim == 1
    y = x[:, None] if mono_in else x
    n = len(y)
    t = np.arange(n) / sr
    idx = np.arange(n, dtype=float)

    out = np.zeros((n, 2))
    for v in range(voices):
        r = rate * (0.7 + 0.5 * v) * rng.uniform(0.9, 1.1)
        ph = rng.random() * 2 * np.pi
        d = (base_ms + depth_ms * np.sin(2 * np.pi * r * t + ph)) * sr / 1000.0
        src = y[:, min(v % 2, y.shape[1] - 1)]
        tapped = np.interp(idx - d, idx, src, left=0.0, right=0.0)
        p = -1.0 + 2.0 * (v / max(1, voices - 1))
        out[:, 0] += tapped * np.cos((p + 1) * np.pi / 4)
        out[:, 1] += tapped * np.sin((p + 1) * np.pi / 4)

    out /= np.sqrt(voices)
    dry = y if y.shape[1] == 2 else np.repeat(y, 2, axis=1)
    return mix * out + (1.0 - mix) * dry
