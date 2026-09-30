"""
Measurement tools, so the master can be judged by numbers rather than hope.

Implements ITU-R BS.1770-4 loudness (LUFS), true-peak detection, crest
factor, stereo correlation and a band-energy breakdown. These are the same
metrics a mastering engineer reads off a meter.
"""

import numpy as np
from scipy.signal import lfilter, resample_poly, welch

from dsp.core import SR


# --------------------------------------------------------------------------
# Loudness (ITU-R BS.1770-4)
# --------------------------------------------------------------------------

# The two-stage "K-weighting" curve. Stage 1 is a high shelf approximating the
# acoustic effect of a human head; stage 2 is a high-pass reflecting how little
# the very low end contributes to perceived loudness. Coefficients are the
# spec's published values for 48 kHz.
_K_SHELF_B = [1.53512485958697, -2.69169618940638, 1.19839281085285]
_K_SHELF_A = [1.0, -1.69065929318241, 0.73248077421585]
_K_HPF_B = [1.0, -2.0, 1.0]
_K_HPF_A = [1.0, -1.99004745483398, 0.99007225036621]


def k_weight(x, sr=SR):
    if sr != 48000:
        raise ValueError("K-weighting coefficients here are 48 kHz only")
    y = lfilter(_K_SHELF_B, _K_SHELF_A, x, axis=0)
    return lfilter(_K_HPF_B, _K_HPF_A, y, axis=0)


def _block_powers(x, sr, window=0.400, overlap=0.75):
    """Mean-square power of each gating block, summed across channels."""
    w = int(window * sr)
    hop = int(w * (1.0 - overlap))
    y = k_weight(x, sr)
    if y.ndim == 1:
        y = y[:, None]
    n = len(y)
    if n < w:
        return np.zeros(0)
    starts = np.arange(0, n - w + 1, hop)
    return np.array([np.sum(np.mean(y[s:s + w] ** 2, axis=0)) for s in starts])


def lufs_integrated(x, sr=SR):
    """
    Gated integrated loudness.

    Two gates: an absolute one at -70 LUFS drops digital silence, and a
    relative one 10 LU below the ungated mean drops quiet passages. Without
    them a track's long fade-out would drag the reported loudness down and
    make it look quieter than it plays.
    """
    p = _block_powers(x, sr)
    if len(p) == 0:
        return -np.inf
    l = -0.691 + 10.0 * np.log10(np.maximum(p, 1e-12))

    keep = l > -70.0
    if not np.any(keep):
        return -np.inf
    rel = -0.691 + 10.0 * np.log10(np.mean(p[keep])) - 10.0
    keep &= l > rel
    if not np.any(keep):
        return -np.inf
    return -0.691 + 10.0 * np.log10(np.mean(p[keep]))


def lufs_short_term(x, sr=SR, window=3.0, hop=0.5):
    """Rolling 3-second loudness -- shows how the arrangement breathes."""
    w, h = int(window * sr), int(hop * sr)
    y = k_weight(x, sr)
    if y.ndim == 1:
        y = y[:, None]
    out = []
    for s in range(0, max(1, len(y) - w + 1), h):
        p = np.sum(np.mean(y[s:s + w] ** 2, axis=0))
        out.append((s / sr, -0.691 + 10.0 * np.log10(max(p, 1e-12))))
    return out


def loudness_range(x, sr=SR):
    """LRA: the spread between the quiet and loud parts, in LU. Club tracks
    typically land between 3 and 8 LU -- enough contrast to have a breakdown,
    not so much that the quiet sections vanish on a dancefloor."""
    st = np.array([v for _, v in lufs_short_term(x, sr, 3.0, 0.1)])
    st = st[st > -70.0]
    if len(st) < 2:
        return 0.0
    st = st[st > (np.mean(st) - 20.0)]
    return float(np.percentile(st, 95) - np.percentile(st, 10))


# --------------------------------------------------------------------------
# Peak / dynamics
# --------------------------------------------------------------------------

def true_peak_db(x, sr=SR, oversample=4):
    """
    Inter-sample peak. Reconstructing the analog waveform between samples can
    exceed the highest sample value; a converter or MP3 decoder will then clip
    even though the file measures under 0 dBFS.
    """
    y = x if x.ndim == 2 else x[:, None]
    pk = 0.0
    for c in range(y.shape[1]):
        up = resample_poly(y[:, c].astype(np.float32), oversample, 1)
        pk = max(pk, float(np.max(np.abs(up))))
    return 20.0 * np.log10(max(pk, 1e-12))


def sample_peak_db(x):
    return 20.0 * np.log10(max(float(np.max(np.abs(x))), 1e-12))


def crest_factor_db(x):
    """Peak-to-RMS ratio. Too low means over-limited and fatiguing; a
    four-to-the-floor master usually lands around 8-12 dB."""
    rms = float(np.sqrt(np.mean(x ** 2)))
    return 20.0 * np.log10(max(float(np.max(np.abs(x))), 1e-12) / max(rms, 1e-12))


# --------------------------------------------------------------------------
# Stereo
# --------------------------------------------------------------------------

def correlation(x):
    """
    Phase correlation, -1..+1.
      +1  identical channels (mono)
       0  fully decorrelated -- wide, still mono-safe
      -1  channels cancel completely when summed to mono
    A club master should stay comfortably positive.
    """
    l, r = x[:, 0], x[:, 1]
    d = np.sqrt(np.mean(l ** 2) * np.mean(r ** 2))
    return float(np.mean(l * r) / d) if d > 0 else 0.0


def mono_compatibility_db(x):
    """
    Level change when the mix is summed to mono.

    Reference is the RMS of an average channel, so two identical channels read
    exactly 0.0 dB and anti-phase channels read -inf. Anything below about
    -1.5 dB means phase cancellation is eating the mix on club systems, where
    the subwoofer feed is almost always a mono sum.
    """
    ref = np.sqrt(np.mean(x ** 2))                     # mean over both channels
    mono = np.sqrt(np.mean(((x[:, 0] + x[:, 1]) * 0.5) ** 2))
    return 20.0 * np.log10(max(mono, 1e-12) / max(ref, 1e-12))


def bass_correlation(x, sr=SR, fc=120.0):
    """Correlation of the low end only -- the part that must not cancel."""
    from dsp import filters as F
    lo = F.apply(x, F.lowpass(fc, 0.707, sr))
    return correlation(lo)


def high_correlation(x, sr=SR, fc=300.0):
    """
    Correlation above `fc`.

    Overall correlation is a poor width gauge for dance music: the low end
    carries most of the energy and is deliberately mono, which drags the
    figure toward +1 no matter how wide the rest is. Measuring only the range
    that is *allowed* to be wide shows what the listener actually perceives.
    """
    from dsp import filters as F
    hi = F.apply(x, F.highpass(fc, 0.707, sr))
    return correlation(hi)


# --------------------------------------------------------------------------
# Spectrum
# --------------------------------------------------------------------------

BANDS = [("sub", 20, 60), ("low", 60, 120), ("low-mid", 120, 400),
         ("mid", 400, 2000), ("high-mid", 2000, 6000),
         ("high", 6000, 12000), ("air", 12000, 20000)]


def band_energy(x, sr=SR):
    """Share of total energy per band, in dB relative to the total."""
    mono = np.mean(x, axis=1) if x.ndim == 2 else x
    f, p = welch(mono, sr, nperseg=8192)
    total = np.trapezoid(p, f)
    out = []
    for name, lo, hi in BANDS:
        m = (f >= lo) & (f < hi)
        e = np.trapezoid(p[m], f[m]) if np.any(m) else 0.0
        out.append((name, lo, hi, 10.0 * np.log10(max(e / max(total, 1e-20), 1e-12))))
    return out


def report(x, sr=SR, name="master"):
    """Full measurement printout."""
    lines = [f"--- {name} ---",
             f"  duration         {len(x)/sr:8.2f} s",
             f"  integrated       {lufs_integrated(x, sr):8.2f} LUFS",
             f"  loudness range   {loudness_range(x, sr):8.2f} LU",
             f"  sample peak      {sample_peak_db(x):8.2f} dBFS",
             f"  true peak        {true_peak_db(x, sr):8.2f} dBTP",
             f"  crest factor     {crest_factor_db(x):8.2f} dB",
             f"  correlation      {correlation(x):8.2f}",
             f"  bass correlation {bass_correlation(x, sr):8.2f}",
             f"  high correlation {high_correlation(x, sr):8.2f}",
             f"  mono sum delta   {mono_compatibility_db(x):8.2f} dB",
             "  spectral balance:"]
    for n, lo, hi, d in band_energy(x, sr):
        bar = "#" * max(0, int(40 + d * 1.2))
        lines.append(f"    {n:9s} {lo:5d}-{hi:<6d} {d:6.1f} dB  {bar}")
    return "\n".join(lines)
