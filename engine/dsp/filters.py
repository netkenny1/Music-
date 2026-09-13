"""
Filters: RBJ biquad cookbook, cascades, and a block-rate modulated filter.

Why biquads and not hand-rolled one-poles everywhere: a biquad (2-pole,
2-zero) is the smallest structure that gives independent control over cutoff
AND resonance, and scipy runs it in C via `lfilter`, so a 10-million-sample
track filters in milliseconds instead of minutes.

For *moving* filters (the classic house filter sweep) we recompute
coefficients once per short block rather than per sample. Real plugins do
exactly this -- a 64-sample block at 48 kHz is a 1.3 ms control rate, far
faster than any audible sweep, and it keeps the cost near zero.
"""

import numpy as np
from scipy.signal import lfilter, lfilter_zi, sosfilt

from .core import SR


# --------------------------------------------------------------------------
# RBJ cookbook coefficients
# --------------------------------------------------------------------------

def _w0_alpha(fc, q, sr):
    fc = float(np.clip(fc, 10.0, 0.492 * sr))   # keep below Nyquist
    q = max(float(q), 0.05)
    w0 = 2.0 * np.pi * fc / sr
    alpha = np.sin(w0) / (2.0 * q)
    return w0, alpha, np.cos(w0)


def lowpass(fc, q=0.707, sr=SR):
    w0, alpha, cw = _w0_alpha(fc, q, sr)
    b = np.array([(1 - cw) / 2, 1 - cw, (1 - cw) / 2])
    a = np.array([1 + alpha, -2 * cw, 1 - alpha])
    return b / a[0], a / a[0]


def highpass(fc, q=0.707, sr=SR):
    w0, alpha, cw = _w0_alpha(fc, q, sr)
    b = np.array([(1 + cw) / 2, -(1 + cw), (1 + cw) / 2])
    a = np.array([1 + alpha, -2 * cw, 1 - alpha])
    return b / a[0], a / a[0]


def bandpass(fc, q=1.0, sr=SR):
    """Constant peak-gain bandpass (0 dB at centre)."""
    w0, alpha, cw = _w0_alpha(fc, q, sr)
    b = np.array([alpha, 0.0, -alpha])
    a = np.array([1 + alpha, -2 * cw, 1 - alpha])
    return b / a[0], a / a[0]


def peaking(fc, gain_db, q=1.0, sr=SR):
    """Bell EQ -- boost or cut a band without touching the rest."""
    w0, alpha, cw = _w0_alpha(fc, q, sr)
    A = 10.0 ** (gain_db / 40.0)
    b = np.array([1 + alpha * A, -2 * cw, 1 - alpha * A])
    a = np.array([1 + alpha / A, -2 * cw, 1 - alpha / A])
    return b / a[0], a / a[0]


def lowshelf(fc, gain_db, s=0.7, sr=SR):
    w0 = 2 * np.pi * float(np.clip(fc, 10, 0.49 * sr)) / sr
    A = 10.0 ** (gain_db / 40.0)
    cw, sw = np.cos(w0), np.sin(w0)
    alpha = sw / 2 * np.sqrt((A + 1 / A) * (1 / s - 1) + 2)
    tsa = 2 * np.sqrt(A) * alpha
    b = np.array([A * ((A + 1) - (A - 1) * cw + tsa),
                  2 * A * ((A - 1) - (A + 1) * cw),
                  A * ((A + 1) - (A - 1) * cw - tsa)])
    a = np.array([(A + 1) + (A - 1) * cw + tsa,
                  -2 * ((A - 1) + (A + 1) * cw),
                  (A + 1) + (A - 1) * cw - tsa])
    return b / a[0], a / a[0]


def highshelf(fc, gain_db, s=0.7, sr=SR):
    w0 = 2 * np.pi * float(np.clip(fc, 10, 0.49 * sr)) / sr
    A = 10.0 ** (gain_db / 40.0)
    cw, sw = np.cos(w0), np.sin(w0)
    alpha = sw / 2 * np.sqrt((A + 1 / A) * (1 / s - 1) + 2)
    tsa = 2 * np.sqrt(A) * alpha
    b = np.array([A * ((A + 1) + (A - 1) * cw + tsa),
                  -2 * A * ((A - 1) + (A + 1) * cw),
                  A * ((A + 1) + (A - 1) * cw - tsa)])
    a = np.array([(A + 1) - (A - 1) * cw + tsa,
                  2 * ((A - 1) - (A + 1) * cw),
                  (A + 1) - (A - 1) * cw - tsa])
    return b / a[0], a / a[0]


# --------------------------------------------------------------------------
# Application helpers
# --------------------------------------------------------------------------

def apply(x, coeffs):
    """Run a biquad over mono or stereo, per channel."""
    b, a = coeffs
    if x.ndim == 1:
        return lfilter(b, a, x)
    return np.stack([lfilter(b, a, x[:, c]) for c in range(x.shape[1])], axis=-1)


def chain(x, *coeff_list):
    """Run several biquads in series (an EQ curve)."""
    y = x
    for c in coeff_list:
        y = apply(y, c)
    return y


def lp24(x, fc, q=0.707, sr=SR):
    """
    24 dB/oct lowpass as two cascaded biquads.

    Only the *second* stage carries the resonance. Splitting Q evenly across
    both stages produces a broad, woolly hump; concentrating it at the end
    gives the sharp, vocal resonant peak that ladder filters are loved for.
    """
    y = apply(x, lowpass(fc, 0.54, sr))
    return apply(y, lowpass(fc, q, sr))


def hp24(x, fc, q=0.707, sr=SR):
    y = apply(x, highpass(fc, 0.54, sr))
    return apply(y, highpass(fc, q, sr))


def bandlimit(x, lo, hi, sr=SR, q=0.707):
    """Keep only the band between lo and hi."""
    return apply(apply(x, highpass(lo, q, sr)), lowpass(hi, q, sr))


# --------------------------------------------------------------------------
# Modulated (swept) filter
# --------------------------------------------------------------------------

def sweep_lowpass(x, fc_curve, q=1.0, sr=SR, block=48, poles=4):
    """
    Lowpass whose cutoff follows `fc_curve` (a per-sample array).

    Coefficients are refreshed every `block` samples and filter state carries
    across block boundaries, so the sweep is continuous with no clicks. This
    is the engine behind every filter-opening build-up in the track.
    """
    n = len(x)
    fc_curve = np.broadcast_to(np.asarray(fc_curve, dtype=float), (n,))
    out = np.empty(n)

    stages = 2 if poles >= 4 else 1
    zis = [np.zeros(2) for _ in range(stages)]

    for i in range(0, n, block):
        j = min(i + block, n)
        fc = float(fc_curve[i])
        seg = x[i:j]

        # Fast path: most channels are silent for most of a track, and a
        # settled filter fed silence outputs silence. Skipping those blocks
        # cuts render time roughly in half without changing a sample.
        if not seg.any() and all(np.all(np.abs(z) < 1e-9) for z in zis):
            out[i:j] = 0.0
            continue

        if stages == 2:
            b, a = lowpass(fc, 0.54, sr)
            seg, zis[0] = lfilter(b, a, seg, zi=zis[0])
            b, a = lowpass(fc, q, sr)
            seg, zis[1] = lfilter(b, a, seg, zi=zis[1])
        else:
            b, a = lowpass(fc, q, sr)
            seg, zis[0] = lfilter(b, a, seg, zi=zis[0])
        out[i:j] = seg

    return out


def sweep_bandpass(x, fc_curve, q=4.0, sr=SR, block=48):
    """Swept resonant bandpass -- used for riser/noise-sweep ear candy."""
    n = len(x)
    fc_curve = np.broadcast_to(np.asarray(fc_curve, dtype=float), (n,))
    out = np.empty(n)
    zi = np.zeros(2)
    for i in range(0, n, block):
        j = min(i + block, n)
        b, a = bandpass(float(fc_curve[i]), q, sr)
        seg, zi = lfilter(b, a, x[i:j], zi=zi)
        out[i:j] = seg
    return out


# --------------------------------------------------------------------------
# Specialised tools
# --------------------------------------------------------------------------

def dc_block(x, sr=SR, fc=18.0):
    """Remove DC offset and sub-sonic rumble. DC offset silently eats
    headroom -- it pushes the waveform off-centre so one side clips early."""
    return apply(x, highpass(fc, 0.707, sr))


def formant(x, vowel="ah", sr=SR, q=9.0, mix=1.0):
    """
    Resonant formant bank -- three parallel bandpasses at the peaks that make
    a human vocal tract sound like a vowel. Applied to a saw it produces the
    "synthetic voice" texture used for the track's ear candy.
    """
    tables = {
        "ah": [(730, 0.0), (1090, -6.0), (2440, -12.0)],
        "ooh": [(300, 0.0), (870, -14.0), (2240, -20.0)],
        "eh": [(530, 0.0), (1840, -8.0), (2480, -14.0)],
        "ee": [(270, 0.0), (2290, -6.0), (3010, -12.0)],
        "uh": [(640, 0.0), (1190, -7.0), (2390, -14.0)],   # schwa -- the tech-house chop
    }
    out = np.zeros_like(x)
    for fc, g in tables[vowel]:
        out += apply(x, bandpass(fc, q, sr)) * 10.0 ** (g / 20.0)
    return mix * out * 1.6 + (1.0 - mix) * x
