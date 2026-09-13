"""
Dynamics: compression, sidechain ducking, saturation, and limiting.

Three ideas do most of the work here:

1. **Log-domain ballistics.** Gain reduction is smoothed in decibels, not in
   linear amplitude. Our hearing is logarithmic, so dB-domain attack/release
   produces the even, musical "breathing" of a good hardware compressor;
   linear smoothing sounds lumpy and grabby.

2. **Control-rate detection.** The envelope follower runs once per small block
   and the resulting gain curve is interpolated back to audio rate and
   smoothed. That removes the only genuinely serial part of the algorithm
   without changing the sound, and it is what real plugins do internally.

3. **Oversampled saturation.** Distortion creates new harmonics above the
   original signal. At 48 kHz many of those land past Nyquist and fold back as
   dissonant aliasing. Running the non-linearity at 4x rate and filtering on
   the way back down keeps the warmth and drops the grit.
"""

import numpy as np
from scipy.signal import resample_poly
from scipy.ndimage import minimum_filter1d

from .core import SR, db, to_db
from . import filters as F


# --------------------------------------------------------------------------
# Detection helpers
# --------------------------------------------------------------------------

def _block_peak(x, block):
    """Peak magnitude per block. Pads to a whole number of blocks."""
    n = len(x)
    nb = int(np.ceil(n / block))
    pad = nb * block - n
    xp = np.abs(np.concatenate([x, np.zeros(pad)]))
    return xp.reshape(nb, block).max(axis=1), nb


def _true_block_peak(y, block, sr, oversample=4):
    """
    Per-block peak measured on an oversampled copy of the signal.

    A digital sample stream only stores points on the waveform; the analog
    signal a converter reconstructs between those points can be higher than
    any stored sample. Those inter-sample (true) peaks are what clip a DAC or
    an MP3 decoder even when the file measures below 0 dBFS. Detecting on a
    4x-upsampled copy sees them; detecting on the raw samples does not.
    """
    n = len(y)
    nb = int(np.ceil(n / block))
    peaks = np.zeros(nb)
    for c in range(y.shape[1]):
        up = resample_poly(y[:, c].astype(np.float32), oversample, 1)
        bp, _ = _block_peak(up, block * oversample)
        if len(bp) < nb:
            bp = np.concatenate([bp, np.zeros(nb - len(bp))])
        peaks = np.maximum(peaks, bp[:nb])
    return peaks


def _expand(curve, block, n):
    """Block-rate curve -> sample rate, linearly interpolated."""
    src = np.arange(len(curve)) * block + block * 0.5
    return np.interp(np.arange(n), src, curve)


def _win_convolve(x, win):
    """
    Convolve with a normalised window, padding with *edge values* rather than
    zeros.

    This matters more than it looks. `np.convolve(..., mode="same")` pads with
    zeros, so a control signal sitting at 1.0 gets dragged toward 0 at both
    ends of the buffer -- which in a limiter reads as "turn the first few
    milliseconds of the track down by 6 dB". Edge padding keeps the curve flat
    where the signal is flat.
    """
    r = len(win) // 2
    padded = np.concatenate([np.full(r, x[0]), x, np.full(r, x[-1])])
    return np.convolve(padded, win, mode="same")[r:r + len(x)]


def _smooth(x, sr, ms):
    """Raised-cosine moving average -- removes stair-steps from an
    interpolated control signal so gain changes never zipper."""
    w = max(3, int(sr * ms / 1000.0))
    win = np.hanning(w)
    win /= win.sum()
    return _win_convolve(x, win)


def _gain_computer(level_db, threshold, ratio, knee):
    """
    Static compression curve with a soft knee, in dB.
    Returns the gain change (<= 0) to apply at each level.
    """
    over = level_db - threshold
    slope = 1.0 / ratio - 1.0
    below = over < -knee / 2
    above = over > knee / 2
    inside = ~(below | above)

    g = np.zeros_like(over)
    g[above] = slope * over[above]
    if knee > 0:
        k = over[inside] + knee / 2
        g[inside] = slope * (k * k) / (2 * knee)
    return g


# --------------------------------------------------------------------------
# Compressor
# --------------------------------------------------------------------------

def compress(x, sr=SR, threshold=-18.0, ratio=3.0, attack=0.010,
             release=0.150, knee=6.0, makeup="auto", mix=1.0,
             sidechain=None, block=16, hpf_sidechain=None):
    """
    Stereo-linked feed-forward compressor.

    Stereo linking (one gain curve derived from both channels) is essential:
    independent per-channel compression makes the stereo image lurch sideways
    every time one side is louder, which is instantly audible on headphones.

    `mix` enables parallel / "New York" compression -- blending the squashed
    signal under the dry one adds density and sustain while leaving the
    original transients intact.
    """
    mono_in = x.ndim == 1
    y = x[:, None] if mono_in else x
    n = len(y)

    det = np.max(np.abs(y), axis=1) if sidechain is None else np.abs(
        sidechain if sidechain.ndim == 1 else np.max(np.abs(sidechain), axis=1))

    if hpf_sidechain:
        # Keep bass energy from triggering the whole mix down -- a standard
        # trick so kick drums don't pump the entire compressor.
        det = np.abs(F.apply(det, F.highpass(hpf_sidechain, 0.707, sr)))

    peaks, nb = _block_peak(det, block)
    level_db = to_db(peaks)
    target = _gain_computer(level_db, threshold, ratio, knee)

    # dB-domain ballistics at control rate
    bt = block / sr
    a_coef = np.exp(-bt / max(attack, 1e-5))
    r_coef = np.exp(-bt / max(release, 1e-5))
    gr = np.empty(nb)
    g = 0.0
    for i in range(nb):
        t = target[i]
        c = a_coef if t < g else r_coef      # moving down = attack
        g = c * g + (1.0 - c) * t
        gr[i] = g

    gain = db(_smooth(_expand(gr, block, n), sr, 1000.0 * block / sr * 1.5))

    if makeup == "auto":
        # Restore roughly what the threshold/ratio pair removes at the top.
        mk = db(-_gain_computer(np.array([0.0]), threshold, ratio, knee)[0] * 0.65)
    else:
        mk = db(makeup)

    wet = y * gain[:, None] * mk
    out = mix * wet + (1.0 - mix) * y
    return out[:, 0] if mono_in else out


# --------------------------------------------------------------------------
# Sidechain ducking
# --------------------------------------------------------------------------

def duck_envelope(n, trigger_samples, sr=SR, depth=0.75, hold=0.012,
                  release=0.24, curve=1.7):
    """
    The house "pump", generated directly from the kick grid rather than
    detected from audio.

    Producers reach for a volume-shaper plugin instead of a real compressor
    here for a reason: the curve is identical every bar, perfectly aligned to
    the beat, and free of the detector overshoot that smears the first few
    milliseconds. That rhythmic certainty is what makes the groove lock.

    Returns a gain curve in [1-depth, 1].
    """
    env = np.ones(n)
    nh = int(hold * sr)
    nr = max(1, int(release * sr))
    # one dip shape, reused at every trigger
    shape = np.concatenate([
        np.zeros(nh),                                   # held fully down
        1.0 - (1.0 - np.linspace(0, 1, nr)) ** curve,   # smooth recovery
    ])
    dip = 1.0 - depth * (1.0 - shape)

    for t in trigger_samples:
        t = int(t)
        if t >= n:
            continue
        end = min(t + len(dip), n)
        seg = dip[:end - t]
        # multiply, so overlapping triggers don't stack into silence
        env[t:end] = np.minimum(env[t:end], seg)

    return _smooth(env, sr, 3.0)


# --------------------------------------------------------------------------
# Saturation
# --------------------------------------------------------------------------

def saturate(x, drive=1.5, mode="tanh", sr=SR, oversample=4, mix=1.0):
    """
    Waveshaping distortion with anti-aliasing via oversampling.

    - "tanh"  : symmetric soft clip, adds odd harmonics -> warm, glue-like.
    - "tube"  : asymmetric, adds even harmonics too -> richer, more "analog".
    - "fold"  : wavefolding, aggressive and bright -> texture, not glue.
    """
    mono_in = x.ndim == 1
    y = x[:, None] if mono_in else x
    out = np.empty_like(y)

    for c in range(y.shape[1]):
        ch = y[:, c].astype(np.float32)
        up = resample_poly(ch, oversample, 1).astype(np.float32) if oversample > 1 else ch
        d = up * drive

        if mode == "tanh":
            w = np.tanh(d)
        elif mode == "tube":
            # asymmetric transfer: compresses positive half harder
            w = np.where(d >= 0, np.tanh(d), np.tanh(d * 0.72) * 0.88)
        elif mode == "fold":
            w = np.sin(d * 1.35)
        else:
            raise ValueError(mode)

        w = resample_poly(w, 1, oversample) if oversample > 1 else w
        w = w[:len(ch)] if len(w) >= len(ch) else np.pad(w, (0, len(ch) - len(w)))
        # normalise so drive changes tone, not level
        out[:, c] = w.astype(float) / np.tanh(drive) if mode != "fold" else w.astype(float)

    out = mix * out + (1.0 - mix) * y
    return out[:, 0] if mono_in else out


# --------------------------------------------------------------------------
# Limiting
# --------------------------------------------------------------------------

def limit(x, sr=SR, ceiling_db=-1.0, lookahead=0.005, release=0.070,
          block=16, true_peak=True):
    """
    Look-ahead brickwall limiter.

    The gain curve is built in three stages, each of which can only ever lower
    the gain relative to what the signal requires:

      1. `need`  -- the instantaneous gain that would put this block exactly at
                    the ceiling (1.0 where the signal is already below it).
      2. a running minimum over the look-ahead window, so the gain is already
         down *before* a transient arrives rather than chasing it.
      3. dB-free release ballistics: drop instantly, recover slowly, so bass
         notes are not modulated at their own frequency (which is what makes a
         cheap limiter sound like it is farting).

    Finally the curve is smoothed with a Hann window. Because the running
    minimum's radius is deliberately wider than the smoothing radius, every
    value averaged into a smoothed sample is already <= that sample's own
    requirement -- so smoothing can never reintroduce an overshoot. No clipping
    stage is needed to catch it.

    With `true_peak` enabled the detector measures an oversampled copy, so the
    ceiling is honoured in dBTP (true peak) rather than dBFS. This is the
    difference between a master that survives MP3 encoding and one that
    crackles on playback despite measuring clean as a WAV.
    """
    mono_in = x.ndim == 1
    y = x[:, None] if mono_in else x
    n = len(y)

    ceiling = db(ceiling_db)
    if true_peak:
        peak = _true_block_peak(y, block, sr)
        nb = len(peak)
    else:
        peak, nb = _block_peak(np.max(np.abs(y), axis=1), block)

    need = np.ones(nb)
    hot = peak > ceiling
    need[hot] = ceiling / peak[hot]

    # --- stage 2: look-ahead running minimum ------------------------------
    la = max(1, int(lookahead * sr))
    la_blocks = int(np.ceil(la / block)) + 1
    need = minimum_filter1d(need, size=2 * la_blocks + 1, mode="nearest")

    # --- stage 3: instant attack, exponential release ---------------------
    r_coef = np.exp(-(block / sr) / max(release, 1e-4))
    gr = np.empty(nb)
    g = 1.0
    for i in range(nb):
        t = need[i]
        g = t if t < g else r_coef * g + (1.0 - r_coef) * t
        gr[i] = g

    # --- smooth, with a radius smaller than the look-ahead radius ---------
    gain = _expand(gr, block, n)
    win = np.hanning(2 * la + 1)
    win /= win.sum()
    gain = _win_convolve(gain, win)

    out = y * np.minimum(gain, 1.0)[:, None]
    return out[:, 0] if mono_in else out


def normalize(x, peak_db=-1.0):
    """Scale so the loudest sample sits at `peak_db`."""
    p = float(np.max(np.abs(x)))
    return x if p == 0 else x * (db(peak_db) / p)


def exciter(x, sr=SR, band=(900.0, 3500.0), keep_above=2600.0,
            drive=3.0, mix=0.5, mode="tanh"):
    """
    Aphex-style harmonic exciter: generate new high harmonics from the band
    below them, then blend only the new content back in.

    This exists because opening a lowpass and adding presence are not the same
    operation. A filter can only reveal harmonics the oscillator already
    produced; if the source is genuinely band-limited -- a supersaw stack
    filtered at 1.5 kHz, say -- there is nothing above the cutoff to uncover,
    and a shelving boost on the master only amplifies noise.

    The order of operations is what makes it work, and getting it wrong makes
    the effect useless. Saturating a *high-passed* copy produces almost nothing
    when the source is dark, because the high-pass has already thrown away
    everything that could have been distorted. Instead:

      1. band-pass the region that still has energy (`band`),
      2. saturate *that* -- a 1.2 kHz partial breeds new ones at 2.4 and
         3.6 kHz, exactly where the presence is missing,
      3. high-pass the result at `keep_above` so only the newly created
         harmonics survive, not a second copy of the source,
      4. blend.

    The saturation runs oversampled; without it the new harmonics -- which by
    construction sit near the top of the band -- would alias back down into the
    midrange as inharmonic tones.
    """
    lo, hi = band
    src = F.apply(x, F.highpass(lo, 0.707, sr))
    src = F.apply(src, F.lowpass(hi, 0.707, sr))

    # Normalise into the shaper. A waveshaper is only nonlinear near full
    # scale: tanh(0.02) is 0.02 to four decimal places, so feeding it a quiet
    # band -- which an isolated 1-3 kHz slice of a mix always is -- produces no
    # harmonics at all regardless of the drive setting. Scaling to unity first
    # makes `drive` mean the same thing whatever the source level, and the
    # original peak is restored on the way out.
    peak = float(np.max(np.abs(src)))
    if peak < 1e-9:
        return x
    har = saturate(src / peak, drive, mode, sr, oversample=4) * peak
    har = F.hp24(har, keep_above, 0.707, sr)
    return x + har * mix


def tilt(x, pivot=700.0, slope_db=3.0, sr=SR):
    """
    Broadband spectral tilt: shelve the top up and the bottom down by the same
    amount around a pivot, so the overall level barely moves.

    A tilt is the right tool for "too dark" or "too bright" as a whole. Two
    opposing shelves keep the correction gentle and phase-coherent across the
    whole spectrum, where a single large shelf would pile the entire change
    into one end and change the loudness with it.
    """
    y = F.apply(x, F.lowshelf(pivot, -slope_db, 0.5, sr))
    return F.apply(y, F.highshelf(pivot, slope_db, 0.5, sr))
