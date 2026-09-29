"""
Studio plugins: real VST3 processors, hosted through Spotify's Pedalboard.

The engine's own DSP (engine/dsp) does everything a track needs, but two jobs
are where decades of commercial plugin design pay off and a from-scratch
implementation shows its seams: **algorithmic reverb** (a dense, modulated
tail that never rings or flutters) and **mastering limiting** (a limiter
that shapes gain reduction so a loud master still sounds open). For those,
this module loads free, open-source, professional plugins:

* **Dragonfly Hall / Plate Reverb** (Michael Willis, GPL) -- modulated
  hall and plate algorithms in the tradition of the Lexicon units that
  every deep-house record of the 1990s was mixed through.
* **LSP Limiter / Compressor** (Linux Studio Plugins) -- an oversampled
  look-ahead brick-wall limiter and a feed-forward bus compressor.

Install (Debian/Ubuntu)::

    pip install pedalboard
    apt install dragonfly-reverb-vst3 lsp-plugins-vst3

When either the host or the plugins are missing, every function here falls
back to the engine's built-in processing, so the renderer still runs
anywhere -- `available()` says which path was taken.

Plugins are loaded inside the calling process on first use. The mixer runs
its buses in forked worker processes; a VST instance must never cross a
fork, so nothing is loaded at import time.
"""
import os

import numpy as np

VST3_DIR = os.environ.get("VST3_DIR", "/usr/lib/vst3")
HALL = os.path.join(VST3_DIR, "DragonflyHallReverb.vst3")
PLATE = os.path.join(VST3_DIR, "DragonflyPlateReverb.vst3")
LSP = os.path.join(VST3_DIR, "lsp-plugins.vst3")

_cache = {}


def _pb():
    try:
        import pedalboard  # noqa: F401
        return pedalboard
    except Exception:
        return None


def available():
    """Which plugins can be used here: {'reverb': bool, 'master': bool}."""
    pb = _pb()
    return {
        "reverb": bool(pb) and os.path.exists(HALL) and os.path.exists(PLATE),
        "master": bool(pb) and os.path.exists(LSP),
    }


def _load(path, name=None):
    key = (os.getpid(), path, name)
    if key not in _cache:
        pb = _pb()
        # LSP and DPF print host diagnostics on stderr; they are noise here.
        fd = os.dup(2)
        try:
            with open(os.devnull, "w") as dn:
                os.dup2(dn.fileno(), 2)
                _cache[key] = (pb.load_plugin(path, plugin_name=name) if name
                               else pb.load_plugin(path))
        finally:
            os.dup2(fd, 2)
            os.close(fd)
    return _cache[key]


def _run(plugin, x, sr, tail_s=0.0):
    """Process (n, 2) float audio; returns (n + tail, 2)."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = np.stack([x, x], axis=1)
    if tail_s > 0:
        x = np.concatenate([x, np.zeros((int(tail_s * sr), 2), np.float32)])
    plugin.reset()
    y = plugin.process(x.T.copy(), sample_rate=float(sr), reset=True)
    return np.asarray(y, dtype=np.float64).T


# --------------------------------------------------------------------------
# Reverb
# --------------------------------------------------------------------------

# Room sizes and decays are chosen for 130 BPM: the hall's 2.8 s tail spans
# about six beats, long enough to be a space around the pad rather than an
# echo of it, and the sidechain duck on the bus keeps it out of the kick.
HALL_PARAMS = dict(dry_level=0.0, early_level=12.0, late_level=100.0,
                   size_m=34.0, width=100.0, predelay_ms=28.0, diffuse=92.0,
                   low_cut_hz=180.0, low_cross_hz=450.0, low_mult_x=1.1,
                   high_cut_hz=8200.0, high_cross_hz=4800.0, high_mult_x=0.55,
                   spin_hz=1.4, wander_ms=22.0, decay_s=2.8, early_send=25.0,
                   modulation=35.0)
PLATE_PARAMS = dict(dry_level=0.0, wet_level=100.0, algorithm="Nested",
                    width=100.0, predelay_ms=18.0, decay_s=1.6,
                    low_cut_hz=200.0, high_cut_hz=11000.0, dampen_hz=7500.0)


def _configure(plugin, params):
    for k, v in params.items():
        setattr(plugin, k, v)
    return plugin


def _energy_gain(fn, sr):
    """RMS output/input of a linear processor, measured with white noise."""
    rng = np.random.default_rng(7)
    n = int(1.5 * sr)
    x = rng.standard_normal((n, 2)) * 0.05
    y = fn(x)
    return float(np.sqrt(np.mean(y ** 2)) / np.sqrt(np.mean(x ** 2)))


def reverb(kind, x, sr):
    """
    Wet-only Dragonfly hall or plate, cropped to the input length. The mixer
    scales it to the energy gain of the built-in IR it replaces (see
    `_energy_gain`), so the bus levels keep their meaning.
    """
    path, params = (HALL, HALL_PARAMS) if kind == "hall" else (PLATE, PLATE_PARAMS)
    plugin = _configure(_load(path), params)
    return _run(plugin, x, sr, tail_s=params["decay_s"] * 1.5)[: len(x)]


# --------------------------------------------------------------------------
# Mastering
# --------------------------------------------------------------------------

def glue(x, sr, threshold_db=-16.0, ratio=2.0, attack_ms=30.0,
         release_ms=220.0, knee_db=-8.0):
    """LSP Compressor Stereo as the mix-bus glue: slow attack, 1-2 dB GR."""
    c = _load(LSP, "Compressor Stereo")
    for k, v in dict(ratio=ratio, attack_threshold_db=threshold_db,
                     attack_time_ms=attack_ms, release_time_ms=release_ms,
                     makeup_gain_db=0.0).items():
        if k in c.parameters:
            setattr(c, k, v)
    return _run(c, x, sr)[: len(x)]


def limit(x, sr, ceiling_db=-1.0, lookahead_ms=5.0, release_ms=20.0):
    """
    LSP Limiter Stereo: 'Herm Thin' gain shaping (smooth, fast recovery),
    full 4x oversampling so inter-sample peaks are caught, 5 ms look-ahead.
    The threshold sits 0.3 dB under the ceiling to leave room for the
    reconstruction filter; the caller still verifies true peak.
    """
    lim = _load(LSP, "Limiter Stereo")
    lim.operating_mode = "Herm Thin"
    lim.oversampling = "Full x4/24 bit"
    lim.dithering = "None"
    lim.threshold_db = round(ceiling_db - 0.3, 2)
    lim.lookahead_ms = lookahead_ms
    lim.release_time_ms = min(release_ms, 20.0)   # LSP's range is 0.25-20 ms
    lim.automatic_level_regulation = False
    lim.gain_boost = False          # else LSP makes up the threshold back to 0 dBFS
    lim.input_gain_db = 0.0
    lim.output_gain_db = 0.0
    # Look-ahead and the oversampling filters delay the signal; measure the
    # delay with a quiet chirp (well under threshold, so the limiter is
    # linear) and remove it, so the master stays sample-locked to the grid.
    key = ("lat", os.getpid(), sr, lookahead_ms, lim.oversampling)
    if key not in _cache:
        n = int(0.5 * sr)
        t = np.arange(n) / sr
        ch = 0.05 * np.sin(2 * np.pi * (200.0 * t + 4000.0 * t ** 2))
        ch *= np.hanning(n)
        probe = np.stack([ch, ch], axis=1)
        out = _run(lim, probe, sr, tail_s=0.1)[:, 0]
        xc = np.correlate(out, ch, mode="full")[n - 1:]
        _cache[key] = int(np.argmax(np.abs(xc)))
    lat = _cache[key]
    y = _run(lim, x, sr, tail_s=(lat + 64) / sr)
    return y[lat: lat + len(x)]
