"""
Instrument voices.

Each function synthesises one note or hit and returns a raw buffer. Nothing
here knows about tempo or arrangement -- that is the sequencer's job. Mixing,
sends and spatial placement happen later too, so these stay dry and centred
unless the sound is inherently stereo (pads, stabs).

The sound-design decisions are commented inline, because in a track like this
the *design* of the kick and bass matters far more than the notes they play.
"""

import numpy as np

from dsp.core import (SR, sine, saw, square, noise, supersaw, phasor,
                      perc_env, adsr, fade, cents, midi_to_hz, db, stereo)
from dsp import filters as F
from dsp import dynamics as D
from dsp import space as S


# ==========================================================================
# Drums
# ==========================================================================

def kick(sr=SR, dur=0.60, f_start=168.0, f_end=47.0, pitch_tau=0.023,
         decay=0.33, drive=2.1, click=1.0, seed=11):
    """
    Club kick: pitch-swept sine body + transient click, then saturated.

    The pitch envelope is what makes it a kick rather than a bass note. It
    starts near 170 Hz and falls to ~47 Hz in about 25 ms. Our ears read that
    fast downward sweep as a physical impact; hold the pitch steady and you
    get a boop instead of a thump.

    Saturation matters just as much. It generates harmonics at 94, 141, 188 Hz
    from the 47 Hz fundamental, so the kick is still clearly audible on phone
    and laptop speakers that cannot reproduce 47 Hz at all -- the brain
    reconstructs the missing fundamental from the harmonic series.
    """
    n = int(dur * sr)
    t = np.arange(n) / sr

    f = f_end + (f_start - f_end) * np.exp(-t / pitch_tau)
    body = sine(f, n, sr)

    # Two stacked amplitude envelopes: a long one for weight, a very short one
    # for the initial punch that gives the kick its front-of-speaker impact.
    weight = perc_env(n, sr, attack=0.0008, decay=decay, curve=4.2)
    punch = perc_env(n, sr, attack=0.0004, decay=0.042, curve=5.5)
    body *= (weight + 0.45 * punch)

    # Click: a few milliseconds of bright noise plus a high sine tick. This is
    # the part that cuts through a loud room and defines the exact downbeat.
    clk = F.hp24(noise(n, seed=seed), 1900, 0.8, sr)
    clk *= perc_env(n, sr, attack=0.0002, decay=0.0045, curve=9.0)
    tick = sine(1350.0, n, sr) * perc_env(n, sr, 0.0002, 0.0032, 10.0)

    out = body + (clk * 0.30 + tick * 0.22) * click
    out = D.saturate(out, drive, "tube", sr, oversample=4)
    out = F.chain(out,
                  F.highpass(27.0, 0.707, sr),    # kill sub-sonic rumble
                  F.peaking(58.0, 2.2, 1.0),      # weight
                  F.peaking(340.0, -3.4, 1.1),    # clear the boxy mud zone
                  F.peaking(3400.0, 2.0, 1.2))    # let the click through
    return fade(out, sr, 0.0002, 0.01)


def clap(sr=SR, seed=23, bright=1.0):
    """
    Layered clap.

    A real clap is many hands landing a few milliseconds apart, not one burst.
    Four short noise bursts spaced 9-11 ms reproduce that scatter, and a longer
    filtered tail underneath gives the room sound that glues them together.
    """
    n = int(0.45 * sr)
    out = np.zeros(n)

    for k, (off, g) in enumerate([(0.0, 0.80), (0.0095, 1.0),
                                  (0.0185, 0.92), (0.0275, 0.68)]):
        i = int(off * sr)
        ln = n - i
        burst = noise(ln, seed=seed + k) * perc_env(ln, sr, 0.0004, 0.011, 7.0)
        out[i:] += burst * g

    # body/tail -- the "room" of the clap
    out += noise(n, seed=seed + 50) * perc_env(n, sr, 0.002, 0.155, 3.2) * 0.40

    out = F.bandlimit(out, 900.0, 9000.0 * bright, sr)
    out = F.chain(out,
                  F.peaking(1650.0, 4.0, 1.1),    # hand "slap"
                  F.peaking(3900.0, 3.2, 1.4),    # snap / air
                  F.peaking(600.0, -3.0, 1.0))    # de-honk
    return fade(out, sr, 0.0003, 0.02)


def snare(sr=SR, dur=0.24, tune=185.0, seed=31, snap=1.0):
    """Snare for build-up rolls: tonal shell + noise wires."""
    n = int(dur * sr)
    shell = (sine(tune, n, sr) + sine(tune * 1.48, n, sr) * 0.6) * \
        perc_env(n, sr, 0.0005, 0.075, 4.5)
    wires = F.bandlimit(noise(n, seed=seed), 1500.0, 10000.0, sr) * \
        perc_env(n, sr, 0.0004, 0.10, 3.6)
    out = shell * 0.55 + wires * 0.75 * snap
    out = D.saturate(out, 1.5, "tanh", sr, oversample=2)
    out = F.chain(out, F.highpass(150.0, 0.707, sr), F.peaking(3200.0, 3.0, 1.3))
    return fade(out, sr, 0.0003, 0.012)


def hihat(sr=SR, dur=0.055, tone=1.0, seed=41, open_hat=False):
    """
    Hi-hat: inharmonic metal + noise.

    The 808 trick -- six square waves at deliberately non-integer frequency
    ratios. Integer ratios would fuse into a pitched tone; these ratios stay
    metallic and unpitched, which is exactly what a cymbal is.
    """
    n = int(dur * sr)
    ratios = [2.0, 3.0, 4.16, 5.43, 6.79, 8.21]
    base = 318.0 * tone

    metal = np.zeros(n)
    for r in ratios:
        metal += square(base * r, n, sr)
    metal /= len(ratios)

    mix = metal * 0.50 + noise(n, seed=seed) * 0.50
    mix = F.hp24(mix, 6800.0 if not open_hat else 6200.0, 0.8, sr)
    mix = F.apply(mix, F.lowpass(15500.0, 0.707, sr))

    env = perc_env(n, sr, 0.0003, dur * (0.55 if not open_hat else 0.42), 
                   4.0 if not open_hat else 2.2)
    return fade(mix * env, sr, 0.0002, 0.006)


def shaker(sr=SR, dur=0.09, seed=57):
    """Shaker: narrow band of noise with a soft attack -- fills the 16ths
    without competing with the hats for the same frequency slot."""
    n = int(dur * sr)
    x = F.bandlimit(noise(n, seed=seed), 4200.0, 11000.0, sr)
    return fade(x * perc_env(n, sr, 0.004, dur * 0.4, 3.0), sr, 0.002, 0.008)


def rim(sr=SR, dur=0.10, tune=420.0, seed=61):
    """Rimshot/click percussion for off-grid groove accents."""
    n = int(dur * sr)
    tone = (sine(tune, n, sr) + square(tune * 2.31, n, sr) * 0.35) * \
        perc_env(n, sr, 0.0003, 0.016, 7.0)
    nz = F.bandlimit(noise(n, seed=seed), 1800.0, 8000.0, sr) * \
        perc_env(n, sr, 0.0002, 0.010, 8.0)
    out = tone * 0.7 + nz * 0.5
    return fade(F.apply(out, F.highpass(300.0, 0.707, sr)), sr, 0.0002, 0.008)


def tom(sr=SR, dur=0.42, f_start=210.0, f_end=95.0, seed=67):
    """Pitched tom for fills."""
    n = int(dur * sr)
    t = np.arange(n) / sr
    f = f_end + (f_start - f_end) * np.exp(-t / 0.055)
    out = sine(f, n, sr) * perc_env(n, sr, 0.001, 0.20, 3.8)
    out += F.bandlimit(noise(n, seed=seed), 400.0, 5000.0, sr) * \
        perc_env(n, sr, 0.0005, 0.02, 6.0) * 0.25
    return fade(D.saturate(out, 1.4, "tanh", sr, oversample=2), sr, 0.0005, 0.015)


def crash(sr=SR, dur=2.6, seed=71, bright=1.0):
    """Crash cymbal: dense bright noise with a long, slightly metallic decay."""
    n = int(dur * sr)
    metal = np.zeros(n)
    for r in [1.0, 1.41, 1.73, 2.19, 2.74, 3.37, 4.11]:
        metal += square(410.0 * r, n, sr)
    x = metal / 7.0 * 0.35 + noise(n, seed=seed) * 0.65
    x = F.bandlimit(x, 700.0 * bright, 16000.0, sr)
    env = perc_env(n, sr, 0.001, dur * 0.42, 2.4)
    return fade(x * env, sr, 0.001, 0.05)


# ==========================================================================
# Bass
# ==========================================================================

def bass(freq, dur, sr=SR, cutoff=420.0, res=1.7, env_amount=2.4,
         sub=1.0, saw_level=0.55, drive=1.6, decay=0.09, seed=83,
         release=0.045):
    """
    Two-layer house bass: pure sine sub + filtered detuned saws.

    Splitting the bass in two is the standard approach because the two layers
    have different jobs. The sine owns 40-80 Hz -- that is the physical weight
    you feel in a club, and it must stay clean, so it gets no filtering and no
    detune. The saw layer lives above 150 Hz and carries the note's *identity*
    on speakers too small to reproduce the sub at all.

    The filter envelope (cutoff opening briefly on each note) is what gives the
    layer its forward, plucked attack instead of a flat drone.
    """
    n = int(dur * sr)
    rng = np.random.default_rng(seed)

    amp = adsr(n, sr, a=0.004, d=decay, s=0.72, r=release, curve=2.2)

    sub_layer = sine(freq, n, sr, phase0=0.0)
    # a touch of second harmonic keeps the sub audible on small speakers
    sub_layer += sine(freq * 2.0, n, sr) * 0.14

    s = (saw(freq * cents(-7.0), n, sr, phase0=rng.random()) +
         saw(freq * cents(+8.0), n, sr, phase0=rng.random())) * 0.5

    fenv = cutoff * (1.0 + env_amount * perc_env(n, sr, 0.002, 0.07, 4.0))
    s = F.sweep_lowpass(s, np.clip(fenv, 60.0, 0.45 * sr), res, sr, poles=4)
    s = F.apply(s, F.highpass(90.0, 0.707, sr))   # keep saws out of the sub's lane

    out = (sub_layer * sub + s * saw_level) * amp
    out = D.saturate(out, drive, "tanh", sr, oversample=4)
    out = F.apply(out, F.highpass(28.0, 0.707, sr))
    return fade(out, sr, 0.003, 0.012)


def reese(freq, dur, sr=SR, cutoff=700.0, detune=22.0, seed=89):
    """Wide detuned bass for breakdown moments. Two saws beating against each
    other; the phase interference creates the slow movement."""
    n = int(dur * sr)
    rng = np.random.default_rng(seed)
    a = saw(freq * cents(-detune), n, sr, phase0=rng.random())
    b = saw(freq * cents(+detune), n, sr, phase0=rng.random())
    env = adsr(n, sr, a=0.02, d=0.2, s=0.8, r=0.1, curve=2.0)
    left = F.lp24(a * 0.6 + b * 0.4, cutoff, 1.2, sr) * env
    right = F.lp24(a * 0.4 + b * 0.6, cutoff, 1.2, sr) * env
    return stereo(left, right)


# ==========================================================================
# Harmony
# ==========================================================================

def stab(freqs, dur, sr=SR, cutoff=2400.0, res=1.35, decay=0.22,
         detune=11.0, voices=3, spread=0.85, drive=1.25, seed=97,
         attack=0.004, sustain=0.25):
    """
    Plucked chord stab -- the signature house chord sound.

    Every note of the chord gets its own detuned saw stack, and each stack is
    spread across the stereo field. The filter envelope snaps open for ~60 ms
    then closes, which is what turns a sustained pad into a percussive stab
    that can sit on offbeats without cluttering the groove.
    """
    n = int(dur * sr)
    left = np.zeros(n)
    right = np.zeros(n)

    for i, f in enumerate(freqs):
        l, r = supersaw(f, n, sr, voices=voices, detune_cents=detune,
                        spread=spread, seed=seed + i * 13)
        left += l
        right += r
    left /= np.sqrt(len(freqs))
    right /= np.sqrt(len(freqs))

    fenv = cutoff * (0.30 + 1.0 * perc_env(n, sr, 0.003, 0.055, 4.5))
    fenv = np.clip(fenv, 120.0, 0.45 * sr)
    left = F.sweep_lowpass(left, fenv, res, sr, poles=4)
    right = F.sweep_lowpass(right, fenv, res, sr, poles=4)

    amp = adsr(n, sr, a=attack, d=decay, s=sustain, r=min(0.25, dur * 0.4), curve=2.6)
    out = stereo(left * amp, right * amp)
    out = D.saturate(out, drive, "tanh", sr, oversample=2)
    # Chords never need energy below ~150 Hz -- that space belongs to kick+bass.
    out = F.apply(out, F.highpass(165.0, 0.707, sr))
    return np.stack([fade(out[:, 0], sr), fade(out[:, 1], sr)], axis=-1)


def pad(freqs, dur, sr=SR, cutoff=1500.0, attack=0.9, release=1.4,
        detune=16.0, voices=7, seed=101, sub_level=0.0):
    """
    Wide evolving pad. Slow attack, heavy detune, gentle filter.

    A slow LFO drifts the cutoff so the texture never sits still -- static pads
    are the fastest way to make a mix sound synthetic.
    """
    n = int(dur * sr)
    t = np.arange(n) / sr
    left = np.zeros(n)
    right = np.zeros(n)

    for i, f in enumerate(freqs):
        l, r = supersaw(f, n, sr, voices=voices, detune_cents=detune,
                        spread=1.0, seed=seed + i * 7)
        left += l
        right += r
        if sub_level > 0 and i == 0:
            s = sine(f * 0.5, n, sr) * sub_level
            left += s
            right += s
    left /= np.sqrt(len(freqs))
    right /= np.sqrt(len(freqs))

    lfo = cutoff * (1.0 + 0.35 * np.sin(2 * np.pi * 0.07 * t))
    left = F.sweep_lowpass(left, lfo, 0.9, sr, block=256, poles=4)
    right = F.sweep_lowpass(right, lfo * 1.03, 0.9, sr, block=256, poles=4)

    amp = adsr(n, sr, a=attack, d=0.4, s=0.85, r=release, curve=1.8)
    out = stereo(left * amp, right * amp)
    out = S.chorus(out, sr, rate=0.19, depth_ms=6.5, base_ms=17.0, mix=0.45)
    out = F.apply(out, F.highpass(120.0, 0.707, sr))
    return out


def keys(freqs, dur, sr=SR, decay=0.9, seed=113, bright=1.0):
    """Electric-piano-ish tine using phase modulation -- warmer and less
    aggressive than saws, good for breakdowns."""
    n = int(dur * sr)
    out_l = np.zeros(n)
    out_r = np.zeros(n)
    rng = np.random.default_rng(seed)

    for i, f in enumerate(freqs):
        ph, _ = phasor(f, n, sr, phase0=rng.random())
        mod = sine(f * 3.0, n, sr) * 2.1 * perc_env(n, sr, 0.001, 0.05 * bright, 5.0)
        tone = np.sin(2 * np.pi * ph + mod)
        tone *= perc_env(n, sr, 0.003, decay, 2.6)
        p = -0.5 + i / max(1, len(freqs) - 1)
        out_l += tone * np.cos((p + 1) * np.pi / 4)
        out_r += tone * np.sin((p + 1) * np.pi / 4)

    g = 1.0 / np.sqrt(len(freqs))
    return stereo(fade(out_l * g, sr), fade(out_r * g, sr))


# ==========================================================================
# Ear candy
# ==========================================================================

def bell(freq, dur, sr=SR, ratio=2.01, index=3.2, decay=0.42, seed=127):
    """
    2-operator FM bell.

    One sine modulates another's phase. The modulator's own decay envelope
    means the tone is bright at the attack and mellows as it rings -- the same
    behaviour as a struck physical object, and the reason FM bells sound alive
    where a filtered saw sounds static.
    """
    n = int(dur * sr)
    rng = np.random.default_rng(seed)
    ph, _ = phasor(freq, n, sr, phase0=rng.random())
    mod = sine(freq * ratio, n, sr) * index * perc_env(n, sr, 0.0008, decay * 0.35, 4.0)
    out = np.sin(2 * np.pi * ph + mod) * perc_env(n, sr, 0.001, decay, 3.2)
    return fade(F.apply(out, F.highpass(250.0, 0.707, sr)), sr, 0.001, 0.015)


def pluck(freq, dur, sr=SR, decay=0.30, cutoff=3200.0, seed=131):
    """Short filtered saw pluck for the arpeggio line."""
    n = int(dur * sr)
    rng = np.random.default_rng(seed)
    x = saw(freq, n, sr, phase0=rng.random())
    fenv = cutoff * (0.22 + 1.0 * perc_env(n, sr, 0.001, 0.05, 5.0))
    x = F.sweep_lowpass(x, np.clip(fenv, 150.0, 0.45 * sr), 2.2, sr, poles=4)
    x *= perc_env(n, sr, 0.002, decay, 3.0)
    return fade(F.apply(x, F.highpass(200.0, 0.707, sr)), sr, 0.001, 0.012)


def vox_chop(freq, dur, sr=SR, vowel="ah", seed=137, decay=0.5):
    """
    Synthetic vocal texture: a saw pushed through a formant filter bank.

    The three resonant peaks imitate a vocal tract shaped for a vowel. It is
    not a convincing voice in isolation, but buried in a mix with reverb and
    delay it reads as a human element, which is what makes a house track feel
    warm rather than purely mechanical.
    """
    n = int(dur * sr)
    rng = np.random.default_rng(seed)
    src = (saw(freq, n, sr, phase0=rng.random()) * 0.7 +
           saw(freq * cents(6.0), n, sr, phase0=rng.random()) * 0.3)
    v = F.formant(src, vowel, sr, q=11.0)
    # a little breath noise sells the illusion
    v += F.bandlimit(noise(n, seed=seed), 2000.0, 7000.0, sr) * 0.05
    env = adsr(n, sr, a=0.03, d=decay, s=0.45, r=min(0.3, dur * 0.4), curve=2.2)
    out = v * env
    out = F.chain(out, F.highpass(180.0, 0.707, sr), F.peaking(2800.0, 2.5, 1.2))
    return fade(out, sr, 0.004, 0.02)


def riser(dur, sr=SR, f0=220.0, f1=9000.0, seed=149, tonal=0.35, curve=1.7):
    """
    Build-up riser: a noise band sweeping upward plus a rising saw.

    Both the frequency sweep and the volume ramp are exponential. Linear ramps
    feel like they stall halfway, because pitch and loudness perception are
    both logarithmic -- an exponential sweep reads as constant acceleration.
    """
    n = int(dur * sr)
    t = np.arange(n) / sr
    prog = (t / dur) ** curve

    sweep = f0 * (f1 / f0) ** prog
    nz = F.sweep_bandpass(noise(n, seed=seed), sweep, q=3.2, sr=sr) * 2.2

    tone = saw(f0 * 0.5 * (f1 / f0 / 3.0) ** prog, n, sr) * tonal
    tone = F.sweep_lowpass(tone, np.clip(sweep * 1.6, 100, 0.45 * sr), 1.4, sr)

    env = prog ** 0.85
    out = (nz + tone) * env
    return fade(F.apply(out, F.highpass(200.0, 0.707, sr)), sr, 0.01, 0.006)


def downlifter(dur, sr=SR, f0=6000.0, f1=140.0, seed=151):
    """Falling sweep to mark a section ending -- the mirror of the riser."""
    n = int(dur * sr)
    t = np.arange(n) / sr
    prog = (t / dur) ** 0.6
    sweep = f0 * (f1 / f0) ** prog
    x = F.sweep_bandpass(noise(n, seed=seed), sweep, q=2.6, sr=sr) * 2.0
    env = (1.0 - prog) ** 1.2
    return fade(x * env, sr, 0.005, 0.03)


def reverse_crash(dur=1.9, sr=SR, seed=157):
    """Reversed cymbal -- swells into the downbeat it precedes."""
    c = crash(sr, dur, seed=seed)
    return fade(c[::-1].copy(), sr, 0.02, 0.004)


def sub_drop(dur=1.4, sr=SR, f0=110.0, f1=32.0, seed=163):
    """Deep pitch-falling sine to underline a drop."""
    n = int(dur * sr)
    t = np.arange(n) / sr
    f = f1 + (f0 - f1) * np.exp(-t / (dur * 0.28))
    env = np.exp(-t / (dur * 0.45))
    out = sine(f, n, sr) * env
    return fade(F.apply(out, F.highpass(26.0, 0.707, sr)), sr, 0.008, 0.06)


def noise_sweep(dur, sr=SR, up=True, seed=167, lo=300.0, hi=12000.0):
    """Broad filtered-noise swoosh used as a transition texture."""
    n = int(dur * sr)
    t = np.arange(n) / sr
    prog = t / dur
    fc = lo * (hi / lo) ** (prog if up else (1.0 - prog))
    x = F.sweep_lowpass(noise(n, seed=seed), fc, 1.1, sr)
    x = F.apply(x, F.highpass(250.0, 0.707, sr))
    env = np.sin(np.pi * prog) ** 1.3
    return fade(x * env, sr, 0.01, 0.02)
