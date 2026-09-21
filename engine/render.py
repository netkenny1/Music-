"""
Renderer: sequences the composition into the mixer and produces the master.

Run with:  python3 engine/render.py [--out DIR] [--quiet]
"""

import argparse
import os
import sys
import time
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dsp.core import SR, midi_to_hz, db, stereo
from dsp import filters as F
from dsp import dynamics as D
from dsp import space as S
import instruments as I
import composition as C
import groove as G
from mixer import Mixer, master_chain
import analysis as A


# ==========================================================================
# Sample cache
# ==========================================================================

class DrumCache:
    """
    Pre-renders a few variants of each drum hit.

    Rotating between 3-4 slightly different renders of the same drum is how
    hardware samplers with round-robin layers avoid the "machine gun" effect,
    where an identical waveform repeated on every 16th reads as obviously
    synthetic. The variation is tiny -- a few Hz of tuning, a different noise
    seed -- but it is what stops a hi-hat line from sounding stapled on.
    """

    def __init__(self, sr=SR):
        self.sr = sr
        self.kick = [I.kick(sr, seed=11 + i, f_start=168 + i * 3,
                            decay=0.33 + i * 0.006, drive=2.1 + i * 0.05)
                     for i in range(3)]
        self.clap = [I.clap(sr, seed=23 + i * 7) for i in range(3)]
        self.snare = [I.snare(sr, seed=31 + i * 5, tune=185 + i * 4)
                      for i in range(3)]
        self.hat = [I.hihat(sr, 0.052 + i * 0.004, tone=1.0 + i * 0.03,
                            seed=41 + i * 9) for i in range(4)]
        self.ohat = [I.hihat(sr, 0.30 + i * 0.02, tone=0.97 + i * 0.03,
                             seed=45 + i * 9, open_hat=True) for i in range(3)]
        self.shaker = [I.shaker(sr, 0.085 + i * 0.006, seed=57 + i * 11)
                       for i in range(4)]
        self.crash = [I.crash(sr, 2.7, seed=71 + i * 8) for i in range(2)]

    def pick(self, bank, i):
        return bank[i % len(bank)]


# ==========================================================================
# Automation
# ==========================================================================

def build_filter_curve(clock, n, sr=SR):
    """
    One master cutoff curve for the whole track, assembled from each section's
    `filter_sweep`.

    Sweeps are exponential in frequency, because pitch perception is
    logarithmic: a linear ramp from 600 Hz to 9 kHz spends most of its time in
    the top octave and sounds like it lurches at the start then stalls.
    """
    curve = np.full(n, 20000.0)
    for sec in C.SECTIONS:
        a = max(0, clock.at(sec.start))
        b = min(clock.at(sec.end), n)
        if b <= a:
            continue
        sweep = sec.parts.get("filter_sweep")
        if sweep is None:
            curve[a:b] = 20000.0
        else:
            f0, f1 = sweep
            full_a, full_b = clock.at(sec.start), clock.at(sec.end)
            t = np.linspace(0.0, 1.0, full_b - full_a)[a - full_a:b - full_a]
            curve[a:b] = f0 * (f1 / f0) ** (t ** 0.85)
    # A short smoothing pass stops the joins between sections from clicking.
    w = int(0.05 * sr)
    win = np.hanning(w) / np.hanning(w).sum()
    pad = np.concatenate([np.full(w, curve[0]), curve, np.full(w, curve[-1])])
    return np.convolve(pad, win, mode="same")[w:w + n]


# How loud each section sits relative to the drops, before mastering.
# Without this the limiter flattens everything to the same level and the
# track has no arc -- measured loudness range collapses to under 3 LU, which
# is what "loud but boring" sounds like on a meter.
SECTION_GAIN = {
    "intro": 0.80, "main_a": 0.95, "break": 0.55, "build": 0.86,
    "drop": 1.00, "drop_b": 1.00, "outro": 0.82,
}


def build_section_gain(clock, n, sr=SR):
    """Per-section level envelope, cross-faded so the joins are inaudible."""
    g = np.ones(n)
    for sec in C.SECTIONS:
        a = max(0, clock.at(sec.start))
        b = min(clock.at(sec.end), n)
        if b > a:
            g[a:b] = SECTION_GAIN.get(sec.name, 1.0)
    w = int(0.35 * sr)                      # ~1/5 bar cross-fade
    win = np.hanning(w) / np.hanning(w).sum()
    pad = np.concatenate([np.full(w, g[0]), g, np.full(w, g[-1])])
    return np.convolve(pad, win, mode="same")[w:w + n]


def part_state(sec, key, local_bar):
    """
    Whether `key` plays in this bar, and with what value.

    Supports `key=value`, plus `key_from=bar` and `key_until=bar` so a part can
    enter or leave partway through a section (the breakdown's kick returning at
    bar 8, the outro's bass dropping out at bar 8).
    Returns None when the part is silent.
    """
    has = sec.parts.get(key)
    frm = sec.parts.get(f"{key}_from")
    until = sec.parts.get(f"{key}_until")
    if has is None and frm is None and until is None:
        return None
    if frm is not None and local_bar < frm:
        return None
    if until is not None and local_bar >= until:
        return None
    return has if has is not None else True


def pattern_hits(name, swing_ok=True):
    """Yield (step, velocity) for each hit in a named 16-step pattern."""
    pat = C.P[name]
    for step, ch in enumerate(pat):
        v = C.VELOCITY.get(ch, 0.0)
        if v > 0:
            yield step, v


# ==========================================================================
# Sequencer
# ==========================================================================

def sequence(mx, clock, cache, sr=SR, verbose=True, bars=None):
    """
    Walk the arrangement and schedule every note into the mixer.

    The loop body is the whole record: the same eight bars of kick, clap,
    hats, bass, stab and pad, decided once and then *phrased* -- turned on
    and off on 8-bar boundaries and marked with fills -- rather than varied.
    """
    rng = np.random.default_rng(2024)
    kick_triggers = []

    def humanize(pos, amount_ms=1.6):
        """Tiny timing jitter. Perfectly quantised percussion is rhythmically
        correct and emotionally dead; a millisecond or two of scatter is what
        the ear reads as a person playing."""
        return pos + int(rng.normal(0.0, amount_ms) * sr / 1000.0)

    def place(part, bar, step, swung=True, jitter=1.4):
        """
        Sample position for a hit, with swing, micro-timing and jitter.

        Each instrument sits slightly ahead of or behind the grid (see
        groove.MICRO_TIMING). The kick's offset is zero and stays zero: it is
        the reference the rest is heard against.
        """
        pos = clock.at(bar, step, swung=swung) + G.micro_offset(part, sr)
        return pos if jitter <= 0 else humanize(pos, jitter)

    counters = {k: 0 for k in
                ("kick", "clap", "hat", "ohat", "shaker", "snare", "crash")}

    # Note pools: a few round-robin variants per (pitch, tone) instead of a
    # fresh synthesis for every hit. Same trick as the drum cache.
    pools = {}

    def pooled(kind, key, make, variants=3):
        slot = pools.setdefault((kind, key), [])
        if len(slot) < variants:
            slot.append(make(len(slot)))
            return slot[-1]
        counters[kind] = counters.get(kind, 0) + 1
        return slot[counters[kind] % variants]

    def nxt(k):
        counters[k] += 1
        return counters[k]

    for sec in C.SECTIONS:
        if verbose:
            print(f"  {sec.name:8s} bars {sec.start:3d}-{sec.end - 1:3d}")

        for lb in range(sec.length):
            bar = sec.start + lb
            if bars is not None and not (bars[0] - 2 <= bar < bars[1]):
                continue
            chord = C.chord_at(bar)
            e = sec.energy
            fill = C.fill_kind(sec, lb)
            last_bar = lb == sec.length - 1
            thin = lb in sec.parts.get("thin_bars", ())

            silence = sec.parts.get("silence_from")
            sil_step = silence[1] if (silence and silence[0] == lb) else 16

            def muted(step):
                """True where the arrangement deliberately stops."""
                return step >= sil_step

            # ---------------- kick ----------------------------------------
            if part_state(sec, "kick", lb) is not None:
                k8 = sec.parts.get("kick_8ths_from")
                if k8 is not None and lb >= k8:
                    pat = "kick_8ths"
                elif fill == "fill_16":
                    pat = "kick_drop4"
                else:
                    pat = "kick"
                for step, vel in pattern_hits(pat):
                    if muted(step):
                        continue
                    pos = clock.at(bar, step)          # never micro-shifted
                    kick_triggers.append(pos)
                    mx.channels["kick"].add(
                        cache.pick(cache.kick, nxt("kick")) * vel, pos)

            # ---------------- clap + fills --------------------------------
            if part_state(sec, "clap", lb) and not thin:
                for step, vel in pattern_hits("clap"):
                    if muted(step):
                        continue
                    mx.channels["clap"].add(
                        cache.pick(cache.clap, nxt("clap")) * vel *
                        rng.uniform(0.93, 1.0), place("clap", bar, step, jitter=1.1))

            # The fill is a clap roll into the next phrase. Its velocity
            # ramps so the last hit lands hardest, right before the downbeat.
            if fill and not thin:
                for step, vel in pattern_hits(fill):
                    if muted(step):
                        continue
                    mx.channels["clap"].add(
                        cache.pick(cache.clap, nxt("clap")) * vel * 0.75,
                        place("clap", bar, step, jitter=0.8))

            # The last bar of a section swells a reversed cymbal into the
            # next section's downbeat, where a crash answers it.
            if last_bar and sec.name in C.GROOVE_SECTIONS and sec.name != "outro":
                rc = I.reverse_crash(1.5, sr)
                mx.channels["fx"].add(rc * 0.42, clock.at(bar + 1) - len(rc))

            # ---------------- hats ----------------------------------------
            hp = part_state(sec, "hat", lb)
            if hp and not thin:
                name = hp if isinstance(hp, str) else "hat"
                h16 = sec.parts.get("hat16_from")
                if h16 is not None and lb >= h16:
                    name = "hat"
                for step, vel in pattern_hits(name):
                    if muted(step):
                        continue
                    # brightness tracks section energy: quieter sections get
                    # darker hats, which reads as "further away"
                    g = vel * rng.uniform(0.9, 1.05) * (0.74 + 0.32 * e)
                    mx.channels["hat"].add(
                        cache.pick(cache.hat, nxt("hat")) * g,
                        place("hat", bar, step, jitter=1.2))

            if part_state(sec, "ohat", lb) and not thin:
                for step, vel in pattern_hits("ohat"):
                    if muted(step):
                        continue
                    mx.channels["ohat"].add(
                        cache.pick(cache.ohat, nxt("ohat")) * vel *
                        rng.uniform(0.92, 1.04), place("ohat", bar, step, jitter=1.2))

            if part_state(sec, "shaker", lb) and not thin:
                for step, vel in pattern_hits("shaker"):
                    if muted(step):
                        continue
                    mx.channels["shaker"].add(
                        cache.pick(cache.shaker, nxt("shaker")) * vel *
                        rng.uniform(0.85, 1.08),
                        place("shaker", bar, step, jitter=1.8))

            # ---------------- bass ----------------------------------------
            bp = part_state(sec, "bass", lb)
            if bp:
                name = bp if isinstance(bp, str) else "bass"
                if sec.parts.get("bounce_second_half") and (lb % 8) >= 4:
                    name = "bass_bounce2"
                hits = list(pattern_hits(name))
                last_of_chord = (bar % C.CHORD_BARS) == C.CHORD_BARS - 1
                for k, (step, vel) in enumerate(hits):
                    if muted(step):
                        continue
                    # Legato: every note lasts until the next one, wrapping
                    # into the next bar, so it holds through the kick. That
                    # overlap is what the sidechain pumps -- a bass that only
                    # plays between kicks has nothing for the duck to shape.
                    nxt_step = hits[k + 1][0] if k + 1 < len(hits) else 16 + hits[0][0]
                    # the octave hits: the 16th after beat 3 in the bounce,
                    # the 16th before beats 2 and 4 in the double bounce
                    octave = ((name == "bass_bounce" and step == 8) or
                              (name == "bass_bounce2" and step in (5, 13)))
                    note = chord.bass + (12 if octave else 0)
                    if step == 14 and last_of_chord and not octave:
                        note = C.approach_note(bar)
                    if octave:
                        dur = clock.dur(1) / sr + 0.04
                    else:
                        dur = clock.dur(nxt_step - step) / sr + 0.04
                    # Deep: the saws sit low (cutoff 260-480 Hz) with the
                    # filter envelope giving each note a rounded "boing" of
                    # an attack, over a full-level sine. The octave hits
                    # open the filter further so they pop out of the line.
                    b = I.bass(midi_to_hz(note), dur, sr,
                               cutoff=(260.0 + 220.0 * e) * (1.6 if octave else 1.0),
                               res=1.9, env_amount=2.6, decay=0.10,
                               sub=1.0 if not octave else 0.6,
                               saw_level=0.42 + 0.18 * e,
                               drive=1.6 + 0.3 * e,
                               seed=83 + bar * 3 + step)
                    pos = place("bass", bar, step, swung=False, jitter=0)
                    mx.channels["bass"].add(b * vel, pos)
                    # sine sub under the roots (never the octave): the
                    # chest-level weight a filtered saw cannot give
                    if sec.parts.get("sub_layer") and not octave:
                        mx.channels["sub"].add(
                            I.sub_note(midi_to_hz(note), dur, sr) * vel * 0.8, pos)

            # ---------------- chord stab riff -----------------------------
            if part_state(sec, "stab", lb) and not thin:
                name = "stab_a" if bar % 2 == 0 else "stab_b"
                freqs = [midi_to_hz(m) for m in chord.voicing]
                cut = round((1600.0 + 2800.0 * e) / 50.0) * 50.0
                for step, vel in pattern_hits(name):
                    if muted(step):
                        continue
                    pos = place("stab", bar, step, jitter=0)
                    org = pooled("organ", (tuple(chord.voicing),),
                                 lambda v: I.organ(freqs, clock.dur(3) / sr + 0.12,
                                                   sr, decay=0.26, seed=211 + v * 7))
                    mx.channels["stab"].add(org * vel * 0.85, pos)
                    # a quiet detuned-saw layer on top for width and air
                    saw_layer = pooled("stab", (tuple(chord.voicing), cut),
                                       lambda v: I.stab(freqs, clock.dur(2) / sr + 0.12,
                                                        sr, cutoff=cut, decay=0.14,
                                                        detune=8.0, seed=97 + v * 13))
                    mx.channels["stab"].add(saw_layer * vel * 0.32, pos)

            # ---------------- pad (one long note per chord) ----------------
            if part_state(sec, "pad", lb) and bar % C.CHORD_BARS == 0:
                dur = C.CHORD_BARS * clock.bar + 0.9
                cut = round(900.0 + 2000.0 * e)
                p = pooled("pad", (tuple(chord.voicing), cut),
                           lambda v: I.pad([midi_to_hz(m) for m in chord.voicing],
                                           dur, sr, cutoff=cut, attack=0.5,
                                           release=1.1, seed=101 + v * 11),
                           variants=2)
                mx.channels["pad"].add(p, clock.at(bar) - int(0.03 * sr))
                # Air: the top voice doubled an octave up, quiet. The close
                # voicings sit inside one octave around middle C; one high
                # voice is what makes the bed sound rich rather than thick.
                tcut = round(2000.0 + 2400.0 * e)
                top = pooled("padair", (chord.voicing[-1], tcut),
                             lambda v: I.pad([midi_to_hz(chord.voicing[-1] + 12)],
                                             dur, sr, cutoff=tcut, voices=5,
                                             detune=12.0, attack=0.9,
                                             release=1.3, seed=103 + v * 11),
                             variants=2)
                mx.channels["pad"].add(top * (0.28 + 0.18 * e),
                                       clock.at(bar) - int(0.03 * sr))

            # ---------------- breakdown electric piano --------------------
            if part_state(sec, "keys_chords", lb):
                freqs = [midi_to_hz(m) for m in chord.voicing]
                if bar % C.CHORD_BARS == 0:
                    k = I.keys(freqs, clock.dur(24) / sr, sr, decay=1.5,
                               seed=113 + bar)
                    mx.channels["keys"].add(k, place("keys", bar, 0, jitter=0))
                else:
                    k = I.keys(freqs, clock.dur(8) / sr, sr, decay=0.7,
                               seed=113 + bar)
                    mx.channels["keys"].add(k * 0.6, place("keys", bar, 6, jitter=0))

            # ---------------- drop electric-piano riff --------------------
            if part_state(sec, "keys_riff", lb) and not thin:
                freqs = [midi_to_hz(m) for m in chord.voicing]
                for j, (step, vel) in enumerate(pattern_hits("keys")):
                    if muted(step):
                        continue
                    k = pooled("keysriff", (tuple(chord.voicing), j % 2),
                               lambda v: I.keys(freqs, clock.dur(3) / sr, sr,
                                                decay=0.32, seed=115 + v * 3))
                    mx.channels["keys"].add(k * vel * [1.0, 0.8, 0.9][j],
                                            place("keys", bar, step, jitter=0.8))

            # ---------------- vocal hook ----------------------------------
            if part_state(sec, "vox", lb):
                for vb, step, vowel, ln in C.VOX_HOOK:
                    if lb % 4 != vb or muted(step):
                        continue
                    v = I.vox_chop(midi_to_hz(chord.top), clock.dur(ln) / sr, sr,
                                   vowel=vowel, seed=137 + bar + step, decay=0.35)
                    mx.channels["vox"].add(v * 0.9, place("vox", bar, step, jitter=0))

            # ---------------- snare roll ----------------------------------
            if part_state(sec, "snare_roll", lb) and lb >= sec.length - 4:
                k = lb - (sec.length - 4)
                div = [2.0, 2.0, 1.0, 0.5][k]          # 8ths -> 32nds
                step = 0.0
                while step < 16:
                    if muted(step):
                        break
                    pos = clock.at(bar, step)
                    v = 0.30 + 0.65 * ((k * 16 + step) / 64.0)
                    mx.channels["fx"].add(
                        cache.pick(cache.snare, nxt("snare")) * v * 0.6, pos)
                    step += div

            # ---------------- transition FX -------------------------------
            for cb in sec.parts.get("crash_at", []):
                if lb == cb:
                    mx.channels["fx"].add(
                        cache.pick(cache.crash, nxt("crash")) * 0.55,
                        clock.at(bar))
            for cb in sec.parts.get("reverse_crash_at", []):
                if lb == cb:
                    rc = I.reverse_crash(1.9, sr)
                    mx.channels["fx"].add(rc * 0.5, clock.at(bar + 1) - len(rc))

            # The sub drop lives on its own unducked channel: on the sub
            # channel the kick's sidechain would swallow the very impact it
            # is meant to reinforce.
            if part_state(sec, "sub_drop", lb) and lb == 0:
                mx.channels["impact"].add(I.sub_drop(1.5, sr) * 0.8, clock.at(bar))

            if part_state(sec, "riser", lb) and lb == sec.length - 8:
                dur = 8 * clock.bar
                cut = sec.parts.get("silence_from")
                if cut:
                    # end the riser exactly where everything else stops, so
                    # the gap is real silence rather than a held sweep
                    dur = ((cut[0] - lb) * 16 + cut[1]) * clock.step
                mx.channels["fx"].add(
                    I.riser(dur, sr, 240.0, 9000.0, seed=149 + bar, curve=2.2) * 0.34,
                    clock.at(bar))

            if part_state(sec, "downlifter", lb) and lb == 0:
                mx.channels["fx"].add(
                    I.downlifter(1.8, sr, seed=151 + bar) * 0.35, clock.at(bar))

            # a swoosh under the big section changes
            if lb == 0 and sec.name in ("break", "build", "drop"):
                sw = I.noise_sweep(1.6, sr, up=False, seed=167 + bar)
                mx.channels["fx"].add(sw * 0.2, clock.at(bar) - int(0.8 * sr))

    return sorted(set(kick_triggers))


# ==========================================================================
# Mixer setup
# ==========================================================================

def build_mixer(n, sr, fcurve):
    """
    Create every channel with its frequency slot, stereo position and depth.

    Thirteen channels. Low end and backbeat dead centre; the top of the kit and
    the chords fanned out. Everything tonal except the bass is sidechained
    to the kick -- the pad and bass hard, so the whole harmonic bed pumps
    in time, which is the physical sensation of house.
    """
    mx = Mixer(n, sr)

    # --- effect returns ---------------------------------------------------
    beat = 60.0 / C.BPM
    mx.bus("room", S.reverb_ir(sr, rt60=0.85, predelay=0.008, damping=0.55,
                               width=0.9, er_level=0.8, seed=7),
           gain_db=-8.0, eq=[F.highpass(400.0, 0.707, sr),
                              F.lowpass(9000.0, 0.707, sr)], width=1.1)

    mx.bus("plate", S.reverb_ir(sr, rt60=1.9, predelay=0.022, damping=0.45,
                                width=1.15, er_level=0.35, seed=17),
           gain_db=-9.5, eq=[F.highpass(320.0, 0.707, sr),
                              F.lowpass(11000.0, 0.707, sr)],
           width=1.2, duck=0.4)

    mx.bus("hall", S.reverb_ir(sr, rt60=3.6, predelay=0.045, damping=0.62,
                               width=1.3, er_level=0.25, seed=27),
           gain_db=-11.0, eq=[F.highpass(260.0, 0.707, sr),
                              F.lowpass(8000.0, 0.707, sr)],
           width=1.35, duck=0.5)

    # Dotted-eighth delay: 0.75 of a beat, lands between the 16ths.
    mx.bus("delay", S.delay_ir(sr, time=beat * 0.75, feedback=0.40,
                               repeats=12, ping_pong=True, damping=0.55),
           gain_db=-10.0, eq=[F.highpass(380.0, 0.707, sr),
                              F.lowpass(7000.0, 0.707, sr)],
           width=1.3, duck=0.5)

    # --- channels ---------------------------------------------------------
    mx.channel("kick", gain_db=-7.5, pan=0.0,
               comp=dict(threshold=-12.0, ratio=2.2, attack=0.012,
                         release=0.120, knee=4.0, makeup=1.0),
               sends={"room": 0.05})

    # The sub is sidechained too: it holds through the kick like the bass,
    # and two things at 45-90 Hz at once is mud, not weight.
    mx.channel("sub", gain_db=-14.5, pan=0.0, mono_below=200.0,
               hp=26.0, lp=140.0, duck=0.92)

    mx.channel("impact", gain_db=-13.0, pan=0.0, mono_below=200.0,
               hp=26.0, lp=160.0)

    mx.channel("bass", gain_db=-9.5, pan=0.0, duck=0.92, mono_below=140.0,
               hp=28.0,
               eq=[F.peaking(95.0, 0.8, 1.0, sr),
                   F.peaking(280.0, -2.0, 1.0, sr),
                   F.peaking(1100.0, 1.0, 0.9, sr)],
               excite=dict(band=(700.0, 2600.0), keep_above=2200.0,
                           drive=3.0, mix=0.35, mode="tube"),
               comp=dict(threshold=-20.0, ratio=3.5, attack=0.006,
                         release=0.085, knee=5.0, makeup=3.0),
               filter_curve=fcurve)

    mx.channel("clap", gain_db=-9.0, pan=0.0, width=1.25, hp=220.0,
               comp=dict(threshold=-20.0, ratio=2.5, attack=0.003,
                         release=0.100, makeup=2.0),
               sends={"room": 0.40, "plate": 0.16})

    mx.channel("hat", gain_db=-11.5, pan=0.13, width=1.15, hp=420.0,
               eq=[F.peaking(7000.0, 2.0, 0.9, sr),
                   F.highshelf(10000.0, 2.0, 0.7, sr),
                   F.highshelf(13500.0, 1.8, 0.6, sr)],
               sends={"room": 0.12}, filter_curve=fcurve)

    mx.channel("ohat", gain_db=-12.0, pan=-0.20, width=1.22, hp=420.0,
               eq=[F.highshelf(12000.0, 1.6, 0.6, sr)],
               duck=0.30, sends={"room": 0.16}, filter_curve=fcurve)

    mx.channel("shaker", gain_db=-19.0, pan=0.40, width=1.1, hp=2500.0,
               sends={"room": 0.10}, filter_curve=fcurve)

    mx.channel("stab", gain_db=-10.5, width=1.30, duck=0.75, hp=170.0,
               eq=[F.peaking(430.0, -1.5, 1.0, sr),
                   F.peaking(2400.0, 1.5, 0.9, sr)],
               excite=dict(band=(800.0, 3000.0), keep_above=2600.0,
                           drive=4.0, mix=1.0, mode="tube"),
               comp=dict(threshold=-22.0, ratio=2.5, attack=0.008,
                         release=0.130, makeup=2.5),
               sends={"plate": 0.28, "delay": 0.16, "room": 0.08},
               filter_curve=fcurve)

    # The pad is the thing the sidechain is heard on: a sustained chord
    # dipping 95 % on every kick is the pump.
    mx.channel("pad", gain_db=-15.0, width=1.50, duck=0.95, hp=150.0,
               eq=[F.peaking(330.0, -2.2, 0.9, sr),
                   F.highshelf(9000.0, 1.5, 0.7, sr)],
               excite=dict(band=(700.0, 2400.0), keep_above=2500.0,
                           drive=4.4, mix=1.20, mode="tube"),
               sends={"hall": 0.55, "plate": 0.15},
               filter_curve=fcurve)

    mx.channel("keys", gain_db=-14.0, pan=-0.12, width=1.20, duck=0.6, hp=200.0,
               eq=[F.peaking(400.0, -1.0, 1.0, sr)],
               excite=dict(band=(800.0, 3000.0), keep_above=2600.0,
                           drive=4.4, mix=1.20, mode="tube"),
               sends={"plate": 0.34, "delay": 0.18, "hall": 0.12},
               filter_curve=fcurve)

    mx.channel("vox", gain_db=-16.0, width=1.18, duck=0.55, hp=220.0,
               eq=[F.peaking(1800.0, 2.2, 1.0, sr)],
               excite=dict(band=(900.0, 3200.0), keep_above=2700.0,
                           drive=4.4, mix=1.20, mode="tube"),
               sends={"hall": 0.40, "delay": 0.26, "plate": 0.18},
               filter_curve=fcurve)

    mx.channel("fx", gain_db=-14.5, width=1.40, hp=180.0,
               sends={"hall": 0.30, "plate": 0.18})

    return mx


# ==========================================================================
# File output
# ==========================================================================

def _quantize(x, bits, seed=999):
    """
    Convert float to fixed point with TPDF dither.

    Quantisation without dither correlates the rounding error with the signal,
    which turns it into harmonic distortion audible on quiet fades. Adding a
    triangular random signal of about one LSB decorrelates it -- the noise
    floor rises marginally and the distortion disappears entirely.
    """
    rng = np.random.default_rng(seed)
    full = 2 ** (bits - 1) - 1
    lsb = 1.0 / full
    tpdf = (rng.random(x.shape) - rng.random(x.shape)) * lsb
    y = np.clip(x + tpdf, -1.0, 1.0)
    return np.round(y * full).astype(np.int32)


def write_wav(path, x, sr, bits=24):
    data = _quantize(x, bits)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(bits // 8)
        w.setframerate(sr)
        if bits == 24:
            b = data.astype("<i4").tobytes()
            raw = np.frombuffer(b, dtype=np.uint8).reshape(-1, 4)[:, :3].tobytes()
        else:
            raw = data.astype("<i2").tobytes()
        w.writeframes(raw)
    return path


# Tags a DJ's software reads. Rekordbox, Serato and Traktor all key their
# libraries on BPM and initial key; without them the track has to be
# re-analysed on import, and the analysers are not always right about the key.
MP3_TAGS = {
    "title": "Midnight Transit",
    "genre": "House",
    "TBPM": str(int(C.BPM)) if float(C.BPM).is_integer() else str(C.BPM),
    "TKEY": "Fm",                       # F minor -- Camelot 4A
    "date": "2026",
    "comment": "124 BPM, F minor (Camelot 4A). 32-bar beat intro and outro "
               "for mixing. Synthesised entirely in code.",
}


def encode_mp3(wav_path, mp3_path, bitrate="320k", artist=None):
    """
    Encode a 320 kbps MP3 via ffmpeg, if it is available.

    The master is limited to -0.9 dBFS rather than 0.0 precisely so this step
    is safe: MP3 decoding reconstructs inter-sample peaks that can sit above
    the highest sample in the source, and a master pushed to 0.0 will clip on
    playback even though the WAV measures clean.
    """
    import shutil
    import subprocess
    if not shutil.which("ffmpeg"):
        print("  (ffmpeg not found -- skipping MP3)")
        return None
    tags = dict(MP3_TAGS)
    if artist:
        tags["artist"] = artist
    meta = [a for k, v in tags.items() for a in ("-metadata", f"{k}={v}")]
    r = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", wav_path,
         "-codec:a", "libmp3lame", "-b:a", bitrate, "-id3v2_version", "3",
         *meta, mp3_path],
        capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  (MP3 encode failed: {r.stderr.strip()[:200]})")
        return None
    return mp3_path


# ==========================================================================
# Main
# ==========================================================================

def build_track(sr=SR, verbose=True, keep_stems=False, bars=None,
                parallel=True):
    """
    Render the track, or with `bars=(a, b)` just bars a..b-1.

    A partial render is for iterating on a mix: a 24-bar drop takes a
    fraction of the time of the full 192. It goes through exactly the same
    sequencer, mixer and master chain, so what you hear in the window is
    what that window will sound like in the full render -- with two honest
    caveats. Loudness targeting sees only the window, so the limiter drive
    can differ by a fraction of a dB; and anything sustaining in from more
    than two bars before the window is not there.
    """
    t0 = time.time()
    origin = 0 if bars is None else bars[0]
    clock = C.Clock(C.BPM, sr, C.SWING, origin_bar=origin)
    last = C.TOTAL_BARS if bars is None else bars[1]
    n = clock.bars_to_samples(last - origin) + int(5.0 * sr)   # room for tails

    if verbose:
        print(C.describe())
        print(f"\nrendering {n/sr:.1f}s at {sr} Hz\n")
        print("[1/5] pre-rendering drum variants")
    cache = DrumCache(sr)

    if verbose:
        print("[2/5] building automation + mixer")
    fcurve = build_filter_curve(clock, n, sr)
    mx = build_mixer(n, sr, fcurve)

    if verbose:
        print("[3/5] sequencing")
    kicks = sequence(mx, clock, cache, sr, verbose, bars=bars)
    kicks = [k for k in kicks if k >= 0]

    # The pump follows the actual kick events, so it stops automatically
    # wherever the kick drops out.
    # Depth 0.9, 30 ms hold, 180 ms release: fully recovered 210 ms after
    # the kick, which at 124 BPM is 30 ms before the off-beat where the
    # bass and the open hat land. The pump is deep and it never eats the
    # off-beat -- that timing is the whole trick.
    duck = D.duck_envelope(n, kicks, sr, depth=0.90, hold=0.030,
                           release=0.180, curve=2.4)

    if verbose:
        print(f"[4/5] mixing ({len(mx.channels)} channels, "
              f"{len(kicks)} kick triggers)")
    mix = mx.render(duck, verbose, keep_stems=keep_stems, parallel=parallel)

    mix *= build_section_gain(clock, n, sr)[:, None]

    if verbose:
        print("[5/5] mastering")
    # -10.5 rather than -9.3: at -9.3 the limiter was loudness-saturated,
    # flattening the kick's transient to buy level the waveform could not
    # give. DJs gain-match in the booth; punch is what they cannot add back.
    master = master_chain(mix, sr, target_lufs=-10.5, ceiling_db=-1.0,
                          verbose=verbose)

    if verbose:
        print(f"\nrendered in {time.time() - t0:.1f}s")
    return master, mix, clock, mx


def main():
    ap = argparse.ArgumentParser(description="Render the house track.")
    ap.add_argument("--out", default="output", help="output directory")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--bars", default=None, metavar="A-B",
                    help="render only bars A..B-1, e.g. 64-88 for the main drop")
    ap.add_argument("--artist", default=None,
                    help="artist name written into the MP3 tags")
    ap.add_argument("--serial", action="store_true",
                    help="single-process mixing (for timing comparisons)")
    args = ap.parse_args()

    verbose = not args.quiet
    bars = None
    if args.bars:
        a, b = (int(v) for v in args.bars.split("-"))
        bars = (a, b)
    master, mix, clock, _mx = build_track(SR, verbose, bars=bars,
                                          parallel=not args.serial)

    os.makedirs(args.out, exist_ok=True)
    stem = os.path.join(args.out, "midnight_transit")
    if bars:
        stem += f"_bars{bars[0]}-{bars[1]}"      # never overwrite the master

    outputs = []
    outputs.append(write_wav(f"{stem}_master_24bit_48k.wav", master, SR, 24))

    # 16-bit / 44.1 kHz for CD-rate delivery. resample_poly's 147/160 ratio is
    # exact (44100/48000), so this is a clean rational resample rather than an
    # interpolation with drift.
    from scipy.signal import resample_poly
    cd = np.stack([resample_poly(master[:, c], 147, 160) for c in range(2)],
                  axis=-1)
    # Resampling can nudge peaks slightly above the source ceiling.
    cd = np.clip(cd, -db(-1.0), db(-1.0))
    outputs.append(write_wav(f"{stem}_master_16bit_44k.wav", cd, 44100, 16))

    mp3 = encode_mp3(outputs[0], f"{stem}.mp3", artist=args.artist)
    if mp3:
        outputs.append(mp3)

    print()
    print(A.report(master, SR, "MASTER"))
    print("\nfiles written:")
    for f in outputs:
        print(f"  {f}  ({os.path.getsize(f)/1e6:.1f} MB)")

    return stem


if __name__ == "__main__":
    main()
