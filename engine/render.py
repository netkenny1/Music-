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
        self.rim = [I.rim(sr, seed=61 + i * 6, tune=420 + i * 12)
                    for i in range(3)]
        self.crash = [I.crash(sr, 2.7, seed=71 + i * 8) for i in range(2)]
        self.tom = [I.tom(sr, seed=67 + i * 4, f_start=210 - i * 25,
                          f_end=95 - i * 10) for i in range(3)]

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
    "intro": 0.68, "build1": 0.84, "drop1": 1.00, "break": 0.61,
    "build2": 0.88, "drop2": 1.00, "outro": 0.78,
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
    """Walk the arrangement and schedule every note into the mixer."""
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

        Micro-timing is the deliberate part: each instrument sits slightly
        ahead of or behind the grid (see groove.MICRO_TIMING), which is what
        separates a groove from a quantised loop. The kick's offset is zero
        and stays zero -- it is the reference the rest is heard against.
        """
        pos = clock.at(bar, step, swung=swung) + G.micro_offset(part, sr)
        return pos if jitter <= 0 else humanize(pos, jitter)

    counters = {k: 0 for k in
                ("kick", "clap", "hat", "ohat", "shaker", "rim", "snare",
                 "crash", "tom")}

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
            fills = sec.parts.get("fill_bars", [])
            is_fill = lb in fills

            # --- expectation-violation events for this bar ----------------
            is_hole = sec.parts.get("hole_bar") == lb
            silence = sec.parts.get("silence_from")
            sil_step = silence[1] if (silence and silence[0] == lb) else 16

            def muted(step):
                """True where the arrangement deliberately stops."""
                return step >= sil_step

            kick_pat = sec.parts.get("kick_pattern_bars", {}).get(lb)
            push_clap = lb in sec.parts.get("clap_push_bars", [])
            ante_bass = lb in sec.parts.get("bass_ante_bars", [])
            late_stab = lb in sec.parts.get("stab_late_bars", [])

            # ---------------- the delayed drop ----------------------------
            # The single strongest violation in the track. Everything the
            # build promised for this downbeat simply does not arrive: no
            # kick, no groove, just a sub and the tail of the riser hanging
            # in the air. Then the kick enters EARLY, on the last 8th, so the
            # beat first fails to appear where predicted and then pre-empts
            # the next downbeat. Two violations, opposite directions, one bar.
            if is_hole:
                mx.channels["sub"].add(I.sub_drop(2.2, sr, f0=120.0) * 0.85,
                                       clock.at(bar))
                rc = I.reverse_crash(1.9, sr)
                mx.channels["fx"].add(rc * 0.55, clock.at(bar + 1) - len(rc))
                anticip = clock.at(bar, 14)
                kick_triggers.append(anticip)
                mx.channels["kick"].add(
                    cache.pick(cache.kick, nxt("kick")) * 0.9, anticip)
                mx.channels["ohat"].add(
                    cache.pick(cache.ohat, nxt("ohat")) * 0.7,
                    place("ohat", bar, 14))

            # ---------------- kick ----------------------------------------
            if not is_hole and part_state(sec, "kick", lb) is not None:
                pat = kick_pat or ("kick_fill" if is_fill else "kick")
                for step, vel in pattern_hits(pat):
                    if muted(step):
                        continue
                    pos = clock.at(bar, step)          # never micro-shifted
                    kick_triggers.append(pos)
                    mx.channels["kick"].add(
                        cache.pick(cache.kick, nxt("kick")) * vel, pos)
                # A missing kick needs something to mark the moment, or it
                # reads as a mistake rather than a decision.
                if kick_pat == "kick_hole1":
                    mx.channels["fx"].add(
                        cache.pick(cache.crash, nxt("crash")) * 0.42,
                        clock.at(bar))

            # ---------------- clap ----------------------------------------
            cp = part_state(sec, "clap", lb)
            if cp and not is_hole:
                # A clap a 16th early arrives before the ear has finished
                # predicting beat 2 -- the backbeat is still there, just not
                # where it was promised.
                name = "clap_push" if push_clap else (cp if isinstance(cp, str) else "clap")
                for step, vel in pattern_hits(name):
                    if muted(step):
                        continue
                    mx.channels["clap"].add(
                        cache.pick(cache.clap, nxt("clap")) * vel *
                        rng.uniform(0.93, 1.0), place("clap", bar, step, jitter=1.1))

            # ---------------- hats ----------------------------------------
            hp = part_state(sec, "hat", lb)
            if hp:
                name = hp if isinstance(hp, str) else "hat"
                for step, vel in pattern_hits(name):
                    if muted(step):
                        continue
                    # brightness tracks section energy: quieter sections get
                    # darker hats, which reads as "further away"
                    g = vel * rng.uniform(0.88, 1.06) * (0.72 + 0.35 * sec.energy)
                    mx.channels["hat"].add(
                        cache.pick(cache.hat, nxt("hat")) * g,
                        place("hat", bar, step, jitter=1.5))

            if part_state(sec, "ohat", lb) and not is_hole:
                for step, vel in pattern_hits("ohat"):
                    if muted(step):
                        continue
                    mx.channels["ohat"].add(
                        cache.pick(cache.ohat, nxt("ohat")) * vel *
                        rng.uniform(0.9, 1.05), place("ohat", bar, step, jitter=1.4))

            if part_state(sec, "shaker", lb) and not is_hole:
                for step, vel in pattern_hits("shaker"):
                    if muted(step):
                        continue
                    mx.channels["shaker"].add(
                        cache.pick(cache.shaker, nxt("shaker")) * vel *
                        rng.uniform(0.82, 1.08),
                        place("shaker", bar, step, jitter=2.0))

            if part_state(sec, "rim", lb) and lb % 2 == 1 and not is_hole:
                for step, vel in pattern_hits("rim"):
                    if muted(step):
                        continue
                    mx.channels["rim"].add(
                        cache.pick(cache.rim, nxt("rim")) * vel,
                        place("rim", bar, step, jitter=2.2))

            # --- 3-against-4 polyrhythm ------------------------------------
            # A hit every 3 sixteenths against a 16-step bar will not realign
            # with the downbeat for three bars (lcm(3,16) = 48). The layer
            # rotates against the pulse and locks back in every third bar --
            # a slow tension-and-release the listener feels without naming.
            poly_from = sec.parts.get("poly_from")
            if poly_from is not None and lb >= poly_from and not is_hole:
                for pb, pstep in G.polyrhythm_steps(3, 1, offset=(lb * 16) % 3):
                    if muted(pstep):
                        continue
                    mx.channels["rim"].add(
                        cache.pick(cache.rim, nxt("rim")) * 0.42,
                        place("rim", bar, pstep, swung=False, jitter=1.0),
                        pan=0.55)

            # ---------------- bass ----------------------------------------
            bp = part_state(sec, "bass", lb)
            if bp and not is_hole:
                # The anticipated bass lands a 16th BEFORE the beat and holds
                # through it, so the downbeat is felt but never struck.
                name = "bass_ante" if ante_bass else (bp if isinstance(bp, str) else "bass")
                hits = list(pattern_hits(name))
                for k, (step, vel) in enumerate(hits):
                    # a note lasts until the next one, so the line is legato
                    nxt_step = hits[k + 1][0] if k + 1 < len(hits) else 16
                    steps = min(nxt_step - step, 4)
                    # octave lift on the pickup into the next bar
                    note = chord.bass + (12 if step >= 15 else 0)
                    dur = clock.dur(steps) / sr + 0.06
                    if muted(step):
                        continue
                    b = I.bass(midi_to_hz(note), dur, sr,
                               cutoff=340.0 + 260.0 * sec.energy,
                               res=1.7, saw_level=0.42 + 0.22 * sec.energy,
                               drive=1.5 + 0.3 * sec.energy,
                               seed=83 + bar * 3 + step)
                    mx.channels["bass"].add(
                        b * vel, place("bass", bar, step, swung=False, jitter=0))

            # ---------------- chord stabs ---------------------------------
            sp = part_state(sec, "stab", lb)
            if sp and not is_hole:
                name = "stab_late" if late_stab else (sp if isinstance(sp, str) else "stab")
                for step, vel in pattern_hits(name):
                    if muted(step):
                        continue
                    pos = place("stab", bar, step, jitter=0)
                    st = I.stab([midi_to_hz(m) for m in chord.voicing],
                                clock.dur(3) / sr + 0.18, sr,
                                cutoff=2900.0 + 3100.0 * sec.energy,
                                decay=0.20, detune=11.0,
                                seed=97 + bar * 5 + step)
                    mx.channels["stab"].add(st * vel * 0.9, pos)

            # ---------------- pad (one long note per chord) ----------------
            if part_state(sec, "pad", lb) and bar % C.CHORD_BARS == 0:
                dur = C.CHORD_BARS * clock.bar + 1.1
                p = I.pad([midi_to_hz(m) for m in chord.voicing], dur, sr,
                          cutoff=1250.0 + 2050.0 * sec.energy,
                          attack=0.8, release=1.3, seed=101 + bar)
                mx.channels["pad"].add(p, clock.at(bar) - int(0.05 * sr))
                # Air voice: the top note doubled an octave up, quiet. The
                # close voicings all sit inside one octave around middle C,
                # which is warm but leaves 1-3 kHz with no *musical* content.
                # One high voice fills that band with something harmonic
                # instead of leaving it to the hats.
                top = I.pad([midi_to_hz(chord.voicing[-1] + 12)], dur, sr,
                            cutoff=2200.0 + 2600.0 * sec.energy, voices=5,
                            detune=12.0, attack=1.1, release=1.5,
                            seed=103 + bar)
                mx.channels["pad"].add(top * (0.26 + 0.16 * sec.energy),
                                       clock.at(bar) - int(0.05 * sr))

            # ---------------- breakdown keys ------------------------------
            if part_state(sec, "keys", lb) and lb % 2 == 0:
                for step in (0, 6, 10):
                    k = I.keys([midi_to_hz(m) for m in chord.voicing],
                               clock.dur(6) / sr, sr, decay=0.85,
                               seed=113 + bar * 3 + step)
                    mx.channels["keys"].add(
                        k * (1.0 if step == 0 else 0.6),
                        place("keys", bar, step, jitter=0))

            # ---------------- arpeggio ear candy --------------------------
            if part_state(sec, "arp", lb):
                notes = chord.arp + chord.arp[-2::-1]      # up then back down
                for step, vel in pattern_hits("arp"):
                    if muted(step):
                        continue
                    note = notes[(step // 2 + lb) % len(notes)]
                    pos = place("arp", bar, step, jitter=0)
                    tone = I.bell(midi_to_hz(note), 0.34, sr,
                                  ratio=2.01, index=2.4, decay=0.30,
                                  seed=127 + bar * 7 + step)
                    # alternate sides so the delay throws it around the room
                    mx.channels["arp"].add(tone * vel * 0.8, pos,
                                           pan=0.45 if step % 4 == 0 else -0.45)

            # ---------------- vocal texture -------------------------------
            if part_state(sec, "vox", lb) and lb % 4 == 0:
                vowel = ["ah", "ooh", "eh", "ooh"][(bar // 2) % 4]
                note = chord.voicing[1] + 12
                v = I.vox_chop(midi_to_hz(note), clock.dur(10) / sr, sr,
                               vowel=vowel, seed=137 + bar, decay=0.45)
                mx.channels["vox"].add(v * 0.9, place("vox", bar, 2, jitter=0))

            # ---------------- melodic hook --------------------------------
            if part_state(sec, "melody", lb):
                cycle_start = (bar // 8) * 8
                for mstep, note, mlen in C.MELODY:
                    mbar = cycle_start + mstep // 16
                    if mbar != bar:
                        continue
                    step = mstep % 16
                    pos = place("melody", bar, step, jitter=0)
                    dur = clock.dur(mlen) / sr + 0.25
                    if sec.name == "break":
                        tone = I.keys([midi_to_hz(note)], dur, sr,
                                      decay=dur * 0.8, seed=113 + mstep)
                    else:
                        tone = I.pluck(midi_to_hz(note), dur, sr,
                                       decay=dur * 0.55, cutoff=3600.0,
                                       seed=131 + mstep)
                    mx.channels["melody"].add(tone * 0.85, pos)

            # ---------------- counter-melody ------------------------------
            if part_state(sec, "counter", lb):
                cycle_start = (bar // 8) * 8
                for mstep, note, mlen in C.COUNTER:
                    mbar = cycle_start + mstep // 16
                    if mbar != bar:
                        continue
                    step = mstep % 16
                    if muted(step):
                        continue
                    pos = place("counter", bar, step, jitter=0)
                    dur = clock.dur(mlen) / sr + 0.30
                    tone = I.pluck(midi_to_hz(note), dur, sr,
                                   decay=dur * 0.6, cutoff=5200.0,
                                   seed=139 + mstep)
                    mx.channels["counter"].add(tone * 0.80, pos)

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

            # ---------------- stutter / beat repeat ------------------------
            # An accelerating retrigger: each repeat arrives sooner than the
            # last, so prediction error piles up instead of resetting. The
            # downbeat that follows resolves the whole accumulation at once.
            for (sb, sstep, spart) in sec.parts.get("stutter_at", []):
                if lb != sb:
                    continue
                # Stop at the cut, not at the barline. A stutter that runs
                # through the intended silence fills the very gap it exists
                # to set up.
                steps_left = min(16, sil_step) - sstep
                if steps_left <= 0:
                    continue
                sched = G.stutter_schedule(steps_left)
                step_s = clock.step

                if spart == "stab":
                    src = I.stab([midi_to_hz(m) for m in chord.voicing],
                                 0.34, sr, cutoff=2800.0, decay=0.12,
                                 seed=97 + bar)
                    ch, gain = "stab", 0.75
                elif spart == "arp":
                    src = I.bell(midi_to_hz(chord.arp[1]), 0.30, sr,
                                 index=2.8, decay=0.24, seed=127 + bar)
                    ch, gain = "arp", 0.85
                else:
                    src = I.vox_chop(midi_to_hz(chord.voicing[1] + 12), 0.36,
                                     sr, vowel="ah", seed=137 + bar)
                    ch, gain = "vox", 0.9

                mx.channels[ch].add(
                    I.stutter(src, sr, step_s, sched, pitch_rise=0.45) * gain,
                    clock.at(bar, sstep))

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

            if part_state(sec, "sub_drop", lb) and lb == 0:
                mx.channels["sub"].add(I.sub_drop(1.5, sr) * 0.75, clock.at(bar))

            if part_state(sec, "riser", lb) and lb == sec.length - 4:
                dur = 4 * clock.bar
                cut = sec.parts.get("silence_from")
                if cut:
                    # end the riser exactly where everything else stops, so
                    # the gap is real silence rather than a held sweep
                    dur = ((cut[0] - lb) * 16 + cut[1]) * clock.step
                if dur > 0.2:
                    mx.channels["fx"].add(
                        I.riser(dur, sr, 240.0, 9000.0, seed=149 + bar) * 0.32,
                        clock.at(bar))

            if part_state(sec, "downlifter", lb) and lb == 0:
                mx.channels["fx"].add(
                    I.downlifter(1.8, sr, seed=151 + bar) * 0.35, clock.at(bar))

            # a swoosh under every section change smooths the transition
            if lb == 0 and sec.start > 0:
                sw = I.noise_sweep(1.6, sr, up=False, seed=167 + bar)
                mx.channels["fx"].add(sw * 0.22, clock.at(bar) - int(0.8 * sr))

            # tom fill into each drop
            if is_fill and lb == sec.length - 1:
                for j, step in enumerate([10, 12, 13, 14, 15]):
                    mx.channels["fx"].add(
                        cache.pick(cache.tom, nxt("tom")) * (0.3 + j * 0.07),
                        clock.at(bar, step, swung=True))

    return sorted(set(kick_triggers))


# ==========================================================================
# Mixer setup
# ==========================================================================

def build_mixer(n, sr, fcurve):
    """
    Create every channel with its frequency slot, stereo position and depth.

    The panning plan keeps the low end and the backbeat dead centre -- they
    carry the groove and must feel like they come from in front of you -- and
    fans the ornamental parts out to the sides where there is room.
    """
    mx = Mixer(n, sr)

    # --- effect returns ---------------------------------------------------
    # Three rooms at three distances, plus a tempo-synced delay.
    beat = 60.0 / C.BPM
    mx.bus("room", S.reverb_ir(sr, rt60=0.85, predelay=0.008, damping=0.55,
                               width=0.9, er_level=0.8, seed=7),
           gain_db=-8.0, eq=[F.highpass(400.0, 0.707, sr),
                              F.lowpass(9000.0, 0.707, sr)], width=1.1)

    mx.bus("plate", S.reverb_ir(sr, rt60=1.9, predelay=0.022, damping=0.45,
                                width=1.15, er_level=0.35, seed=17),
           gain_db=-9.5, eq=[F.highpass(320.0, 0.707, sr),
                              F.lowpass(11000.0, 0.707, sr)],
           width=1.2, duck=0.35)

    mx.bus("hall", S.reverb_ir(sr, rt60=3.6, predelay=0.045, damping=0.62,
                               width=1.3, er_level=0.25, seed=27),
           gain_db=-11.0, eq=[F.highpass(260.0, 0.707, sr),
                              F.lowpass(8000.0, 0.707, sr)],
           width=1.35, duck=0.45)

    # Dotted-eighth delay: 0.75 of a beat. It lands between the 16ths instead
    # of on top of them, so echoes add motion without thickening the groove.
    mx.bus("delay", S.delay_ir(sr, time=beat * 0.75, feedback=0.40,
                               repeats=12, ping_pong=True, damping=0.55),
           gain_db=-10.0, eq=[F.highpass(380.0, 0.707, sr),
                              F.lowpass(7000.0, 0.707, sr)],
           width=1.3, duck=0.4)

    # --- channels ---------------------------------------------------------
    mx.channel("kick", gain_db=-7.5, pan=0.0,
               comp=dict(threshold=-12.0, ratio=2.2, attack=0.012,
                         release=0.120, knee=4.0, makeup=1.0),
               sends={"room": 0.05})

    mx.channel("sub", gain_db=-15.0, pan=0.0, mono_below=200.0,
               hp=26.0, lp=140.0)

    mx.channel("bass", gain_db=-9.5, pan=0.0, duck=0.78, mono_below=140.0,
               hp=28.0,
               eq=[F.peaking(95.0, 0.8, 1.0, sr),
                   F.peaking(280.0, -2.0, 1.0, sr),
                   F.peaking(1200.0, 1.4, 0.9, sr)],
               excite=dict(band=(700.0, 2600.0), keep_above=2200.0,
                           drive=3.4, mix=0.55, mode="tube"),
               comp=dict(threshold=-20.0, ratio=3.5, attack=0.006,
                         release=0.085, knee=5.0, makeup=3.0),
               filter_curve=fcurve)

    mx.channel("clap", gain_db=-10.0, pan=0.0, width=1.25, hp=220.0,
               comp=dict(threshold=-20.0, ratio=2.5, attack=0.003,
                         release=0.100, makeup=2.0),
               sends={"room": 0.40, "plate": 0.16})

    mx.channel("hat", gain_db=-11.5, pan=0.13, width=1.15, hp=420.0,
               eq=[F.peaking(7000.0, 2.0, 0.9, sr),
                   F.highshelf(10000.0, 2.0, 0.7, sr)],
               sends={"room": 0.12}, filter_curve=fcurve)

    mx.channel("ohat", gain_db=-13.0, pan=-0.20, width=1.22, hp=420.0,
               duck=0.30, sends={"room": 0.16}, filter_curve=fcurve)

    mx.channel("shaker", gain_db=-18.0, pan=0.40, width=1.1, hp=2500.0,
               sends={"room": 0.10}, filter_curve=fcurve)

    mx.channel("rim", gain_db=-18.5, pan=-0.44, hp=430.0,
               sends={"room": 0.22, "delay": 0.12}, filter_curve=fcurve)

    mx.channel("stab", gain_db=-10.2, width=1.30, duck=0.62, hp=170.0,
               eq=[F.peaking(430.0, -1.7, 1.0, sr)],
               excite=dict(band=(800.0, 3000.0), keep_above=2600.0,
                           drive=4.8, mix=1.40, mode="tube"),
               comp=dict(threshold=-22.0, ratio=2.5, attack=0.008,
                         release=0.130, makeup=2.5),
               sends={"plate": 0.30, "delay": 0.18, "room": 0.08},
               filter_curve=fcurve)

    mx.channel("pad", gain_db=-16.4, width=1.50, duck=0.55, hp=150.0,
               eq=[F.peaking(330.0, -2.2, 0.9, sr),
                   F.highshelf(9000.0, 1.5, 0.7, sr)],
               excite=dict(band=(700.0, 2400.0), keep_above=2500.0,
                           drive=4.4, mix=1.20, mode="tube"),
               sends={"hall": 0.55, "plate": 0.15},
               filter_curve=fcurve)

    mx.channel("keys", gain_db=-14.6, width=1.20, duck=0.35, hp=200.0,
               eq=[F.peaking(400.0, -1.0, 1.0, sr)],
               excite=dict(band=(800.0, 3000.0), keep_above=2600.0,
                           drive=4.4, mix=1.20, mode="tube"),
               sends={"plate": 0.34, "delay": 0.18, "hall": 0.12},
               filter_curve=fcurve)

    mx.channel("arp", gain_db=-15.2, width=1.25, duck=0.42, hp=680.0,
               eq=[F.highshelf(8000.0, 1.5, 0.7, sr)],
               excite=dict(band=(1000.0, 3800.0), keep_above=3000.0,
                           drive=4.6, mix=1.10, mode="tanh"),
               sends={"delay": 0.48, "plate": 0.22},
               filter_curve=fcurve)

    mx.channel("vox", gain_db=-16.6, width=1.18, duck=0.45, hp=220.0,
               eq=[F.peaking(1800.0, 2.2, 1.0, sr)],
               excite=dict(band=(900.0, 3200.0), keep_above=2700.0,
                           drive=4.4, mix=1.20, mode="tube"),
               sends={"hall": 0.40, "delay": 0.22, "plate": 0.18},
               filter_curve=fcurve)

    mx.channel("melody", gain_db=-15.2, pan=-0.16, width=1.1, duck=0.35,
               hp=330.0, eq=[F.peaking(1400.0, 1.8, 0.9, sr)],
               excite=dict(band=(900.0, 3200.0), keep_above=2700.0,
                           drive=4.4, mix=1.20, mode="tube"),
               sends={"plate": 0.32, "delay": 0.30},
               filter_curve=fcurve)

    # Opposite side from the hook (-0.16), higher and drier, so the two
    # lines read as a conversation rather than a doubling.
    mx.channel("counter", gain_db=-16.8, pan=0.26, width=1.1, duck=0.35,
               hp=520.0, eq=[F.peaking(2600.0, 1.5, 0.9, sr)],
               excite=dict(band=(1200.0, 4000.0), keep_above=3200.0,
                           drive=4.4, mix=1.10, mode="tube"),
               sends={"plate": 0.26, "delay": 0.34},
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


def encode_mp3(wav_path, mp3_path, bitrate="320k"):
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
    r = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", wav_path,
         "-codec:a", "libmp3lame", "-b:a", bitrate, mp3_path],
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
    fraction of the time of the full 104. It goes through exactly the same
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
    duck = D.duck_envelope(n, kicks, sr, depth=0.80, hold=0.010,
                           release=0.235, curve=1.75)

    if verbose:
        print(f"[4/5] mixing ({len(mx.channels)} channels, "
              f"{len(kicks)} kick triggers)")
    mix = mx.render(duck, verbose, keep_stems=keep_stems, parallel=parallel)

    mix *= build_section_gain(clock, n, sr)[:, None]

    if verbose:
        print("[5/5] mastering")
    master = master_chain(mix, sr, target_lufs=-9.3, ceiling_db=-1.0,
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

    mp3 = encode_mp3(outputs[0], f"{stem}.mp3")
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
