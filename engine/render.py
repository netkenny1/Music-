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
        a = clock.at(sec.start)
        b = min(clock.at(sec.end), n)
        if b <= a:
            continue
        sweep = sec.parts.get("filter_sweep")
        if sweep is None:
            curve[a:b] = 20000.0
        else:
            f0, f1 = sweep
            t = np.linspace(0.0, 1.0, b - a)
            curve[a:b] = f0 * (f1 / f0) ** (t ** 0.85)
    # A short smoothing pass stops the joins between sections from clicking.
    w = int(0.05 * sr)
    win = np.hanning(w) / np.hanning(w).sum()
    pad = np.concatenate([np.full(w, curve[0]), curve, np.full(w, curve[-1])])
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

def sequence(mx, clock, cache, sr=SR, verbose=True):
    """Walk the arrangement and schedule every note into the mixer."""
    rng = np.random.default_rng(2024)
    kick_triggers = []

    def humanize(pos, amount_ms=1.6):
        """Tiny timing jitter. Perfectly quantised percussion is rhythmically
        correct and emotionally dead; a millisecond or two of scatter is what
        the ear reads as a person playing."""
        return pos + int(rng.normal(0.0, amount_ms) * sr / 1000.0)

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
            chord = C.chord_at(bar)
            fills = sec.parts.get("fill_bars", [])
            is_fill = lb in fills

            # ---------------- kick ----------------------------------------
            if part_state(sec, "kick", lb) is not None:
                pat = "kick_fill" if is_fill else "kick"
                for step, vel in pattern_hits(pat):
                    pos = clock.at(bar, step)
                    kick_triggers.append(pos)
                    mx.channels["kick"].add(
                        cache.pick(cache.kick, nxt("kick")) * vel, pos)

            # ---------------- clap ----------------------------------------
            cp = part_state(sec, "clap", lb)
            if cp:
                name = cp if isinstance(cp, str) else "clap"
                for step, vel in pattern_hits(name):
                    pos = humanize(clock.at(bar, step, swung=True), 1.1)
                    mx.channels["clap"].add(
                        cache.pick(cache.clap, nxt("clap")) * vel *
                        rng.uniform(0.93, 1.0), pos)

            # ---------------- hats ----------------------------------------
            hp = part_state(sec, "hat", lb)
            if hp:
                name = hp if isinstance(hp, str) else "hat"
                for step, vel in pattern_hits(name):
                    pos = humanize(clock.at(bar, step, swung=True), 1.5)
                    # brightness tracks section energy: quieter sections get
                    # darker hats, which reads as "further away"
                    g = vel * rng.uniform(0.88, 1.06) * (0.72 + 0.35 * sec.energy)
                    mx.channels["hat"].add(
                        cache.pick(cache.hat, nxt("hat")) * g, pos)

            if part_state(sec, "ohat", lb):
                for step, vel in pattern_hits("ohat"):
                    pos = humanize(clock.at(bar, step, swung=True), 1.4)
                    mx.channels["ohat"].add(
                        cache.pick(cache.ohat, nxt("ohat")) * vel *
                        rng.uniform(0.9, 1.05), pos)

            if part_state(sec, "shaker", lb):
                for step, vel in pattern_hits("shaker"):
                    pos = humanize(clock.at(bar, step, swung=True), 2.0)
                    mx.channels["shaker"].add(
                        cache.pick(cache.shaker, nxt("shaker")) * vel *
                        rng.uniform(0.82, 1.08), pos)

            if part_state(sec, "rim", lb) and lb % 2 == 1:
                for step, vel in pattern_hits("rim"):
                    pos = humanize(clock.at(bar, step, swung=True), 2.2)
                    mx.channels["rim"].add(
                        cache.pick(cache.rim, nxt("rim")) * vel, pos)

            # ---------------- bass ----------------------------------------
            bp = part_state(sec, "bass", lb)
            if bp:
                name = bp if isinstance(bp, str) else "bass"
                hits = list(pattern_hits(name))
                for k, (step, vel) in enumerate(hits):
                    # a note lasts until the next one, so the line is legato
                    nxt_step = hits[k + 1][0] if k + 1 < len(hits) else 16
                    steps = min(nxt_step - step, 4)
                    # octave lift on the pickup into the next bar
                    note = chord.bass + (12 if step >= 15 else 0)
                    dur = clock.dur(steps) / sr + 0.06
                    b = I.bass(midi_to_hz(note), dur, sr,
                               cutoff=340.0 + 260.0 * sec.energy,
                               res=1.7, saw_level=0.42 + 0.22 * sec.energy,
                               drive=1.5 + 0.3 * sec.energy,
                               seed=83 + bar * 3 + step)
                    mx.channels["bass"].add(b * vel, clock.at(bar, step))

            # ---------------- chord stabs ---------------------------------
            sp = part_state(sec, "stab", lb)
            if sp:
                name = sp if isinstance(sp, str) else "stab"
                for step, vel in pattern_hits(name):
                    pos = clock.at(bar, step, swung=True)
                    st = I.stab([midi_to_hz(m) for m in chord.voicing],
                                clock.dur(3) / sr + 0.18, sr,
                                cutoff=1700.0 + 1600.0 * sec.energy,
                                decay=0.20, detune=11.0,
                                seed=97 + bar * 5 + step)
                    mx.channels["stab"].add(st * vel * 0.9, pos)

            # ---------------- pad (one long note per chord) ----------------
            if part_state(sec, "pad", lb) and bar % C.CHORD_BARS == 0:
                dur = clock.bar_seconds = C.CHORD_BARS * clock.bar + 1.1
                p = I.pad([midi_to_hz(m) for m in chord.voicing], dur, sr,
                          cutoff=900.0 + 1400.0 * sec.energy,
                          attack=0.8, release=1.3, seed=101 + bar)
                mx.channels["pad"].add(p, clock.at(bar) - int(0.05 * sr))

            # ---------------- breakdown keys ------------------------------
            if part_state(sec, "keys", lb) and lb % 2 == 0:
                for step in (0, 6, 10):
                    k = I.keys([midi_to_hz(m) for m in chord.voicing],
                               clock.dur(6) / sr, sr, decay=0.85,
                               seed=113 + bar * 3 + step)
                    mx.channels["keys"].add(
                        k * (1.0 if step == 0 else 0.6),
                        clock.at(bar, step, swung=True))

            # ---------------- arpeggio ear candy --------------------------
            if part_state(sec, "arp", lb):
                notes = chord.arp + chord.arp[-2::-1]      # up then back down
                for step, vel in pattern_hits("arp"):
                    note = notes[(step // 2 + lb) % len(notes)]
                    pos = clock.at(bar, step, swung=True)
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
                mx.channels["vox"].add(v * 0.9, clock.at(bar, 2, swung=True))

            # ---------------- melodic hook --------------------------------
            if part_state(sec, "melody", lb):
                cycle_start = (bar // 8) * 8
                for mstep, note, mlen in C.MELODY:
                    mbar = cycle_start + mstep // 16
                    if mbar != bar:
                        continue
                    step = mstep % 16
                    pos = clock.at(bar, step, swung=True)
                    dur = clock.dur(mlen) / sr + 0.25
                    if sec.name == "break":
                        tone = I.keys([midi_to_hz(note)], dur, sr,
                                      decay=dur * 0.8, seed=113 + mstep)
                    else:
                        tone = I.pluck(midi_to_hz(note), dur, sr,
                                       decay=dur * 0.55, cutoff=3600.0,
                                       seed=131 + mstep)
                    mx.channels["melody"].add(tone * 0.85, pos)

            # ---------------- snare roll ----------------------------------
            if part_state(sec, "snare_roll", lb) and lb >= sec.length - 4:
                k = lb - (sec.length - 4)
                div = [2.0, 2.0, 1.0, 0.5][k]          # 8ths -> 32nds
                step = 0.0
                while step < 16:
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

            if part_state(sec, "sub_drop", lb) and lb == 0:
                mx.channels["sub"].add(I.sub_drop(1.5, sr) * 0.75, clock.at(bar))

            if part_state(sec, "riser", lb) and lb == sec.length - 4:
                dur = 4 * clock.bar
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
           gain_db=-13.0, eq=[F.highpass(400.0, 0.707, sr),
                              F.lowpass(9000.0, 0.707, sr)], width=1.1)

    mx.bus("plate", S.reverb_ir(sr, rt60=1.9, predelay=0.022, damping=0.45,
                                width=1.15, er_level=0.35, seed=17),
           gain_db=-15.0, eq=[F.highpass(320.0, 0.707, sr),
                              F.lowpass(11000.0, 0.707, sr)],
           width=1.2, duck=0.35)

    mx.bus("hall", S.reverb_ir(sr, rt60=3.6, predelay=0.045, damping=0.62,
                               width=1.3, er_level=0.25, seed=27),
           gain_db=-17.0, eq=[F.highpass(260.0, 0.707, sr),
                              F.lowpass(8000.0, 0.707, sr)],
           width=1.35, duck=0.45)

    # Dotted-eighth delay: 0.75 of a beat. It lands between the 16ths instead
    # of on top of them, so echoes add motion without thickening the groove.
    mx.bus("delay", S.delay_ir(sr, time=beat * 0.75, feedback=0.40,
                               repeats=12, ping_pong=True, damping=0.55),
           gain_db=-16.0, eq=[F.highpass(380.0, 0.707, sr),
                              F.lowpass(7000.0, 0.707, sr)],
           width=1.3, duck=0.4)

    # --- channels ---------------------------------------------------------
    mx.channel("kick", gain_db=-5.5, pan=0.0,
               comp=dict(threshold=-12.0, ratio=2.2, attack=0.012,
                         release=0.120, knee=4.0, makeup=1.0),
               sends={"room": 0.05})

    mx.channel("sub", gain_db=-13.0, pan=0.0, mono_below=200.0,
               hp=26.0, lp=140.0)

    mx.channel("bass", gain_db=-8.0, pan=0.0, duck=0.78, mono_below=140.0,
               hp=28.0,
               eq=[F.peaking(95.0, 1.6, 1.0, sr),
                   F.peaking(280.0, -3.0, 1.0, sr),
                   F.peaking(1200.0, 1.2, 0.9, sr)],
               comp=dict(threshold=-20.0, ratio=3.5, attack=0.006,
                         release=0.085, knee=5.0, makeup=3.0),
               filter_curve=fcurve)

    mx.channel("clap", gain_db=-13.5, pan=0.0, width=1.25, hp=220.0,
               comp=dict(threshold=-20.0, ratio=2.5, attack=0.003,
                         release=0.100, makeup=2.0),
               sends={"room": 0.40, "plate": 0.16})

    mx.channel("hat", gain_db=-19.0, pan=0.13, width=1.15, hp=420.0,
               eq=[F.highshelf(10000.0, 2.0, 0.7, sr)],
               sends={"room": 0.12}, filter_curve=fcurve)

    mx.channel("ohat", gain_db=-20.5, pan=-0.20, width=1.22, hp=420.0,
               duck=0.30, sends={"room": 0.16}, filter_curve=fcurve)

    mx.channel("shaker", gain_db=-26.0, pan=0.40, width=1.1, hp=2500.0,
               sends={"room": 0.10}, filter_curve=fcurve)

    mx.channel("rim", gain_db=-23.0, pan=-0.44, hp=300.0,
               sends={"room": 0.22, "delay": 0.12}, filter_curve=fcurve)

    mx.channel("stab", gain_db=-15.0, width=1.30, duck=0.62, hp=170.0,
               eq=[F.peaking(430.0, -3.2, 1.0, sr),
                   F.peaking(5200.0, 2.0, 0.9, sr)],
               comp=dict(threshold=-22.0, ratio=2.5, attack=0.008,
                         release=0.130, makeup=2.5),
               sends={"plate": 0.30, "delay": 0.18, "room": 0.08},
               filter_curve=fcurve)

    mx.channel("pad", gain_db=-21.0, width=1.50, duck=0.55, hp=150.0,
               eq=[F.peaking(330.0, -3.5, 0.9, sr),
                   F.highshelf(9000.0, 1.5, 0.7, sr)],
               sends={"hall": 0.55, "plate": 0.15},
               filter_curve=fcurve)

    mx.channel("keys", gain_db=-17.0, width=1.20, duck=0.35, hp=200.0,
               eq=[F.peaking(400.0, -2.0, 1.0, sr)],
               sends={"plate": 0.34, "delay": 0.18, "hall": 0.12},
               filter_curve=fcurve)

    mx.channel("arp", gain_db=-23.0, width=1.25, duck=0.42, hp=420.0,
               eq=[F.highshelf(8000.0, 1.5, 0.7, sr)],
               sends={"delay": 0.48, "plate": 0.22},
               filter_curve=fcurve)

    mx.channel("vox", gain_db=-21.0, width=1.18, duck=0.45, hp=220.0,
               eq=[F.peaking(3000.0, 2.0, 1.0, sr)],
               sends={"hall": 0.40, "delay": 0.22, "plate": 0.18},
               filter_curve=fcurve)

    mx.channel("melody", gain_db=-19.0, pan=-0.16, width=1.1, duck=0.35,
               hp=250.0, sends={"plate": 0.32, "delay": 0.30},
               filter_curve=fcurve)

    mx.channel("fx", gain_db=-17.0, width=1.40, hp=180.0,
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


# ==========================================================================
# Main
# ==========================================================================

def build_track(sr=SR, verbose=True):
    t0 = time.time()
    clock = C.Clock(C.BPM, sr, C.SWING)
    n = clock.bars_to_samples(C.TOTAL_BARS) + int(5.0 * sr)   # room for tails

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
    kicks = sequence(mx, clock, cache, sr, verbose)

    # The pump follows the actual kick events, so it stops automatically
    # wherever the kick drops out.
    duck = D.duck_envelope(n, kicks, sr, depth=0.80, hold=0.010,
                           release=0.235, curve=1.75)

    if verbose:
        print(f"[4/5] mixing ({len(mx.channels)} channels, "
              f"{len(kicks)} kick triggers)")
    mix = mx.render(duck, verbose)

    if verbose:
        print("[5/5] mastering")
    master = master_chain(mix, sr, target_peak_db=-0.9, verbose=verbose)

    if verbose:
        print(f"\nrendered in {time.time() - t0:.1f}s")
    return master, mix, clock


def main():
    ap = argparse.ArgumentParser(description="Render the house track.")
    ap.add_argument("--out", default="output", help="output directory")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    verbose = not args.quiet
    master, mix, clock = build_track(SR, verbose)

    os.makedirs(args.out, exist_ok=True)
    stem = os.path.join(args.out, "midnight_transit")

    write_wav(f"{stem}_master_24bit_48k.wav", master, SR, 24)

    from scipy.signal import resample_poly
    cd = np.stack([resample_poly(master[:, c], 147, 160) for c in range(2)], axis=-1)
    cd = np.clip(cd, -db(-0.9), db(-0.9))
    write_wav(f"{stem}_master_16bit_44k.wav", cd, 44100, 16)

    print()
    print(A.report(master, SR, "MASTER"))

    return stem


if __name__ == "__main__":
    main()
