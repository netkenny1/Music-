"""
The composition: tempo, key, harmony, rhythm patterns and song structure.

Musical decisions
-----------------
**124 BPM.** Squarely in the house pocket. Fast enough to drive, slow enough
that the offbeat hi-hat has room to breathe. It is also an easy tempo for a DJ
to beatmatch into 122-128 BPM sets.

**F minor.** Warm and slightly melancholy, the default emotional register of
deep and melodic house. The root at F2 (87.3 Hz) puts the bass fundamental
right where club subwoofers are most efficient.

**i - VI - III - VII** (Fm9 - Dbmaj7 - Abmaj7 - Ebadd9). Every chord is
diatonic and shares at least two notes with its neighbour, so the voicings can
move by step instead of leaping. That smooth voice leading is why the
progression feels like it circles rather than restarts.

**Swing on the 16ths, not the 8ths.** The kick and bass stay dead on the grid
so the track beatmatches cleanly; only the hats, shaker and stabs are nudged
late. That is the difference between a groove and a tempo drift.
"""

from dataclasses import dataclass, field

BPM = 124.0
KEY_NAME = "F minor"
SWING = 0.13          # fraction of a 16th that odd steps are pushed late
STEPS_PER_BAR = 16


# --------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------

class Clock:
    """Converts musical position (bar, step) into sample offsets."""

    def __init__(self, bpm=BPM, sr=48_000, swing=SWING, origin_bar=0):
        self.bpm = bpm
        self.sr = sr
        self.swing = swing
        self.beat = 60.0 / bpm             # seconds per quarter note
        self.bar = self.beat * 4.0
        self.step = self.beat / 4.0        # seconds per 16th
        # Bar that lands on sample 0. Non-zero for a partial render: every
        # position before it comes back negative, and Channel.add() clips
        # negatives, so events from before the window simply fall away.
        self.origin_bar = origin_bar

    def at(self, bar, step=0.0, swung=False):
        """Sample index of a position. `swung` pushes odd 16ths late."""
        offset = 0.0
        if swung and int(step) % 2 == 1:
            offset = self.swing * self.step
        return int(((bar - self.origin_bar) * self.bar
                    + step * self.step + offset) * self.sr)

    def dur(self, steps):
        """Sample length of a number of 16th notes."""
        return int(steps * self.step * self.sr)

    def bars_to_samples(self, bars):
        return int(bars * self.bar * self.sr)


# --------------------------------------------------------------------------
# Harmony
# --------------------------------------------------------------------------

@dataclass
class Chord:
    name: str
    bass: int          # MIDI note for the bassline root
    voicing: list      # MIDI notes for chords/pads (close voicing, ~G3-G4)
    arp: list          # MIDI notes used by the arpeggio, an octave up


# Voice leading across the four chords (top voice: G4 -> F4 -> G4 -> Bb4):
#   56 -> 56 -> 55 -> 58     (Ab3  Ab3  G3   Bb3)
#   60 -> 60 -> 60 -> 63     (C4   C4   C4   Eb4)
#   63 -> 61 -> 63 -> 65     (Eb4  Db4  Eb4  F4)
#   67 -> 65 -> 67 -> 70     (G4   F4   G4   Bb4)
# Every voice moves by a tone or less, or not at all.
PROGRESSION = [
    Chord("Fm9",     bass=41, voicing=[56, 60, 63, 67], arp=[68, 72, 75, 79]),
    Chord("Dbmaj7",  bass=37, voicing=[56, 60, 61, 65], arp=[68, 72, 73, 77]),
    Chord("Abmaj7",  bass=44, voicing=[55, 60, 63, 67], arp=[67, 72, 75, 79]),
    Chord("Ebadd9",  bass=39, voicing=[58, 63, 65, 70], arp=[70, 75, 77, 82]),
]

CHORD_BARS = 2        # each chord lasts two bars, so the cycle is 8 bars


def chord_at(bar):
    """Which chord is sounding in a given bar."""
    return PROGRESSION[(bar // CHORD_BARS) % len(PROGRESSION)]


# --------------------------------------------------------------------------
# Rhythm patterns
# --------------------------------------------------------------------------
# 16 characters = one bar of 16th notes.
#   '.' rest      'x' normal      'X' accent      'o' ghost/soft
VELOCITY = {"x": 1.00, "X": 1.15, "o": 0.55, ".": 0.0}

P = {
    # Four-on-the-floor. The foundation of the entire genre: a kick on every
    # quarter note gives dancers an unmissable pulse to lock to.
    "kick":        "X...x...x...x...",
    "kick_fill":   "X...x...x...x.xx",

    # Backbeat on 2 and 4.
    "clap":        "....X.......X...",
    "clap_ghost":  "....X.....o.X...",

    # Closed hats on every 16th with velocity accents. The alternation of
    # loud and soft is what makes a straight 16th line feel human.
    "hat":         "oXooxXoooXooxXoo",
    "hat_sparse":  "..x...x...x...x.",

    # The signature house open hat: dead on the offbeat 8ths, filling the
    # space between kicks. This one pattern is most of the genre's identity.
    "ohat":        "..x...x...x...x.",

    "shaker":      "o.xo.xo.o.xo.xo.",

    # Bass sits in the gaps between kicks, with a 16th pickup into the bar.
    "bass":        "..x...x...x...xx",
    "bass_busy":   "..xx..x.x.x...xx",
    "bass_simple": "..x...x...x...x.",

    # Offbeat chord stabs -- the same rhythmic role as the open hat, an
    # octave-and-a-half higher in the frequency range.
    "stab":        "..x...x...x...x.",
    "stab_synco":  "..x..x..x.x...x.",

    "rim":         "......x.......o.",

    # --- patterns built to syncopate -------------------------------------
    # The originals articulate every strong beat, which measures as zero
    # syncopation no matter how busy they look. These deliberately leave
    # quarter-note positions empty so earlier notes hold through them.

    # Kick with beat 3 missing. The pulse is established well enough by then
    # that the ear supplies the absent hit -- and feels its absence.
    "kick_hole3":  "X...x.......x...",
    # Kick with the DOWNBEAT missing. Only usable once, deep into a drop.
    "kick_hole1":  "....x...x...x...",

    # Thinner hats: accents on the offbeats only, so the quarter notes are
    # left to the kick rather than being doubled.
    "hat_thin":    "..X...X...X...X.",
    "hat_synco":   "o.X..o.X.o..X.o.",

    # Bass landing a 16th BEFORE the beat and holding through it.
    "bass_ante":   ".x.....x.....x..",
    "bass_synco":  "..x..x.....x..x.",

    # Chord stab that lands late and holds over the next strong beat.
    "stab_late":   "...x.....x....x.",
    # Clap pushed a 16th early -- arrives before the ear expects beat 2/4.
    "clap_push":   "...x.......x....",
    "arp":         "x.x.x.x.x.x.x.x.",
    "arp_dense":   "xxxxxxxxxxxxxxxx",
}


# --------------------------------------------------------------------------
# Arrangement
# --------------------------------------------------------------------------

@dataclass
class Section:
    """
    One block of the song.

    `energy` (0..1) drives automation: master filter opening, reverb amount,
    and hi-hat brightness all scale with it, so the track breathes across its
    runtime instead of being uniformly loud.
    """
    name: str
    start: int                 # first bar
    length: int                # bars
    energy: float
    parts: dict = field(default_factory=dict)

    @property
    def end(self):
        return self.start + self.length


def build_sections():
    """
    Structure, in DJ-friendly 8- and 16-bar blocks.

    The long beat-only intro and outro are deliberate: a club DJ needs 16 bars
    of unambiguous kick to align the track against the one already playing,
    with no melodic content to clash during the blend.
    """
    S = []
    b = 0

    def add(name, length, energy, **parts):
        nonlocal b
        S.append(Section(name, b, length, energy, parts))
        b += length

    # --- 16 bars: DJ intro. Drums only, filtered, gradually opening. -------
    add("intro", 16, 0.30,
        kick=True, hat="hat_sparse", shaker=True, rim=True,
        filter_sweep=(600, 9000), crash_at=[0])

    # --- 8 bars: first build. Bass and full hats arrive. -------------------
    add("build1", 8, 0.55,
        kick=True, hat="hat", ohat=True, shaker=True, clap="clap",
        bass="bass_simple", riser=True, filter_sweep=(4000, 18000))

    # --- 16 bars: drop one. Everything but the ear candy. ------------------
    add("drop1", 16, 0.90,
        kick=True, hat="hat", ohat=True, shaker=True, clap="clap_ghost",
        bass="bass", stab="stab", pad=True, rim=True, crash_at=[0, 8],
        sub_drop=True,
        # first violations, used sparingly: the groove is still being taught
        kick_pattern_bars={7: "kick_hole3"},
        clap_push_bars=[11],
        stutter_at=[(15, 12, "stab")])

    # --- 16 bars: breakdown. Kick drops out for 8 bars, harmony takes over.
    add("break", 16, 0.45,
        kick_from=8, hat_from=12, hat="hat_sparse",
        pad=True, keys=True, vox=True, melody=True,
        clap_from=12, clap="clap", downlifter=True,
        reverse_crash_at=[15], filter_sweep=(1200, 14000))

    # --- 8 bars: second build. Snare roll, riser, everything tightening. ---
    add("build2", 8, 0.75,
        kick=True, hat="hat", shaker=True, bass="bass_simple",
        pad=True, stab="stab", snare_roll=True, riser=True,
        clap="clap", filter_sweep=(2500, 18000),
        stutter_at=[(7, 8, "stab")],
        # total silence on the last beat. The riser stops, everything stops,
        # and the listener is left holding a prediction with nothing to meet it.
        silence_from=(7, 12))

    # --- 24 bars: main drop. Full arrangement plus arpeggio and vocals. ----
    add("drop2", 24, 1.00,
        kick=True, hat="hat", ohat=True, shaker=True, clap="clap_ghost",
        bass="bass_busy", stab="stab_synco", pad=True, rim=True, keys=True,
        arp=True, vox=True, melody=True, crash_at=[8, 16],
        counter=True, counter_from=12,
        sub_drop=True, fill_bars=[7, 15, 23],
        # THE DELAYED DROP. Bar 0 of the drop is a hole: no kick, no groove,
        # just a sub and the tail of the build hanging in the air. The kick
        # then arrives EARLY, on the last 8th of the bar, so the beat both
        # fails to arrive when expected and then pre-empts the next downbeat.
        hole_bar=0,
        kick_pattern_bars={7: "kick_hole3", 15: "kick_hole1", 19: "kick_hole3"},
        clap_push_bars=[11, 19],
        bass_ante_bars=[5, 13, 21],
        stab_late_bars=[9, 17],
        poly_from=8,
        stutter_at=[(23, 8, "arp")])

    # --- 16 bars: DJ outro. Elements peel away, filter closes. -------------
    add("outro", 16, 0.35,
        kick=True, kick_until=14, hat="hat", shaker=True,
        bass="bass_simple", bass_until=8, stab="stab", stab_until=4,
        clap="clap", clap_until=8, filter_sweep=(16000, 900))

    return S


SECTIONS = build_sections()
TOTAL_BARS = SECTIONS[-1].end


# --------------------------------------------------------------------------
# The hook
# --------------------------------------------------------------------------
# (step within the 8-bar cycle, MIDI note, length in 16ths).
# Built from the F natural-minor scale and phrased as call-and-response: a
# rising two-bar question over Fm9/Dbmaj7, answered by a falling one over
# Abmaj7/Ebadd9.
MELODY = [
    (4, 72, 2), (6, 75, 2), (8, 77, 4), (14, 72, 2),          # Fm9
    (16, 68, 4), (22, 72, 2), (24, 70, 6),
    (36, 68, 2), (38, 72, 2), (40, 73, 4), (46, 72, 2),       # Dbmaj7
    (48, 68, 8), (58, 65, 4),
    (68, 75, 2), (70, 72, 2), (72, 75, 4), (78, 77, 2),       # Abmaj7
    (80, 75, 6), (88, 72, 4),
    (100, 70, 2), (102, 72, 2), (104, 75, 4), (110, 77, 2),   # Ebadd9
    (112, 75, 8), (122, 72, 4),
]

# Counter-melody: answers the hook in the gaps it leaves. Every note is a
# chord tone, and each phrase sits a fifth or more above the hook so the two
# lines never cross. The last phrase falls Bb -> G -> F, landing on the root
# of the Fm9 that starts the next cycle -- the resolution is what makes the
# repeat feel earned rather than looped.
COUNTER = [
    (26, 79, 2), (28, 80, 2), (30, 77, 4),                    # Fm9   (G Ab F)
    (60, 80, 2), (62, 77, 2), (64, 73, 4),                    # Dbmaj7 (Ab F Db)
    (90, 79, 2), (92, 80, 2), (94, 84, 2), (96, 79, 4),       # Abmaj7 (G Ab C G)
    (114, 82, 2), (116, 79, 2), (118, 77, 6),                 # Ebadd9 (Bb G F)
]


def groove_report():
    """
    Measure syncopation, and be honest about what the number means.

    Two surfaces are reported, because they say different things:

    * **Full kit.** In four-to-the-floor house this is near zero by design and
      that is correct, not a failure. A continuous 16th hat plus a kick on
      every quarter articulates every metrical position, so nothing is ever
      left hanging. That saturation is exactly what makes the genre danceable:
      it is the stable grid the violations are heard against.

    * **Bass and chords.** This is where bar-level syncopation actually lives
      in house, and it is where the index is worth reading.

    The structural violations -- the delayed drop, a missing downbeat, an
    accelerating stutter, a bar of silence -- do not show up in either number.
    They operate across phrases, not within a bar, and they are counted
    separately below.
    """
    import groove as G

    kit, mel = [], []
    for name, drums, tuned in [
        ("drop1", ["kick", "clap_ghost", "hat", "ohat"], ["bass", "stab"]),
        ("drop2", ["kick", "clap_ghost", "hat", "ohat"],
                  ["bass_busy", "stab_synco"]),
        ("violation bar", ["kick_hole1", "clap_push", "hat_synco", "ohat"],
                          ["bass_ante", "stab_late"]),
    ]:
        k = G.combine(*[P[x] for x in drums], accents_only=True)
        m = G.combine(*[P[x] for x in tuned], accents_only=True)
        kit.append((name, k, G.syncopation(k)))
        mel.append((name, m, G.syncopation(m)))
    return kit, mel


def violation_report():
    """Count the structural expectation violations, and where they land."""
    events = []
    for sec in SECTIONS:
        p = sec.parts
        if p.get("hole_bar") is not None:
            events.append((sec.start + p["hole_bar"], "delayed drop",
                           "downbeat withheld; kick enters early on the last 8th"))
        for lb, pat in p.get("kick_pattern_bars", {}).items():
            what = ("downbeat kick removed" if pat == "kick_hole1"
                    else "beat-3 kick removed")
            events.append((sec.start + lb, "missing kick", what))
        for lb in p.get("clap_push_bars", []):
            events.append((sec.start + lb, "pushed clap",
                           "backbeat arrives a 16th early"))
        for lb in p.get("bass_ante_bars", []):
            events.append((sec.start + lb, "anticipated bass",
                           "bass lands before the beat and holds through it"))
        for lb in p.get("stab_late_bars", []):
            events.append((sec.start + lb, "late chord",
                           "stab lands after the beat"))
        for (lb, step, part) in p.get("stutter_at", []):
            events.append((sec.start + lb, "stutter",
                           f"accelerating {part} retrigger from step {step}"))
        if p.get("silence_from"):
            lb, step = p["silence_from"]
            events.append((sec.start + lb, "silence",
                           f"everything stops from step {step}"))
        if p.get("poly_from") is not None:
            events.append((sec.start + p["poly_from"], "polyrhythm",
                           "3-against-4 layer, realigns every 3 bars"))
    return sorted(events)


def describe():
    """Human-readable summary of the arrangement."""
    lines = [
        f"Key      : {KEY_NAME}",
        f"Tempo    : {BPM:g} BPM   (bar = {4 * 60 / BPM:.3f}s)",
        f"Swing    : {SWING * 100:.0f}% on 16ths",
        f"Harmony  : " + " -> ".join(c.name for c in PROGRESSION) +
        f"  ({CHORD_BARS} bars each)",
        f"Length   : {TOTAL_BARS} bars = {TOTAL_BARS * 4 * 60 / BPM:.1f}s",
        "",
        "Structure:",
    ]
    for s in SECTIONS:
        t = s.start * 4 * 60 / BPM
        lines.append(f"  {int(t)//60:d}:{int(t)%60:02d}  bar {s.start:3d}  "
                     f"{s.name:8s} {s.length:2d} bars   energy {s.energy:.2f}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
    print()
    kit, mel = groove_report()
    print("Syncopation, Longuet-Higgins & Lee index of the accent surface:")
    print("  full kit (saturated by design -- near zero is correct):")
    for name, surf, idx in kit:
        print(f"    {name:14s} {surf}  index {idx:3d}")
    print("  bass + chords (where syncopation lives in house):")
    for name, surf, idx in mel:
        print(f"    {name:14s} {surf}  index {idx:3d}")
    print()
    print("Structural expectation violations:")
    for bar, kind, detail in violation_report():
        t = bar * 4 * 60 / BPM
        print(f"  {int(t)//60}:{int(t)%60:02d}  bar {bar:3d}  {kind:18s} {detail}")
