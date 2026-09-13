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

    def __init__(self, bpm=BPM, sr=48_000, swing=SWING):
        self.bpm = bpm
        self.sr = sr
        self.swing = swing
        self.beat = 60.0 / bpm             # seconds per quarter note
        self.bar = self.beat * 4.0
        self.step = self.beat / 4.0        # seconds per 16th

    def at(self, bar, step=0.0, swung=False):
        """Sample index of a position. `swung` pushes odd 16ths late."""
        offset = 0.0
        if swung and int(step) % 2 == 1:
            offset = self.swing * self.step
        return int((bar * self.bar + step * self.step + offset) * self.sr)

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
        sub_drop=True)

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
        clap="clap", filter_sweep=(2500, 18000))

    # --- 24 bars: main drop. Full arrangement plus arpeggio and vocals. ----
    add("drop2", 24, 1.00,
        kick=True, hat="hat", ohat=True, shaker=True, clap="clap_ghost",
        bass="bass_busy", stab="stab_synco", pad=True, rim=True,
        arp=True, vox=True, melody=True, crash_at=[0, 8, 16],
        sub_drop=True, fill_bars=[7, 15, 23])

    # --- 16 bars: DJ outro. Elements peel away, filter closes. -------------
    add("outro", 16, 0.35,
        kick=True, kick_until=14, hat="hat", shaker=True,
        bass="bass_simple", bass_until=8, stab="stab", stab_until=4,
        clap="clap", clap_until=8, filter_sweep=(16000, 700))

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
