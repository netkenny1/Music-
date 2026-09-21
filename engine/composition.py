"""
The composition: tempo, key, harmony, rhythm patterns and song structure.

What changed, and why
---------------------
The previous arrangement was *busy*: a 16th-note background texture,
tambourine, congas, bells, a counter-melody, hat patterns rotating every two
bars, stutters, mini-drops, pushed claps, a polyrhythm. Everything a
production-music library track does to hold attention for thirty seconds.
That is exactly what made it sound like an advert.

A club record does the opposite. It is an 8-bar loop played for six minutes,
with maybe seven parts, and the interest comes from *phrasing*: what enters
and leaves on the 8-, 16- and 32-bar boundaries, the fill that announces
each boundary, the one breakdown, the one build. The listener is meant to
lock in, not be entertained. So this file now describes that:

* **The kit.** Kick on every quarter. Clap on 2 and 4. Closed hats on the
  16ths with the swung off-16ths as ghost notes. Open hat on every off-beat
  8th -- the single most identifying sound in house. A shaker for air.
  Nothing else on the top.
* **The bass.** Off-beat 8ths, legato, so each note holds through the next
  kick and gets pumped by the sidechain. Root of the chord, a fifth-approach
  into each chord change.
* **The chords.** One stab riff, two bars long, one chord per two bars.
  One pad under it, sidechained hard, so the chords breathe with the kick.
* **The phrasing.** 32-bar DJ intro, 32 bars of groove, 16-bar breakdown,
  16-bar build, 64 bars of drop in two halves, 32-bar DJ outro. Every
  8th bar ends with a small fill, every 16th with a bigger one, every 32nd
  with a crash. That grid is the form of the genre.

Musical decisions kept from before
----------------------------------
**124 BPM.** Squarely in the house pocket, easy to beatmatch into a
122-128 set.

**F minor.** Warm and slightly melancholy. The root at F2 (87.3 Hz) puts the
bass fundamental where club subwoofers are most efficient.

**i - VI - III - VII** (Fm9 - Dbmaj7 - Abmaj7 - Ebadd9). Every chord is
diatonic and shares two notes with its neighbour, so the voicings move by
step. The VII (Eb) has no leading tone to F, which is why the loop circles
instead of cadencing: a house progression is meant to feel endless.

**Swing on the 16ths, not the 8ths.** Kick and bass are dead on the grid so
the track beatmatches; only the odd 16ths (hat ghosts, shaker) are pushed
late. 14 % of a 16th is a 57 % MPC swing -- the house shuffle, not a
triplet feel.
"""

from dataclasses import dataclass, field

BPM = 124.0
KEY_NAME = "F minor"
SWING = 0.14          # fraction of a 16th that odd steps are pushed late
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
    top: int           # MIDI note the vocal hook sings over this chord


# Voice leading across the four chords (top voice: G4 -> F4 -> G4 -> Bb4):
#   56 -> 56 -> 55 -> 58     (Ab3  Ab3  G3   Bb3)
#   60 -> 60 -> 60 -> 63     (C4   C4   C4   Eb4)
#   63 -> 61 -> 63 -> 65     (Eb4  Db4  Eb4  F4)
#   67 -> 65 -> 67 -> 70     (G4   F4   G4   Bb4)
# Every voice moves by a tone or less, or not at all.
PROGRESSION = [
    Chord("Fm9",     bass=41, voicing=[56, 60, 63, 67], top=72),   # C5
    Chord("Dbmaj7",  bass=37, voicing=[56, 60, 61, 65], top=72),   # C5
    Chord("Abmaj7",  bass=44, voicing=[55, 60, 63, 67], top=75),   # Eb5
    Chord("Ebadd9",  bass=39, voicing=[58, 63, 65, 70], top=77),   # F5
]

CHORD_BARS = 2        # each chord lasts two bars, so the cycle is 8 bars
BASS_LOW, BASS_HIGH = 34, 46     # Bb1 .. Bb2: where a house bass lives


def chord_at(bar):
    """Which chord is sounding in a given bar."""
    return PROGRESSION[(bar // CHORD_BARS) % len(PROGRESSION)]


def approach_note(bar):
    """
    The bass note that leads into the next chord.

    The last off-beat before a chord change plays the *fifth of the chord
    that is coming*, kept inside the bass register. Fifth-to-root is the
    strongest pull in tonal music (it is what a dominant does), and disco and
    house basslines have used it as their standard turnaround for fifty
    years: the ear hears the fifth, expects the root, and the new chord
    lands on the downbeat as the answer.
    """
    nxt = chord_at(bar + 1)
    fifth = nxt.bass + 7
    if fifth > BASS_HIGH:
        fifth -= 12
    return fifth


# --------------------------------------------------------------------------
# Rhythm patterns
# --------------------------------------------------------------------------
# 16 characters = one bar of 16th notes.
#   '.' rest      'x' normal      'X' accent      'o' ghost/soft
VELOCITY = {"x": 1.00, "X": 1.15, "o": 0.50, ".": 0.0}

P = {
    # Four-on-the-floor. A kick on every quarter note is the entire genre.
    "kick":        "X...x...x...x...",
    # Beat 4 removed: the fill at the end of a 16-bar phrase. The pulse is
    # so well established by then that the missing kick is a *gesture*,
    # a breath before the next phrase lands.
    "kick_drop4":  "X...x...x.......",
    # Doubled to 8ths for the last two bars of the build.
    "kick_8ths":   "x.x.x.x.x.x.x.x.",

    # Backbeat on 2 and 4.
    "clap":        "....X.......X...",

    # Closed hats. Per beat: accent on the kick, a swung ghost, NOTHING on
    # the off-8th (the open hat lives there -- a hi-hat cannot be open and
    # closed at once, so the closed hat is choked out of that slot), then
    # another ghost. The ghosts are what the swing acts on; this is the
    # house shuffle.
    "hat":         "Xo.oXo.oXo.oXo.o",
    # Intro hats: straight 8ths, no ghosts, before the groove is revealed.
    "hat_8ths":    "x...x...x...x...",

    # THE open hat: dead on the off-beat 8ths, in the space between kicks.
    # Kick-and-open-hat is the pendulum every house record swings on.
    "ohat":        "..x...x...x...x.",

    # Shaker on the 16ths, accents with the open hat. Air, not a part.
    "shaker":      "o.xo.oxo.oxo.oxo",

    # Bass on the off-beat 8ths, a 16th pickup into the next bar. Each note
    # is legato into the next, so it holds through the kick and pumps.
    "bass":        "..x...x...x...x.",
    # Second half of each 8: the octave bounces on the 16th after beat 3,
    # which is the "bouncing" house bassline.
    "bass_bounce": "..x...x.x.x...x.",

    # The chord riff, two bars. Bar one is plain off-beats; bar two pulls
    # the third hit a 16th early and drops the last one, so every two bars
    # there is one small syncopation that resolves on the next downbeat.
    "stab_a":      "..X...x...X...x.",
    "stab_b":      "..X...x.x...X...",

    # Electric piano riff for the second half of the drop: the "x..x..x."
    # tresillo, the Afro-Cuban cell that Chicago house borrowed from disco.
    "keys":        "x..x..x.........",

    # Fills. Clap on beat 4 as 16ths, velocity rising: the 8-bar fill.
    "fill_8":      "............oxxX",
    # Beats 3 and 4, 8ths then 16ths: the 16-bar fill.
    "fill_16":     "........x.x.oxxX",
}


# --------------------------------------------------------------------------
# Arrangement
# --------------------------------------------------------------------------

@dataclass
class Section:
    """
    One block of the song.

    `energy` (0..1) drives automation: master filter opening and hi-hat
    brightness scale with it, so the track breathes across its runtime.
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
    The club edit: 192 bars, 6:11, in 8-, 16- and 32-bar blocks.

    Every part enters or leaves on a phrase boundary. That is not a
    convention for its own sake -- a DJ counts in phrases, and a crowd
    hears in phrases; an element arriving on bar 13 reads as a mistake.
    """
    S = []
    b = 0

    def add(name, length, energy, **parts):
        nonlocal b
        S.append(Section(name, b, length, energy, parts))
        b += length

    # --- 32 bars: DJ intro. One element every 8 bars. ---------------------
    # Bars 0-7 kick and 8th-note hats. 8-15 the open hat and clap: the
    # groove is now identifiable as house. 16-23 the bass, dark. 24-31 the
    # stab riff arrives under a closing-then-opening filter.
    add("intro", 32, 0.45,
        kick=True, hat="hat_8ths", hat_from=0,
        hat16_from=8, ohat_from=8, clap_from=8, shaker_from=16,
        bass_from=16, stab_from=24, sub_layer=True,
        filter_sweep=(2200, 20000), crash_at=[0, 16])

    # --- 32 bars: the groove. Everything in, the pad from bar 8, the vocal
    # hook from bar 16. Fills at 7, 15, 23, 31. ----------------------------
    add("main_a", 32, 0.85,
        kick=True, hat="hat", ohat=True, clap=True, shaker=True,
        bass=True, sub_layer=True, stab=True, pad_from=8, vox_from=16,
        crash_at=[0, 16], bounce_second_half=True)

    # --- 16 bars: breakdown. Drums out. Pad, electric piano, the vocal. --
    # The kick's absence is the point: sixteen bars of nothing to dance to,
    # so that its return in the build is felt in the chest.
    add("break", 16, 0.40,
        pad=True, keys_chords=True, vox=True, shaker_from=8,
        downlifter=True, filter_sweep=(900, 16000))

    # --- 16 bars: build. Kick back at 0. Bass leaves at 8 (no low end is
    # the tension). Snare roll over the last 4 bars, riser over the last 8,
    # kick doubles to 8ths for the last 2, and the last beat is cut. -------
    add("build", 16, 0.75,
        kick=True, hat="hat", hat_from=4, ohat_from=8, clap=True,
        shaker=True, bass_until=8, sub_layer=True, stab=True, pad=True,
        snare_roll=True, riser=True, kick_8ths_from=14,
        filter_sweep=(6000, 20000), silence_from=(15, 12),
        reverse_crash_at=[15])

    # --- 32 bars: THE DROP. Full groove, bounce bass, crash, sub drop. ----
    add("drop", 32, 1.00,
        kick=True, hat="hat", ohat=True, clap=True, shaker=True,
        bass=True, sub_layer=True, stab=True, pad=True, vox=True,
        crash_at=[0, 16], sub_drop=True, bounce_second_half=True)

    # --- 32 bars: second half of the drop. The electric-piano riff is the
    # new element (0-15). Bars 16-23 thin out -- hats and stabs gone, kick,
    # bass and pad only -- and 24-31 bring it all back for the last run. --
    add("drop_b", 32, 1.00,
        kick=True, hat="hat", ohat=True, clap=True, shaker=True,
        bass=True, sub_layer=True, stab=True, pad=True, vox=True,
        keys_riff=True, keys_riff_until=16,
        thin_bars=range(16, 24),
        crash_at=[0, 16, 24], bounce_second_half=True)

    # --- 32 bars: DJ outro. Elements leave every 8, filter closes. --------
    add("outro", 32, 0.60,
        kick=True, hat="hat", ohat=True, clap=True, clap_until=24,
        shaker=True, shaker_until=16, bass=True, bass_until=16,
        sub_layer=True, stab=True, stab_until=8, pad=True, pad_until=8,
        ohat_until=24, filter_sweep=(20000, 1400), crash_at=[0])

    return S


SECTIONS = build_sections()
TOTAL_BARS = SECTIONS[-1].end

# Sections where the groove is running and the 8/16/32-bar fills apply.
GROOVE_SECTIONS = {"intro", "main_a", "drop", "drop_b", "outro"}


def fill_kind(sec, local_bar):
    """
    Which fill, if any, ends this bar.

    The fill lives in the LAST bar of a phrase. Bar 7 of every 8 gets the
    small clap roll; bar 15 of every 16 the bigger roll with the kick's
    fourth beat removed; the last bar of a section additionally gets the
    reverse cymbal into the next downbeat. A listener never counts bars,
    but they feel the 8s -- and the fill is what tells them where they are.
    """
    if sec.name not in GROOVE_SECTIONS:
        return None
    if (local_bar + 1) % 16 == 0:
        return "fill_16"
    if (local_bar + 1) % 8 == 0:
        return "fill_8"
    return None


# --------------------------------------------------------------------------
# The vocal hook
# --------------------------------------------------------------------------
# One short phrase per four bars: (bar within the 4, step, vowel, 16ths).
# Two syllables, off the beat, on the chord's top note. Restraint is the
# point -- a club vocal is a sample that repeats, not a melody.
VOX_HOOK = [
    (2, 2, "ooh", 6),
    (2, 10, "ah", 4),
    (3, 6, "ah", 3),
]


def describe():
    """Human-readable summary of the arrangement."""
    lines = [
        f"Key      : {KEY_NAME}",
        f"Tempo    : {BPM:g} BPM   (bar = {4 * 60 / BPM:.3f}s)",
        f"Swing    : {SWING * 100:.0f}% of a 16th  "
        f"(MPC {50 + SWING * 50:.0f}%)",
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


def groove_report():
    """
    Longuet-Higgins & Lee syncopation of the loop, for the README.

    The full kit measures near zero -- correct for four-on-the-floor, where
    every metrical position is articulated. The bass and chords are where
    the syncopation lives: off-beat 8ths held through the beat score 7 per
    bar, and the stab riff's early hit in bar two adds to that.
    """
    import groove as G
    kit = G.combine(P["kick"], P["clap"], P["hat"], P["ohat"], accents_only=True)
    tuned_a = G.combine(P["bass"], P["stab_a"], accents_only=True)
    tuned_b = G.combine(P["bass_bounce"], P["stab_b"], accents_only=True)
    return [("kit", kit, G.syncopation(kit)),
            ("bass + stab, bar 1", tuned_a, G.syncopation(tuned_a)),
            ("bass + stab, bar 2", tuned_b, G.syncopation(tuned_b))]


if __name__ == "__main__":
    print(describe())
    print()
    print("Syncopation (Longuet-Higgins & Lee):")
    for name, surf, idx in groove_report():
        print(f"  {name:20s} {surf}  index {idx:3d}")
    print()
    print("Phrase fills:")
    for sec in SECTIONS:
        for lb in range(sec.length):
            k = fill_kind(sec, lb)
            if k:
                bar = sec.start + lb
                t = bar * 4 * 60 / BPM
                print(f"  {int(t)//60}:{int(t)%60:02d}  bar {bar:3d}  {k}")
