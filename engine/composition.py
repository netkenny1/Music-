"""
The composition: tempo, key, harmony, rhythm patterns and song structure.

Sonido Nocturno -- Latin tech, 130 BPM, D minor
------------------------------------------------
Modelled on the Cocoa-label minimal/deep-tech sound (the reference for this
edit is Joe Vanditti's "Sonido Latino": 130 BPM, D minor, 6:12, one groove
that never stops). The rules of that style, as applied here:

* **Tempo and key.** 130 BPM, the top of the tech-house pocket; D minor,
  Camelot 7A, so it mixes with the whole Latin-tech shelf (most of it sits
  in D, F, G and A minor). The bass root is D2 = 73 Hz -- lower than the
  previous edit's F2, which is what makes a deep-tech record *deep*.
* **The kit is tighter and drier.** A short minimal-tech kick, clap on 2
  and 4, closed 16th hats, the off-beat open hat, a shaker. No swing to
  speak of (10 % of a 16th, an MPC 55 %): Latin tech runs straight so the
  percussion can carry the shuffle.
* **The Latin layer.** Everything the genre tag is named for lives in the
  percussion and the bass, not in the harmony:
    - a **conga tumbao** (heel-toe ghosts, slap on 2, open tones on 4 and
      4-and, the low drum answering in the second bar);
    - a **2-3 son clave** on a wood block, the timeline every other part is
      phrased against;
    - a **cowbell** on the double-tresillo and a **cascara** on the rim in
      the drops, the timbalero's two hands;
    - **bongo martillo** in the last third;
    - a **tumbao bass**: rolling off-beat 8ths, then the fifth on beat 4
      and the anticipated root on 4-and -- the salsa bass cell, played by a
      sub-heavy synth.
* **Harmony is minimal.** i - iv - VI - V (Dm9, Gm9, Bbmaj7, A7). The V is
  the one thing this loop has that the previous F-minor loop did not: A7's
  C-sharp is a real leading tone, so the cycle *cadences* every 8 bars,
  which is the Latin feel against the modal drift of deep house. One dark
  organ stab riff, one pad; a **montuno piano** (octaves, syncopated, on
  the clave) and a **horn hook** are the only melodic events, and they are
  saved for the second drop.
* **Vocal.** Chopped, rhythmic, two syllables ("eh / oh") on the clave's
  3-side, the way a Latin-tech vocal is a percussion instrument.

The phrasing (32 intro / 32 groove / 16 break / 16 build / 64 drop / 32
outro, fills on every 8th and 16th bar, crash on every 32nd) is unchanged:
it is the form of the genre and the reason the record mixes.
"""

from dataclasses import dataclass, field

TITLE = "Sonido Nocturno"
SLUG = "sonido_nocturno"
BPM = 130.0
KEY_NAME = "D minor"
KEY_TAG = "Dm"          # ID3 TKEY; Camelot 7A
CAMELOT = "7A"
SWING = 0.10          # fraction of a 16th that odd steps are pushed late
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
    voicing: list      # MIDI notes for chords/pads (close voicing, ~E3-F4)
    top: int           # MIDI note the vocal chop sings over this chord


# i - iv - VI - V in D minor. Voice leading, low to high:
#   53 -> 53 -> 53 -> 52     (F3   F3   F3   E3)
#   57 -> 58 -> 57 -> 55     (A3   Bb3  A3   G3)
#   60 -> 62 -> 62 -> 61     (C4   D4   D4   C#4)
#   64 -> 65 -> 65 -> 64     (E4   F4   F4   E4)
# Every voice moves by a tone or less. The A7 is voiced without its root
# (the bass has it) so the C-sharp -- the leading tone -- is exposed on top
# of the stack, and it resolves up a semitone into Dm9's D in the bass.
PROGRESSION = [
    Chord("Dm9",    bass=38, voicing=[53, 57, 60, 64], top=74),   # D5
    Chord("Gm9",    bass=43, voicing=[53, 58, 62, 65], top=74),   # D5
    Chord("Bbmaj7", bass=46, voicing=[53, 57, 62, 65], top=74),   # D5
    Chord("A7",     bass=45, voicing=[52, 55, 61, 64], top=76),   # E5
]

CHORD_BARS = 2        # each chord lasts two bars, so the cycle is 8 bars
BASS_LOW, BASS_HIGH = 36, 47     # C2 .. B2: where a deep-tech bass lives


def chord_at(bar):
    """Which chord is sounding in a given bar."""
    return PROGRESSION[(bar // CHORD_BARS) % len(PROGRESSION)]


def fold(note):
    """Bring a bass note into the bass register by octaves."""
    while note > BASS_HIGH:
        note -= 12
    while note < BASS_LOW:
        note += 12
    return note


def fifth_of(chord):
    """The fifth of a chord, in the bass register: the tumbao's beat-4 note."""
    return fold(chord.bass + 7)


def approach_note(bar):
    """
    The bass note that leads into the next chord.

    The last off-beat before a chord change plays the *fifth of the chord
    that is coming*, kept inside the bass register. Fifth-to-root is the
    strongest pull in tonal music (it is what a dominant does). In a tumbao
    the anticipated note on 4-and is usually the coming chord's root; the
    fifth is the disco/house version of the same idea and it reads as the
    same anticipation.
    """
    return fifth_of(chord_at(bar + 1))


# --------------------------------------------------------------------------
# Rhythm patterns
# --------------------------------------------------------------------------
# 16 characters = one bar of 16th notes.
#   '.' rest      'x' normal      'X' accent      'o' ghost/soft
VELOCITY = {"x": 1.00, "X": 1.15, "o": 0.50, ".": 0.0}

P = {
    # Four-on-the-floor.
    "kick":        "X...x...x...x...",
    # Beat 4 removed: the fill at the end of a 16-bar phrase.
    "kick_drop4":  "X...x...x.......",
    # Doubled to 8ths for the last two bars of the build.
    "kick_8ths":   "x.x.x.x.x.x.x.x.",

    # Backbeat on 2 and 4.
    "clap":        "....X.......X...",

    # Closed hats: accent on the kick, a ghost, nothing on the off-8th (the
    # open hat lives there), another ghost.
    "hat":         "Xo.oXo.oXo.oXo.o",
    "hat_8ths":    "x...x...x...x...",

    # THE open hat, on the off-beat 8ths.
    "ohat":        "..x...x...x...x.",

    # Shaker on the 16ths, accents with the open hat.
    "shaker":      "o.xo.oxo.oxo.oxo",

    # --- Latin percussion --------------------------------------------------
    # 2-3 son clave. Bar A is the "2 side" (beats 2 and 3), bar B the "3
    # side" (1, 2-and, 4). Every Latin part below is written against it:
    # the conga's open tones land with the clave's 4, the bass anticipation
    # with its 2-and.
    "clave_a":     "....x...x.......",
    "clave_b":     "x.....x.....x...",

    # Conga tumbao. Heel-toe ghosts on 1 and 3 (the hand resting on the
    # head), a slap on 2, open tones on 4 and 4-and. In the second bar the
    # last open tone moves to the low drum (the "bombo"), which is what
    # makes a tumbao two bars long rather than one repeated.
    "conga_ghost": "o.o.....o.o.....",
    "conga_slap":  "....X...........",
    "conga_open_a": "............x.x.",     # high drum, bar A
    "conga_open_b": "............x...",     # high drum, bar B
    "conga_low_b":  "......x.......x.",     # low drum, bar B

    # Cowbell on the double tresillo (3+3+2, twice). The accent on 1 is the
    # bell's mouth; the rest is the neck.
    "cowbell":     "X..x..x.x..x..x.",

    # Cascara on the rim of the timbale, 2-3. Written in 8ths on the
    # 16th grid: bar A  x.x.xx.x  /  bar B  x.xx.x.x
    "cascara_a":   "x...x...x.x...x.",
    "cascara_b":   "x...x.x...x...x.",

    # Bongo martillo, simplified: the macho (high) on the beats, the
    # hembra (low) as the heel between them.
    "bongo":       "x.o.x.o.x.o.x.o.",

    # --- bass --------------------------------------------------------------
    # Rolling off-beat 8ths: intro and outro.
    "bass":        "..x...x...x...x.",
    # Tumbao. Off-beats on 1-and, 2-and, 3-and, then the fifth ON beat 4
    # and the anticipated note on 4-and. The two notes on 4 and 4-and are
    # the salsa bass cell; playing them under a four-on-the-floor kick is
    # the whole Latin-tech trick.
    "bass_tumbao":  "..x...x...x.x.x.",
    # Second half of every 8: an octave pop on the 16th after beat 3 as
    # well, so the line bounces before it turns around.
    "bass_tumbao2": "..x...x.x.x.x.x.",

    # The organ stab riff, two bars: off-beats, with one hit pulled early in
    # bar two (on the clave's 2-and) so every two bars there is a
    # syncopation that resolves on the next downbeat.
    "stab_a":      "..X...x...X...x.",
    "stab_b":      "..X...x.x...X...",

    # Montuno piano, two bars, on the clave. Bar A is the double tresillo;
    # bar B anticipates the barline (the hit on the last 16th holds into
    # the next downbeat, which is where every montuno gets its lean).
    "montuno_a":   "x..x..x.x..x..x.",
    "montuno_b":   "..x..x..x.x....x",

    # Horn hook: three stabs on the tresillo, once every four bars.
    "brass":       "..X..X..X.......",

    # Breakdown electric piano: the tresillo, "x..x..x.".
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
    The club edit: 192 bars, 5:54 at 130, in 8-, 16- and 32-bar blocks.

    Every part enters or leaves on a phrase boundary. A DJ counts in
    phrases and a crowd hears in phrases; an element arriving on bar 13
    reads as a mistake.
    """
    S = []
    b = 0

    def add(name, length, energy, **parts):
        nonlocal b
        S.append(Section(name, b, length, energy, parts))
        b += length

    # --- 32 bars: DJ intro. One element every 8 bars. ---------------------
    # 0-7 kick and 8th hats. 8-15 open hat, clap, clave: the timeline is
    # set before anything plays against it. 16-23 bass and shaker. 24-31
    # the conga tumbao and the stab under an opening filter.
    add("intro", 32, 0.45,
        kick=True, hat="hat_8ths", hat_from=0,
        hat16_from=8, ohat_from=8, clap_from=8, clave_from=8,
        bass_from=16, shaker_from=16, sub_layer=True,
        conga_from=24, stab_from=24, pad_from=24,
        filter_sweep=(2200, 20000), crash_at=[0, 16])

    # --- 32 bars: the groove. Everything in; cowbell and the vocal chops
    # from bar 16. Fills at 7, 15, 23, 31. --------------------------------
    add("main_a", 32, 0.85,
        kick=True, hat="hat", ohat=True, clap=True, shaker=True, clave=True,
        conga=True, bass="bass_tumbao", sub_layer=True, stab=True,
        pad_from=8, cowbell_from=16, vox_from=16,
        crash_at=[0, 16], bounce_second_half=True)

    # --- 16 bars: breakdown. Drums out; the clave and conga keep talking
    # over the pad and the electric piano, then the horn answers. ---------
    add("break", 16, 0.40,
        pad=True, keys_chords=True, vox=True, clave=True,
        conga_from=8, brass_from=8, downlifter=True,
        filter_sweep=(900, 16000))

    # --- 16 bars: build. Kick back at 0. Bass leaves at 8. Snare roll over
    # the last 4, riser over the last 8, kick to 8ths for the last 2, the
    # last beat cut. -------------------------------------------------------
    add("build", 16, 0.75,
        kick=True, hat="hat", hat_from=4, ohat_from=8, clap=True,
        shaker=True, clave=True, conga=True, bass_until=8, sub_layer=True,
        stab=True, pad=True, snare_roll=True, riser=True, kick_8ths_from=14,
        filter_sweep=(6000, 20000), silence_from=(15, 12),
        reverse_crash_at=[15])

    # --- 32 bars: THE DROP. Full groove, tumbao bass, cowbell, cascara,
    # the horn hook every four bars, sub drop on the downbeat. -------------
    add("drop", 32, 1.00,
        kick=True, hat="hat", ohat=True, clap=True, shaker=True, clave=True,
        conga=True, cowbell=True, cascara=True, brass=True,
        bass="bass_tumbao", sub_layer=True, stab=True, pad=True, vox=True,
        crash_at=[0, 16], sub_drop=True, bounce_second_half=True)

    # --- 32 bars: second drop. The montuno piano is the new element
    # (0-15), bongos join. 16-23 thin out (kick, bass, clave, pad), 24-31
    # everything back for the last run. -----------------------------------
    add("drop_b", 32, 1.00,
        kick=True, hat="hat", ohat=True, clap=True, shaker=True, clave=True,
        conga=True, cowbell=True, cascara=True, bongo=True, brass=True,
        bass="bass_tumbao", sub_layer=True, stab=True, pad=True, vox=True,
        montuno=True, montuno_until=16,
        thin_bars=range(16, 24),
        crash_at=[0, 16, 24], bounce_second_half=True)

    # --- 32 bars: DJ outro. Elements leave every 8, filter closes. --------
    add("outro", 32, 0.60,
        kick=True, hat="hat", ohat=True, ohat_until=24, clap=True,
        clap_until=24, shaker=True, shaker_until=16, clave=True,
        clave_until=24, conga=True, conga_until=16, bass=True,
        bass_until=16, sub_layer=True, stab=True, stab_until=8,
        pad=True, pad_until=8, filter_sweep=(20000, 1400), crash_at=[0])

    return S


SECTIONS = build_sections()
TOTAL_BARS = SECTIONS[-1].end

# Sections where the groove is running and the 8/16/32-bar fills apply.
GROOVE_SECTIONS = {"intro", "main_a", "drop", "drop_b", "outro"}

# Parts that stay in during the thinned-out bars of the second drop: the
# kick, the bass, the clave and the pad. Everything else drops away.
THIN_KEEP = {"kick", "bass", "sub", "clave", "pad"}


def fill_kind(sec, local_bar):
    """
    Which fill, if any, ends this bar.

    Bar 7 of every 8 gets the small clap roll; bar 15 of every 16 the bigger
    roll with the kick's fourth beat removed; the last bar of a section
    additionally gets the reverse cymbal into the next downbeat.
    """
    if sec.name not in GROOVE_SECTIONS:
        return None
    if (local_bar + 1) % 16 == 0:
        return "fill_16"
    if (local_bar + 1) % 8 == 0:
        return "fill_8"
    return None


# --------------------------------------------------------------------------
# The vocal chops
# --------------------------------------------------------------------------
# (bar within the 4, step, vowel, 16ths). Two syllables on the clave's
# 3-side in bars 0 and 2 -- "eh" on 2-and, "oh" on 3 -- and one longer "ah"
# on the 4 of bar 3. It is a chopped sample, not a melody: a Latin-tech
# vocal is percussion that happens to have a pitch.
VOX_HOOK = [
    (0, 6, "eh", 2),
    (0, 8, "oh", 3),
    (2, 6, "eh", 2),
    (2, 8, "oh", 3),
    (3, 12, "ah", 4),
]


def describe():
    """Human-readable summary of the arrangement."""
    lines = [
        f"Title    : {TITLE}",
        f"Key      : {KEY_NAME}  (Camelot {CAMELOT})",
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

    The full kit measures near zero -- correct for four-on-the-floor. The
    Latin layer is where the syncopation lives: the clave, the tumbao's
    open tones and the bass anticipation all hold through strong beats.
    """
    import groove as G
    kit = G.combine(P["kick"], P["clap"], P["hat"], P["ohat"], accents_only=True)
    latin_a = G.combine(P["clave_a"], P["conga_slap"], P["conga_open_a"],
                        P["cowbell"], accents_only=True)
    latin_b = G.combine(P["clave_b"], P["conga_slap"], P["conga_open_b"],
                        P["conga_low_b"], P["cowbell"], accents_only=True)
    tuned_a = G.combine(P["bass_tumbao"], P["stab_a"], accents_only=True)
    tuned_b = G.combine(P["bass_tumbao2"], P["stab_b"], accents_only=True)
    return [("kit", kit, G.syncopation(kit)),
            ("latin perc, bar A", latin_a, G.syncopation(latin_a)),
            ("latin perc, bar B", latin_b, G.syncopation(latin_b)),
            ("bass + stab, bar A", tuned_a, G.syncopation(tuned_a)),
            ("bass + stab, bar B", tuned_b, G.syncopation(tuned_b))]


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
