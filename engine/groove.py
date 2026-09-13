"""
Groove: syncopation measurement, micro-timing, and expectation violation.

The idea
--------
Pleasure in rhythm comes from *prediction error*. The brain builds a model of
where the next beat falls; reward fires when that model is violated and then
resolved. But Witek et al. (PLOS ONE, 2014) measured syncopation against both
pleasure and desire-to-move and found an **inverted U**: medium syncopation
scores highest, and heavy syncopation scores *worse than none at all*.

That gives a hard design constraint. A violation is only rewarding against an
expectation solid enough to be violated -- so the kick stays nailed to the
grid and the four-on-the-floor pulse is never in doubt, while everything
around it bends. More stutters is not better; this module exists so the amount
can be measured rather than guessed.

Longuet-Higgins & Lee (1984) syncopation index
----------------------------------------------
Every position in a bar has a metrical weight: the downbeat is strongest,
then the half-bar, then quarters, then eighths, then sixteenths. A
syncopation occurs when a note starts on a weak position and *holds through*
a stronger one -- the strong beat arrives and nothing articulates it. The
score is the difference in weight, summed over the bar.
"""

# Metrical hierarchy for one bar of 16th notes in 4/4.
# 0 is strongest (the downbeat); -4 is weakest (an offbeat 16th).
METRIC_WEIGHTS = [0, -4, -3, -4, -2, -4, -3, -4, -1, -4, -3, -4, -2, -4, -3, -4]


def syncopation(pattern, weights=METRIC_WEIGHTS):
    """
    Longuet-Higgins & Lee syncopation index for a 16-step pattern string.

    For each onset, look ahead to the next onset. If any position it holds
    through is metrically stronger than the position it started on, that is a
    syncopation worth the difference in weight. Wraps around the bar, because
    a note on the last 16th holding over the next downbeat is the strongest
    syncopation available.

    Four-on-the-floor scores 0. Offbeat eighths score 7.
    """
    n = len(pattern)
    onsets = [i for i, c in enumerate(pattern) if c != "."]
    if len(onsets) < 1:
        return 0

    total = 0
    for k, i in enumerate(onsets):
        nxt = onsets[(k + 1) % len(onsets)]
        # positions the note holds through, wrapping past the barline
        held, j = [], (i + 1) % n
        while j != nxt and len(held) < n:
            held.append(j)
            j = (j + 1) % n
        if held:
            strongest = max(weights[h] for h in held)
            if strongest > weights[i]:
                total += strongest - weights[i]
    return total


def combine(*patterns, accents_only=False):
    """
    Merge patterns into the single rhythmic surface a listener actually hears.

    Syncopation must be measured on the combined kit, not per instrument. A
    clap on beat 4 looks syncopated alone -- it holds over the next downbeat --
    but in context the kick articulates that downbeat, so nothing is violated.
    Measuring voices in isolation systematically over-reports.

    With `accents_only`, ghost notes ("o") count as rests. This is usually the
    measurement you want. A continuous 16th hi-hat articulates every position,
    which drives the raw index to zero -- not because the groove is square but
    because the surface is saturated. What a listener tracks is the pattern of
    *emphasis*, so the accent surface is the one that carries the meter.
    """
    ignore = ".o" if accents_only else "."
    n = max(len(p) for p in patterns)
    out = []
    for i in range(n):
        hit = any(p[i % len(p)] not in ignore for p in patterns)
        out.append("x" if hit else ".")
    return "".join(out)


def describe_patterns(patterns):
    """Report the syncopation index of each named pattern."""
    rows = []
    for name, pat in sorted(patterns.items()):
        rows.append((name, pat, syncopation(pat)))
    return rows


# --------------------------------------------------------------------------
# Micro-timing
# --------------------------------------------------------------------------
# Per-instrument timing offsets in milliseconds. Negative = ahead of the grid
# ("pushing", urgent), positive = behind it ("laid back", relaxed).
#
# This is how human rhythm sections actually play, and it is a separate axis
# from swing: swing displaces every odd 16th by a fixed fraction, whereas this
# shifts a whole instrument's relationship to the pulse. A clap 9 ms late is
# the difference between a drum machine and a groove.
#
# The kick is pinned at 0.0 and must stay there. It is the reference everything
# else is heard against; move it and there is no grid left to play against.
MICRO_TIMING = {
    "kick":   0.0,     # the anchor -- never moves
    "sub":    0.0,
    "bass":  -4.0,     # slightly ahead: drives the groove forward
    "clap":  +9.0,     # behind the beat: the classic lazy house backbeat
    "hat":   +2.0,
    "ohat":  +5.0,
    "shaker": +7.0,
    "rim":   +4.0,
    "stab":  +3.0,
    "keys":  +6.0,
    "arp":   +1.0,
    "vox":   +8.0,
    "melody": +5.0,
    "counter": +7.0,   # answers the hook, so it sits a touch further back
}


def micro_offset(part, sr):
    """Micro-timing offset for a part, in samples."""
    return int(MICRO_TIMING.get(part, 0.0) * sr / 1000.0)


# --------------------------------------------------------------------------
# Polyrhythm
# --------------------------------------------------------------------------

def polyrhythm_steps(period, bars, steps_per_bar=16, offset=0):
    """
    Onsets every `period` 16ths, running continuously across barlines.

    With period 3 against a 16-step bar the pattern will not realign with the
    downbeat until bar 3 (lcm(3,16) = 48 steps). So the layer slowly rotates
    against the pulse and locks back in every three bars -- a long, slow
    tension-and-release that the listener feels without being able to name.

    Returns (bar, step) pairs.
    """
    out = []
    total = bars * steps_per_bar
    for s in range(offset, total, period):
        out.append((s // steps_per_bar, s % steps_per_bar))
    return out


# --------------------------------------------------------------------------
# Stutter / beat repeat
# --------------------------------------------------------------------------

def stutter_schedule(total_steps, accelerate=True):
    """
    Slice lengths for a beat-repeat fill, in 16th notes.

    Accelerating from 16ths to 32nds to 64ths is the shape that works: each
    repeat arrives sooner than predicted, so prediction error accumulates
    instead of resetting, and the downbeat that finally lands resolves all of
    it at once.
    """
    if not accelerate:
        return [0.5] * int(total_steps * 2)

    schedule, remaining, size = [], float(total_steps), 1.0
    while remaining > 0.01:
        step = min(size, remaining)
        schedule.append(step)
        remaining -= step
        # halve the slice length every two repeats
        if len(schedule) % 2 == 0:
            size = max(size / 2.0, 0.125)
    return schedule
