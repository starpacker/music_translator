"""
jianpu_transpose.py — renumber jianpu output for an arbitrary tonic (1=X).

The pipeline detects *absolute* pitches: a note's staff position gives its
diatonic step, the key signature / accidentals give the chromatic
alteration.  ``jianpu_formatter`` emits those absolute pitches written in
the "1 = C" convention (accidentals as ``#``/``b`` prefixes).

Jianpu, however, is a *movable-do* system: a piece in G major is far more
readable as ``1=G`` (no accidental on every fa) than as ``1=C`` with a
``#4`` sprinkled everywhere.  This module performs the exact renumbering

    absolute midi  →  degree relative to a chosen tonic

on an already-formatted jianpu string, so it can be applied as the very
last formatting step without touching the detection pipeline.

Spelling convention for chromatic degrees (common Chinese practice):
    rem  1 → #1      rem  3 → b3      rem  6 → #4
    rem  8 → b6      rem 10 → b7
"""
import re

# semitone offset of each jianpu degree within a major scale of the tonic
_DEGREE_SEMIS = {1: 0, 2: 2, 3: 4, 4: 5, 5: 7, 6: 9, 7: 11}
_SEMIS_TO_DEGREE = {v: k for k, v in _DEGREE_SEMIS.items()}
# chromatic remainders → preferred spelling (accidental, degree)
_CHROMATIC = {1: ('#', 1), 3: ('b', 3), 6: ('#', 4), 8: ('b', 6), 10: ('b', 7)}

# tonic name → semitones above C
_TONIC_SEMIS = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}

# number of sharps/flats in the key signature → major tonic ("auto" mode)
_SIG_TO_TONIC = {
    ('#', 0): 'C', ('#', 1): 'G', ('#', 2): 'D', ('#', 3): 'A',
    ('#', 4): 'E', ('#', 5): 'B', ('#', 6): 'F#', ('#', 7): 'C#',
    ('b', 0): 'C', ('b', 1): 'F', ('b', 2): 'Bb', ('b', 3): 'Eb',
    ('b', 4): 'Ab', ('b', 5): 'Db', ('b', 6): 'Gb', ('b', 7): 'Cb',
}

# module-level target tonic in semitones above C; 0 == "1=C" (no-op)
TARGET_TONIC_SEMIS = 0


def parse_tonic(name):
    """Parse a tonic name such as ``C``, ``G``, ``bB``/``Bb`` → semitones.

    Accepts an optional leading or trailing ``#``/``b`` accidental.
    Returns an int in 0..11, or ``None`` for ``auto``/empty (the caller
    then resolves the tonic from the detected key signature).
    Raises ValueError on garbage.
    """
    if name is None:
        return 0
    s = str(name).strip()
    if s.lower() in ('auto', ''):
        return None
    acc = 0
    body = s
    if body[:1] in ('#', 'b'):
        acc = 1 if body[0] == '#' else -1
        body = body[1:]
    elif body[-1:] in ('#', 'b') and len(body) > 1:
        acc = 1 if body[-1] == '#' else -1
        body = body[:-1]
    body = body.strip()
    if len(body) != 1 or body.upper() not in _TONIC_SEMIS:
        raise ValueError(f"unrecognised tonic: {name!r}")
    return (_TONIC_SEMIS[body.upper()] + acc) % 12


def tonic_from_key_sig(key_sig):
    """Major tonic (semitones above C) implied by a key-signature dict."""
    if not key_sig:
        return 0
    return parse_tonic(_SIG_TO_TONIC.get(
        (key_sig.get('type'), int(key_sig.get('count', 0))), 'C'))


def set_target_tonic(semitones):
    """Set the global display tonic (0 = 1=C).  Returns the value set."""
    global TARGET_TONIC_SEMIS
    TARGET_TONIC_SEMIS = int(semitones) % 12
    return TARGET_TONIC_SEMIS


def midi_of_token(acc, degree, octave):
    """Absolute midi for accidental+degree+octave in the 1=C convention."""
    return 60 + 12 * octave + _DEGREE_SEMIS[degree] + acc


def token_from_midi(midi, tonic_semis):
    """Inverse of :func:`midi_of_token` for an arbitrary tonic."""
    r = midi - 60 - tonic_semis
    octv = r // 12
    rem = r - 12 * octv
    if rem in _SEMIS_TO_DEGREE:
        acc, deg = '', _SEMIS_TO_DEGREE[rem]
    else:
        acc, deg = _CHROMATIC[rem]
    return acc, deg, octv


_NOTE_RE = re.compile(r'^(?P<acc>[#b]?)(?P<deg>[1-7])(?P<oct>[\',]*)$')

# A note token inside a formatted measure string.  The lookbehind stops us
# from eating digits of duration suffixes (/2, /12, /3*2), rest dashes or
# the "×N" multi-rest marker; the lookahead stops at a duration suffix.
_TOKEN_RE = re.compile(
    r'(?<![0-9a-zA-Z*.,/\-×])[#b]?[1-7][\',]*(?=[\s\]/\-.|]|$)')


def transpose_note_token(token, tonic_semis=None):
    """Renumber one bare note token (e.g. ``#4'``) into the target tonic."""
    if tonic_semis is None:
        tonic_semis = TARGET_TONIC_SEMIS
    m = _NOTE_RE.match(token)
    if not m:
        return token
    acc = {'': 0, '#': 1, 'b': -1}[m.group('acc')]
    deg = int(m.group('deg'))
    octs = m.group('oct')
    octv = octs.count("'") - octs.count(',')
    a2, d2, o2 = token_from_midi(midi_of_token(acc, deg, octv), tonic_semis)
    return a2 + str(d2) + ("'" * o2 if o2 > 0 else ',' * -o2)


def transpose_string(s, tonic_semis=None):
    """Transpose a whole jianpu string (a measure, a line or a full text).

    Rests (``0``), duration suffixes, chord brackets, bar lines and the
    ``×N`` multi-rest marker are preserved verbatim; only note tokens are
    renumbered.
    """
    if tonic_semis is None:
        tonic_semis = TARGET_TONIC_SEMIS
    if tonic_semis == 0:
        return s
    return _TOKEN_RE.sub(lambda m: transpose_note_token(m.group(0), tonic_semis), s)
