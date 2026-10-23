# core/notation/jianpu/measure.py — Measure-level operations for Jianpu conversion.
from typing import Optional

from music21 import chord as m21chord, note as m21note

from ...config import JianpuNote
from .primitives import (
    _ANACRUSIS_CODE_TO_UNITS,
    _QL_TO_ANACRUSIS_CODE,
    clone_jianpu_note,
    jianpu_note_units,
    normalize_jianpu_duration,
    split_duration_chunks,
)


def _detect_trailing_rest_pickup(measure, nominal_length: float) -> Optional[float]:
    """Fallback pickup (弱起) detection for measures where MusicXML has ``implicit="no"``.

    Some OMR engines (e.g. Homr) encode a pickup bar as a *full* measure with
    the actual pickup notes at the start followed by a filler rest — rather than
    using ``implicit="yes"`` (which music21 translates to ``paddingLeft``).

    Detection criteria:
    - The last element in the measure is a rest (trailing-rest pattern).
    - The notes before that rest occupy ≤ half the nominal bar length
      (conservative threshold to avoid false positives on normal first bars
      that simply end with a rest).
    - The note portion matches a recognised anacrusis code to within 0.25 QL.

    Returns the pickup QL duration if detected, else ``None``.
    """
    try:
        elements = sorted(measure.flatten().notesAndRests, key=lambda e: float(e.offset))
    except Exception:
        return None
    if not elements:
        return None
    # Must end with a rest (trailing-rest pattern)
    if not isinstance(elements[-1], m21note.Rest):
        return None
    # Find effective end = offset + duration of last non-rest element
    last_note_end = 0.0
    for e in elements:
        if not isinstance(e, m21note.Rest):
            last_note_end = float(e.offset) + float(e.duration.quarterLength or 0.25)
    if last_note_end < 0.01:
        return None  # all rests — not a pickup
    # Pickup must be ≤ half bar (conservative; avoids treating normal bars as pickups)
    if last_note_end > nominal_length * 0.5 + 0.01:
        return None
    # Match to recognised anacrusis code
    _closest = min(_QL_TO_ANACRUSIS_CODE, key=lambda q: abs(q - last_note_end))
    if abs(_closest - last_note_end) < 0.25:
        return _closest
    return None


def _parse_bar_units(time_signature: str) -> int:
    """Convert a 'num/denom[,pickup]' time signature string to jianpu-ly barLength (64th-note units)."""
    try:
        ts = time_signature.split(',')[0]  # strip optional anacrusis suffix, e.g. "4/4,8" → "4/4"
        num_s, denom_s = ts.split('/')
        return int(64 * int(num_s) / int(denom_s))
    except (ValueError, ZeroDivisionError, AttributeError):
        return 64  # fallback: 4/4


def _parse_anacrusis_units(time_signature: str) -> Optional[int]:
    """Return pickup bar length in 64th-note units if the time signature has an anacrusis suffix (e.g. '4/4,8')."""
    if ',' not in time_signature:
        return None
    code = time_signature.split(',', 1)[1].strip()
    return _ANACRUSIS_CODE_TO_UNITS.get(code)


def _ratio_to_bar_units(ratio: str) -> Optional[int]:
    """'3/4' → 48.  Same arithmetic as jianpu-ly's ``barLength = int(64*num/denom)``."""
    try:
        num_s, denom_s = str(ratio).split('/')
        return int(64 * int(num_s) / int(denom_s))
    except (ValueError, ZeroDivisionError, AttributeError):
        return None


def measure_bar_units(measure) -> Optional[int]:
    """Bar length of a single music21 Measure, in 64th-note units.

    Meter changes mid-score (e.g. 4/4 → 3/4 → 4/4) are stored as a
    ``TimeSignature`` inside the measure where they take effect; music21 also
    folds them into ``measure.barDuration``.  We read both because OMR output
    sometimes carries the signature *without* a matching ``barDuration``.

    Returns ``None`` when the measure carries no meter information at all, so
    the caller can fall back to the piece's nominal signature.
    """
    try:
        signatures = measure.getTimeSignatures(recurse=False)
    except Exception:
        signatures = []
    if signatures:
        units = _ratio_to_bar_units(signatures[0].ratioString)
        if units:
            return units
    try:
        quarters = float(getattr(measure.barDuration, 'quarterLength', 0.0) or 0.0)
    except Exception:
        quarters = 0.0
    if quarters > 0.0:
        return max(2, int(round(quarters * 16)))
    return None


def measure_bar_signature(measure) -> Optional[str]:
    """The measure's time signature as printed (``'3/4'``), or ``None`` if absent."""
    try:
        signatures = measure.getTimeSignatures(recurse=False)
    except Exception:
        return None
    if signatures:
        return signatures[0].ratioString
    return None


class MeasureBarUnits(list):
    """The extracted measure list, plus each measure's own bar length.

    ``bar_units`` is the length of the measure **as the model wrote it** — the sum
    of the note tokens that will be emitted, in 64th-note units — not the length
    implied by a time signature.  The jianpu-ly text builder derives every
    measure's ``\\time`` from the emitted notes themselves (see
    :func:`measure_units`), so these values are a cross-check/diagnostic rather
    than an instruction: a measure is never padded or trimmed to match them.

    This subclasses ``list`` on purpose — every existing caller only iterates or
    slices the result, so keeping the plain-list surface means no call site has
    to change.
    """

    def __init__(self, measures=()):
        super().__init__(measures)
        # Parallel to the measures: content length in 64th-note units, and the
        # time signature the OMR wrote on that measure (``None`` if it merely
        # inherits the previous one).  The signature is kept for diagnostics
        # only — it is never used to lay the measure out.
        self.bar_units: list[int] = []
        self.time_signatures: list[Optional[str]] = []


def measure_units(notes: list[JianpuNote]) -> int:
    """Length of *notes* in 64th-note units, exactly as jianpu-ly will count it.

    This is the single source of truth for a bar's length: whatever the OMR
    recognised for the measure is what the measure is.  Summing the *emitted*
    tokens (rather than music21 offsets) guarantees that the ``\\time`` we write
    equals the ``barPos`` jianpu-ly accumulates, so no note can cross a barline.
    """
    return sum(jianpu_note_units(note) for note in notes)


def layout_time_word(units: int) -> Optional[str]:
    """A ``\\time`` word whose jianpu-ly barLength is exactly *units*.

    jianpu-ly aborts ("barcheck fail") the moment the notes cross its internal
    ``barLength``, so the bar length has to follow the notes instead of the notes
    being bent to a meter.  The word is a *layout* device only: no guessed meter
    is ever printed (``jianpu_staff_start`` suppresses the TimeSignature glyph),
    so the denominator is chosen for exactness and for quarter-note beaming —
    ``beatLength`` becomes 24 when ``num % 3 == 0`` with ``denom > 4``, which
    would break the one-dash-per-quarter rests, hence the /4 and /8 preferences.
    """
    if units <= 0:
        return None
    if units % 16 == 0:
        return f'{units // 16}/4'
    if units % 8 == 0:
        return f'{units // 8}/8'
    return f'{units}/64'


def repair_jianpu_measure(measure_notes: list[JianpuNote], measure_length: float) -> list[JianpuNote]:
    """Trim overflowing notes and pad with rests so the measure total equals the time-signature length."""
    tol = 0.01
    current_total = 0.0
    repaired: list[JianpuNote] = []
    rest_template = JianpuNote('0', '', 0, 0, 1.0, 0, None, True)

    for note in measure_notes:
        if current_total >= measure_length - tol:
            break

        remaining = measure_length - current_total
        if remaining <= tol:
            break

        note_duration = max(float(note.duration), 0.125)
        if note_duration <= remaining + tol:
            new_note = clone_jianpu_note(note, note_duration)
            repaired.append(new_note)
            current_total += new_note.duration
            continue

        for piece_idx, piece in enumerate(split_duration_chunks(remaining)):
            if current_total >= measure_length - tol:
                break
            # Only the first fragment keeps the syllable — see clone_jianpu_note's
            # carry_lyrics docstring.
            new_note = clone_jianpu_note(note, piece, carry_lyrics=(piece_idx == 0))
            repaired.append(new_note)
            current_total += new_note.duration

    remaining = measure_length - current_total
    if remaining > tol:
        for piece in split_duration_chunks(remaining):
            new_rest = clone_jianpu_note(rest_template, piece)
            repaired.append(new_rest)
            current_total += new_rest.duration

    return repaired


def clone_monophonic_element(element, duration: float):
    """Clone a music21 element, normalising its duration for the jianpu grid.

    Propagates ``.lyrics`` onto the new element — constructing a fresh Note from
    just a pitch otherwise silently drops any attached lyric, which is how the
    lyrics-support work first surfaced this (see docs/修复计划2与简谱编辑器规划.md's
    B10 appendix): note_to_jianpu() reads ``element.lyrics``, but every element
    on the extraction hot path passes through here first, so without this the
    lyrics field would always come out empty regardless of what note_to_jianpu
    itself does.

    A music21 ``Chord`` is preserved as a chord (not collapsed to a single
    pitch): ``note_to_jianpu`` then renders its simultaneous pitches as one
    jianpu chord (合音).  Collapsing to the top pitch here was why chords such
    as ``(#4 1)`` used to lose every note but the highest.
    """
    normalized_duration = normalize_jianpu_duration(duration)
    if isinstance(element, m21note.Rest):
        new_element = m21note.Rest()
    elif isinstance(element, m21chord.Chord):
        # Any Chord — including a one-pitch one, which music21 produces for a
        # single stem with a chord flag — must stay a Chord: falling through to
        # the Note branch raised "'Chord' object has no attribute 'pitch'"
        # (中音笙).  A one-pitch Chord renders identically to a Note.
        pitches = list(element.pitches)
        new_element = m21chord.Chord(pitches) if pitches else m21note.Rest()
        new_element.lyrics = list(element.lyrics)
    else:
        new_element = m21note.Note(element.pitch)
        new_element.lyrics = list(element.lyrics)
    new_element.duration.quarterLength = normalized_duration
    return new_element
