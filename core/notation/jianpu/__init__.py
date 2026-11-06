# core/notation/jianpu/__init__.py — Public API for Jianpu conversion.
# Orchestration functions live here; sub-modules provide the building blocks.
import logging as _logging
import re
from typing import Literal, Optional, overload

from ...utils import log_message
from .primitives import (
    _get_score_key_tonic,
    build_lyric_lines,
    clone_jianpu_note,
    duration_suffix,
    format_jianpu_note_text,
    get_duration_render,
    infer_duration_dots,
    jianpu_note_token,
    jianpu_octave_symbol,
    normalize_jianpu_duration,
    note_to_jianpu,
    split_duration_chunks,
    MAX_SANE_BARS,
    _DIATONIC_SEMITONES,
    _CHROMATIC_SHARP,
    _CHROMATIC_FLAT,
    _FLAT_KEY_SEMITONES,
    _QL_TO_ANACRUSIS_CODE,
)
from .measure import (
    _parse_bar_units,
    layout_time_word,
    measure_units,
)
from .extract import (
    _extract_part_repeat_barlines,
    extract_jianpu_measures,
    extract_strict_jianpu_measures,
    strip_grace_notes,
)
from .primitives import fixed_reference

from ...config import JianpuNote  # re-export for callers that import it from here


def choose_measures_per_line(measures: list[list[JianpuNote]]) -> int:
    """Choose measures per line based on average note density (dense → 3, otherwise 4)."""
    if not measures:
        return 4
    sample = measures[: min(len(measures), 24)]
    avg_notes = sum(len(measure) for measure in sample) / max(len(sample), 1)
    if avg_notes >= 9:
        return 3
    return 4


def _header_time_signature(time_sig: str) -> str:
    """The signature word written in the jianpu-ly header.

    Only the bar length of the *first* measure is set by it (any later measure
    whose notes add up to something else gets its own layout ``\\time``), and the
    anacrusis suffix must go: it makes jianpu-ly pre-advance ``barPos`` by
    ``\\partial``, which would offset every barline in the piece.  The pickup bar
    now simply lasts as long as its notes do.
    """
    word = (time_sig or '').split(',')[0].strip()
    return word or '4/4'


def layout_time_tokens(measures, header_units: int) -> dict[int, str]:
    """Map measure index → the ``\\time`` word that must precede it.

    jianpu-ly aborts ("barcheck fail") the moment its running ``barPos`` crosses
    ``barLength``, so every measure has to be introduced with a bar length that
    matches the notes we actually emit for it.  This is *not* a meter: the word
    produced by :func:`layout_time_word` is derived from the measure's own note
    total, it is only listed where that total differs from the previous measure's,
    and the TimeSignature glyph is suppressed in the staff (see
    ``jianpu_staff_start``) so nothing is printed.  That is what makes the notes
    authoritative instead of the meter: nothing is padded or trimmed to fit a
    signature, and no guessed signature can appear in the score.
    """
    if not measures:
        return {}
    tokens: dict[int, str] = {}
    previous = header_units
    for index, measure in enumerate(measures):
        units = measure_units(measure)
        if units <= 0:
            # Nothing to lay out (an empty measure) — leave the bar length alone
            # rather than emitting a degenerate \time 0/… word.
            continue
        if units != previous:
            word = layout_time_word(units)
            if word:
                tokens[index] = word
                previous = units
    return tokens


def _with_time_signatures(measure_texts: list[str], start: int,
                          time_sig_tokens: dict[int, str]) -> list[str]:
    """Prepend the ``\\time`` word to the measures that need one."""
    if not time_sig_tokens:
        return measure_texts
    return [
        f'{time_sig_tokens[start + offset]} {text}' if (start + offset) in time_sig_tokens
        else text
        for offset, text in enumerate(measure_texts)
    ]


def _measure_direction_events(score, part, measure_count: int, event_type, description: str):
    """Find directions in measures, parts or the score without moving them."""
    from music21 import stream
    if part is None or not measure_count:
        return []
    source_measures = list(part.getElementsByClass(stream.Measure))[:measure_count]
    has_measures = bool(source_measures)
    if not source_measures:
        source_measures = [part]
    starts = [float(m.getOffsetInHierarchy(part)) if m is not part else 0.0
              for m in source_measures]
    candidates = [(index, float(mark.getOffsetInHierarchy(measure)), mark)
                  for index, measure in enumerate(source_measures)
                  for mark in measure.recurse().getElementsByClass(event_type)]
    # Marks may also belong directly to a Part or Score instead of a Measure.
    direct_sources = [(part, 0.0)] if has_measures else []
    if score is not part:
        direct_sources.append((score, float(part.getOffsetInHierarchy(score))))
    for container, part_offset in direct_sources:
        for mark in container.getElementsByClass(event_type):
            absolute = float(mark.offset) - part_offset
            indices = [i for i, start in enumerate(starts) if start <= absolute + 1e-7]
            if not indices:
                log_message(f'[jianpu] 谱表之前的{description}无法定位，需人工校对。', _logging.WARNING)
                continue
            index = indices[-1]
            candidates.append((index, absolute - starts[index], mark))
    return candidates


def _measure_tempo_marks(score, part, measure_count: int) -> dict[int, list[tuple[float, str]]]:
    """Collect representable metronome marks at their original measure offsets."""
    from music21 import tempo as m21tempo
    result: dict[int, list[tuple[float, str]]] = {}
    units = {'whole': 1, 'half': 2, 'quarter': 4, 'eighth': 8,
             '16th': 16, '32nd': 32, '64th': 64}
    for index, offset, mark in _measure_direction_events(
            score, part, measure_count, m21tempo.MetronomeMark, '速度记号'):
        referent = mark.referent
        unit = units.get(getattr(referent, 'type', None))
        number = mark.number
        if (unit is None or number is None or number <= 0
                or abs(float(number) - round(float(number))) > 1e-7):
            log_message(f'[jianpu] 小节 {index + 1} 的速度记号无法表示，需人工校对。', _logging.WARNING)
            continue
        dots = int(getattr(referent, 'dots', 0) or 0)
        result.setdefault(index, []).append((offset, f'{unit}{"." * dots}={round(float(number))}'))
    return result


def _measure_dynamic_marks(score, part, measure_count: int) -> dict[int, list[tuple[float, str]]]:
    """Collect supported dynamic marks without confusing them with text commands."""
    from music21 import dynamics
    allowed = {'ppppp', 'pppp', 'ppp', 'pp', 'p', 'mp', 'mf', 'f', 'ff', 'fff',
               'ffff', 'fffff', 'fp', 'sf', 'sff', 'sfz', 'sfp', 'rfz', 'sp', 'spp', 'fz'}
    result: dict[int, list[tuple[float, str]]] = {}
    for index, offset, mark in _measure_direction_events(
            score, part, measure_count, dynamics.Dynamic, '力度记号'):
        value = str(mark.value or '').strip().lower()
        if value not in allowed:
            log_message(f'[jianpu] 小节 {index + 1} 的力度记号 {value!r} 无法表示，需人工校对。',
                        _logging.WARNING)
            continue
        result.setdefault(index, []).append((offset, '\\' + value))
    return result


def _measure_meter_marks(score, part, measure_count: int) -> dict[int, list[tuple[float, str]]]:
    """Collect actual written meters, distinct from layout-only bar lengths."""
    from music21 import meter
    result: dict[int, list[tuple[float, str]]] = {}
    previous = None
    candidates = _measure_direction_events(score, part, measure_count, meter.TimeSignature, '拍号')
    for index, offset, mark in sorted(candidates, key=lambda item: (item[0], item[1])):
        ratio = str(mark.ratioString or '')
        if len(ratio) > 64 or not re.fullmatch(r'[1-9][0-9]*/[1-9][0-9]*(?:\+[1-9][0-9]*/[1-9][0-9]*)*', ratio):
            log_message(f'[jianpu] 小节 {index + 1} 的拍号 {ratio!r} 无法表示，请校对。', _logging.WARNING)
            continue
        if ratio != previous:
            # A little leading space separates the mark from the key label at
            # the start of the staff.  It has no effect on bar timing.
            result.setdefault(index, []).append((offset, '^\\markup{\\concat{\\hspace#3"' + ratio + '"}}'))
            previous = ratio
    return result


def _wedge_marks(measures, pairs):
    """Return paired wedge commands only if both source boundaries fit."""
    from collections import defaultdict
    available = []
    for measure in measures:
        position = 0.0
        boundaries = [0.0]
        for event in measure:
            if not event.is_grace:
                position += event.duration
            boundaries.append(position)
        available.append(boundaries)
    marks = defaultdict(list)
    for start, end, kind in pairs:
        if any(index < 0 or index >= len(available)
               or not any(abs(float(offset) - boundary) < 1e-6 for boundary in available[index])
               for index, offset in (start, end)):
            log_message(f'[jianpu] 小节 {start[0] + 1} 的渐强渐弱线端点不在简谱事件边界，'
                        '整条线未添加，请校对。', _logging.WARNING)
            continue
        command = 'WedgeCrescendo' if kind == 'crescendo' else 'WedgeDiminuendo'
        marks[start[0]].append((float(start[1]), command))
        marks[end[0]].append((float(end[1]), 'WedgeStop'))
    # A stop at the same time as the next start closes the previous wedge first.
    return {index: sorted(items, key=lambda item: (item[0], item[1] != 'WedgeStop'))
            for index, items in marks.items()}


def _measure_text_with_tempos(notes: list[JianpuNote], marks: list[tuple[float, str]],
                              measure_number: int,
                              dynamics: Optional[list[tuple[float, str]]] = None,
                              meters: Optional[list[tuple[float, str]]] = None,
                              wedges: Optional[list[tuple[float, str]]] = None) -> str:
    """Insert marks only at a rendered event boundary, never move them silently."""
    pending = sorted(marks, key=lambda item: item[0])
    pending_dynamics = sorted(dynamics or [], key=lambda item: item[0])
    pending_meters = sorted(meters or [], key=lambda item: item[0])
    pending_wedges = list(wedges or [])
    words: list[str] = []
    position = 0.0
    for note in notes:
        matched_wedges = [mark for mark in pending_wedges if abs(mark[0] - position) < 1e-6]
        words.extend(word for _, word in matched_wedges)
        pending_wedges = [mark for mark in pending_wedges if abs(mark[0] - position) >= 1e-6]
        matched = [mark for mark in pending if abs(mark[0] - position) < 1e-6]
        words.extend(word for _, word in matched)
        pending = [mark for mark in pending if abs(mark[0] - position) >= 1e-6]
        token = jianpu_note_token(note)
        if not note.is_grace:
            matched_dynamics = [mark for mark in pending_dynamics if abs(mark[0] - position) < 1e-6]
            pending_dynamics = [mark for mark in pending_dynamics if abs(mark[0] - position) >= 1e-6]
            matched_meters = [mark for mark in pending_meters if abs(mark[0] - position) < 1e-6]
            pending_meters = [mark for mark in pending_meters if abs(mark[0] - position) >= 1e-6]
            if matched_dynamics or matched_meters:
                # A long Jianpu event may expand to "1 - -".  LilyPond
                # dynamics attach to the *preceding* event, so insert after
                # the first note word, not after the last duration dash.
                parts = token.split()
                index = 1 if len(parts) > 1 and parts[0].endswith('[') else 0
                parts[index + 1:index + 1] = [word for _, word in (*matched_dynamics, *matched_meters)]
                token = ' '.join(parts)
        words.append(token)
        if not note.is_grace:
            position += note.duration
    matched_wedges = [mark for mark in pending_wedges if abs(mark[0] - position) < 1e-6]
    words.extend(word for _, word in matched_wedges)
    pending_wedges = [mark for mark in pending_wedges if abs(mark[0] - position) >= 1e-6]
    for offset, _ in pending:
        log_message(f'[jianpu] 小节 {measure_number} 的速度记号位于 {offset:g} 拍，'
                    '不在可表示的音符边界，需人工校对。', _logging.WARNING)
    for offset, _ in pending_dynamics:
        log_message(f'[jianpu] 小节 {measure_number} 的力度记号位于 {offset:g} 拍，'
                    '不在可表示的音符边界，需人工校对。', _logging.WARNING)
    for offset, _ in pending_meters:
        log_message(f'[jianpu] 小节 {measure_number} 的拍号位于 {offset:g} 拍，'
                    '不在可表示的音符边界，需人工校对。', _logging.WARNING)
    for offset, _ in pending_wedges:
        log_message(f'[jianpu] 小节 {measure_number} 的渐强渐弱线端点位于 {offset:g} 拍，'
                    '无法准确排版，请校对。', _logging.WARNING)
    return ' '.join(words)


def parse_score_to_jianpu(score, pitch_mode: str = 'movable',
                          fixed_tonic: str = 'C') -> tuple[list[list[JianpuNote]], list[str], str]:
    """Parse a score into jianpu format; return (measures, header_lines, time_sig)."""
    key_tonic_semitone, key_header = _get_score_key_tonic(score)
    if pitch_mode == 'fixed':
        key_tonic_semitone, _, tonic_name = fixed_reference(fixed_tonic)
        key_header = f'1={tonic_name}'

    measures, time_signature = extract_jianpu_measures(
        score, key_tonic_semitone, pitch_mode=pitch_mode, fixed_tonic=fixed_tonic)
    header_lines: list[str] = [f'{key_header} {time_signature}', '']

    measures_per_line = choose_measures_per_line(measures)
    log_message(f'[jianpu] parse_score_to_jianpu: {len(measures)} 个小节（每行 {measures_per_line} 个）', _logging.DEBUG)

    for i in range(0, len(measures), measures_per_line):
        line_measures = measures[i:i + measures_per_line]
        try:
            measure_texts = [' '.join(format_jianpu_note_text(note) for note in measure) for measure in line_measures]
            header_lines.append(' | '.join(measure_texts) + ' |')
        except Exception as exc:
            log_message(f'[jianpu] parse_score_to_jianpu 处理小节组 [{i}:{i+measures_per_line}] 失败: {exc}', _logging.WARNING)
            raise

    return measures, header_lines, time_signature


def build_jianpu_ly_text_from_measures(
    measures: 'list[list[JianpuNote]]',
    time_sig: str,
    key_header: str,
    title: str,
    composer: str = '',
    tempo: int = 0,
) -> str:
    """Build jianpu-ly plain-text from pre-computed JianpuNote measures (bypassing score parsing).

    Equivalent to ``build_jianpu_ly_text`` but accepts already-parsed note data
    instead of a music21 Score object.  Used by the dual-engine fusion path.

    *key_header* is the already-formatted jianpu-ly key line (e.g. ``'1=F'``
    or ``'6=D'`` for minor — see ``_get_score_key_tonic``), not a bare tonic name.
    """
    header = ['% jianpu-ly.py', f'title={title}']
    if composer:
        header.append(f'composer={composer}')
    header.append(key_header)
    header.append(_header_time_signature(time_sig))
    if tempo > 0:
        header.append(f'4={tempo}')
    # 合音的高/低八度点需明确归属位置（见 build_jianpu_ly_text 中的同名注释）
    header.append('OctavesAfter')
    header.append('')
    bar_units = _parse_bar_units(time_sig)
    time_sig_tokens = layout_time_tokens(measures, bar_units)
    for i in range(0, len(measures), 4):
        line_measures = measures[i:i + 4]
        # The notes are emitted exactly as extracted: no padding, no trimming.
        measure_texts = [
            ' '.join(jianpu_note_token(note) for note in m)
            for m in line_measures
        ]
        header.append(' | '.join(_with_time_signatures(measure_texts, i, time_sig_tokens)) + ' |')
    header.extend(build_lyric_lines(measures))
    return '\n'.join(header)


@overload
def build_jianpu_ly_text(score, title: str, use_strict_timing: bool = False,
                          composer: str = '', tempo: int = 0,
                          _return_groups: Literal[False] = False, *, _source_wedges=None,
                          pitch_mode: str = 'movable', fixed_tonic: str = 'C') -> str: ...
@overload
def build_jianpu_ly_text(score, title: str, use_strict_timing: bool = False,
                          composer: str = '', tempo: int = 0,
                          *, _return_groups: Literal[True], _source_wedges=None,
                          pitch_mode: str = 'movable', fixed_tonic: str = 'C',
                          ) -> tuple[str, list[list[int]], dict[int, dict[str, bool]]]: ...
def build_jianpu_ly_text(score, title: str, use_strict_timing: bool = False,
                          composer: str = '', tempo: int = 0,
                          _return_groups: bool = False, *, _source_wedges=None,
                          pitch_mode: str = 'movable', fixed_tonic: str = 'C'):
    """Build jianpu-ly plain-text for all parts in the score.

    Source scores are monophonic: each part becomes exactly one jianpu section,
    so one staff always maps to one jianpu line.  A second MusicXML voice inside
    a measure is OMR recognition noise; it is merged into that part's line
    instead of getting a track of its own.

    Parameters
    ----------
    _return_groups
        When *True* the function returns ``(text, voice_groups)`` instead of
        just *text*.  ``voice_groups`` is a ``list[list[int]]`` where each
        inner list contains the 0-based section indices to render on the same
        jianpu staff.  With monophonic sources every group holds one section.
    """
    if pitch_mode not in ('movable', 'fixed'):
        raise ValueError(f'unsupported pitch mode: {pitch_mode}')
    key_tonic_semitone, key_header_line = _get_score_key_tonic(score)
    if pitch_mode == 'fixed':
        key_tonic_semitone, _, tonic_name = fixed_reference(fixed_tonic)
        key_header_line = f'1={tonic_name}'
    if _source_wedges is None:
        from music21 import dynamics as m21dynamics
        if list(score.recurse().getElementsByClass(m21dynamics.DynamicWedge)):
            log_message('[jianpu] 未提供原始 MusicXML，无法可靠确定渐强渐弱线端点；请校对。', _logging.WARNING)
    if use_strict_timing:
        log_message('[jianpu] 严格模式重分小节，无法可靠对应原始速度、力度、渐强渐弱线和拍号位置；请校对。',
                    _logging.WARNING)

    # ── Collect one measures-list per output section ──────────────────────────
    _parts = list(score.parts) if score.parts else []
    _sections: list[list[list[JianpuNote]]] = []
    _section_parts: list = []
    _voice_groups: list[list[int]] = []   # parallel to _sections groups
    _time_sig: str = '4/4'
    _ts_set = False

    if len(_parts) <= 1:
        part = _parts[0] if _parts else None
        if use_strict_timing:
            measures, _time_sig = extract_strict_jianpu_measures(
                score, key_tonic_semitone, pitch_mode=pitch_mode, fixed_tonic=fixed_tonic)
        else:
            measures, _time_sig = extract_jianpu_measures(
                score, key_tonic_semitone, follow_key_changes=True, pitch_mode=pitch_mode,
                fixed_tonic=fixed_tonic)
        _sections = [measures]
        _section_parts = [part if part is not None else score]
        _voice_groups = [[0]]
    else:
        # Multi-part path: one section per part.  Source parts are monophonic;
        # any extra MusicXML voice is OMR noise and is merged into that part's line.
        for part in _parts:
            if use_strict_timing:
                measures, ts = extract_strict_jianpu_measures(score, key_tonic_semitone, _part=part,
                                                               pitch_mode=pitch_mode,
                                                               fixed_tonic=fixed_tonic)
            else:
                measures, ts = extract_jianpu_measures(score, key_tonic_semitone, _part=part,
                                                       follow_key_changes=True, pitch_mode=pitch_mode,
                                                       fixed_tonic=fixed_tonic)
            if not _ts_set:
                _time_sig, _ts_set = ts, True
            if measures:
                _voice_groups.append([len(_sections)])
                _sections.append(measures)
                _section_parts.append(part)

    # ── Assemble jianpu-ly text ───────────────────────────────────────────────
    header = ['% jianpu-ly.py', f'title={title}']
    if composer:
        header.append(f'composer={composer}')
    header.append(key_header_line)
    header.append(_header_time_signature(_time_sig))
    if tempo > 0:
        header.append(f'4={tempo}')
    # 合音（,135 形式）内部的高/低八度点必须明确归属，否则 jianpu-ly 报
    # "Ambiguous octave marks"。OctavesAfter = 记号作用于其前面的音（默认约定）。
    header.append('OctavesAfter')
    header.append('')

    bar_units   = _parse_bar_units(_time_sig)
    measures_per_line = 4

    log_message(
        f'[jianpu] build_jianpu_ly_text: {len(_sections)} 个段落，共 '
        f'{sum(len(s) for s in _sections)} 个小节', _logging.DEBUG,
    )

    for section_idx, measures in enumerate(_sections):
        if section_idx > 0:
            header.append('NextPart')
            # jianpu-ly resets barLength per part; repeat the header signature so
            # the reset value is the same one the tokens below are measured from.
            header.append(_header_time_signature(_time_sig))
        # Each section gets its own staff and its own barLength, so the layout
        # tokens restart from the header's bar length for every section.
        section_time_sigs = layout_time_tokens(measures, bar_units)
        tempo_marks = ({} if use_strict_timing
                       else _measure_tempo_marks(score, _section_parts[section_idx], len(measures)))
        meter_marks = ({} if use_strict_timing
                       else _measure_meter_marks(score, _section_parts[section_idx], len(measures)))
        part_index = next((index for index, part in enumerate(_parts)
                           if part is _section_parts[section_idx]), 0)
        part_wedges = (_source_wedges[part_index] if _source_wedges and part_index < len(_source_wedges) else [])
        wedge_marks = {} if use_strict_timing else _wedge_marks(measures, part_wedges)
        # If sections are ever grouped on one staff, print its dynamic once.
        primary_section = next((group[0] for group in _voice_groups if section_idx in group), section_idx)
        dynamic_marks = (_measure_dynamic_marks(score, _section_parts[section_idx], len(measures))
                         if not use_strict_timing and section_idx == primary_section else {})
        if tempo > 0 and 0 in tempo_marks:
            # A caller-supplied initial tempo overrides source marks at time zero.
            tempo_marks[0] = [(offset, word) for offset, word in tempo_marks[0]
                              if abs(offset) >= 1e-6]
        for i in range(0, len(measures), measures_per_line):
            line_measures = measures[i:i + measures_per_line]
            try:
                # Emitted exactly as extracted: no padding, no trimming.
                measure_texts = [
                    _measure_text_with_tempos(m, tempo_marks.get(i + offset, []), i + offset + 1,
                                              dynamic_marks.get(i + offset, []), meter_marks.get(i + offset, []),
                                              wedge_marks.get(i + offset, []))
                    for offset, m in enumerate(line_measures)
                ]
                changes = getattr(measures, 'key_changes', {})
                measure_texts = [f'{changes[i + offset][1]} {text}'
                                 if i + offset in changes else text
                                 for offset, text in enumerate(measure_texts)]
                header.append(
                    ' | '.join(_with_time_signatures(measure_texts, i, section_time_sigs)) + ' |')
            except Exception as exc:
                log_message(
                    f'[jianpu] 处理声部 {section_idx} 小节组 [{i}:{i+measures_per_line}] 失败: {exc}',
                    _logging.WARNING,
                )
                raise
        header.extend(build_lyric_lines(measures))

    text = '\n'.join(header)
    if _return_groups:
        _first_part = _parts[0] if _parts else None
        _repeat_barlines: dict[int, dict[str, bool]] = (
            _extract_part_repeat_barlines(_first_part) if _first_part is not None else {}
        )
        return text, _voice_groups, _repeat_barlines
    return text
