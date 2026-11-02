# core/notation/jianpu/extract.py — Voice analysis and score extraction for Jianpu.
import logging as _logging
from typing import Optional

from music21 import chord as m21chord, key as m21key, note as m21note, spanner, stream

from ...config import JianpuNote
from ...utils import log_message
from .primitives import (
    MAX_SANE_BARS,
    _QL_TO_ANACRUSIS_CODE,
    infer_duration_dots,
    normalize_jianpu_duration,
    note_to_jianpu,
    split_duration_chunks,
    _get_score_key_tonic,
    _reference_step_from_header,
    _tonic_result,
)
from .measure import (
    MeasureBarUnits,
    _detect_trailing_rest_pickup,
    _parse_bar_units,
    clone_monophonic_element,
    measure_bar_signature,
    measure_units,
    repair_jianpu_measure,
)


def _rest_of(duration: float):
    """A music21 Rest of *duration* quarter lengths."""
    rest = m21note.Rest()
    rest.duration.quarterLength = duration
    return rest


def _element_duration(item) -> float:
    """Quarter length of a collected measure element (a JianpuNote or m21 clone)."""
    if isinstance(item, JianpuNote):
        return float(item.duration)
    duration = getattr(item, 'duration', 0.0)
    return float(getattr(duration, 'quarterLength', duration) or 0.0)


def _mark_tuplet_groups(notes: list[JianpuNote]) -> None:
    """Mark consecutive tuplet events for the jianpu-ly ``3[ ... ]`` syntax."""
    # Grace notes do not consume a tuplet slot and must not split its group.
    notes = [event for event in notes if not event.is_grace]
    index = 0
    while index < len(notes):
        first = notes[index]
        if not first.tuplet_actual or not first.tuplet_normal:
            index += 1
            continue
        ratio = (first.tuplet_actual, first.tuplet_normal)
        end = index + 1
        while end < len(notes) and (notes[end].tuplet_actual, notes[end].tuplet_normal) == ratio:
            end += 1
        cursor = index
        while cursor < end:
            stop = cursor
            explicit = any(n.tuplet_type in ('start', 'stop', 'startStop') for n in notes[cursor:end])
            if explicit:
                if notes[cursor].tuplet_type not in ('start', 'startStop'):
                    log_message('[jianpu] 连音符缺少起始标记或跨越小节，请校对。', _logging.WARNING)
                while stop < end:
                    if stop > cursor and notes[stop].tuplet_type == 'start':
                        stop -= 1
                        break
                    if notes[stop].tuplet_type in ('stop', 'startStop'):
                        break
                    stop += 1
                stop = min(stop, end - 1)
            else:
                stop = min(end - 1, cursor + ratio[0] - 1)
            if ((explicit and notes[stop].tuplet_type not in ('stop', 'startStop'))
                    or (not explicit and stop - cursor + 1 < ratio[0])):
                log_message('[jianpu] 连音符分组不完整或跨越小节，请校对。', _logging.WARNING)
            notes[cursor].tuplet_start = True
            notes[stop].tuplet_stop = True
            cursor = stop + 1
        index = end


def strip_grace_notes(score) -> int:
    """Remove grace notes from *score* in place and return how many were removed.

    Used on a separate MIDI copy because music21's MIDI export mishandles grace
    duration. The Jianpu extractor keeps them as zero-time ornaments.
    """
    removed = 0
    for container in score.recurse(streamsOnly=True, includeSelf=True):
        for el in [e for e in container.notes if e.duration.isGrace]:
            container.remove(el)
            removed += 1
    return removed


def _canonical_offset(seen: dict[int, float], offset: float) -> float:
    """Merge numerical noise only, keeping genuinely distinct onsets apart.

    A fixed 1/64-note grid conflates distinct notes in shorter tuplets.  The
    small tolerance handles floating point noise without snapping to a rhythm.
    """
    for previous in seen.values():
        if abs(previous - offset) <= 1e-7:
            return previous
    seen[len(seen)] = offset
    return offset


def _get_voice_ids_in_part(part) -> list[str]:
    """Return all explicit voices with notes or rests, including short entries.

    Length and rhythmic overlap cannot distinguish an OMR artefact from a real
    counterpoint voice.  Preserve the source and leave uncertain OMR results for
    review instead of silently dropping a potentially genuine voice.
    """
    voice_ids: set[str] = set()
    for measure in part.getElementsByClass(stream.Measure):
        for voice in measure.voices:
            if list(voice.flatten().notesAndRests):
                voice_ids.add(str(voice.id))
    return sorted(voice_ids)


def _extract_part_repeat_barlines(part) -> dict[int, dict[str, bool]]:
    """Return {measure_index: {'start': bool, 'end': bool}} for measures with repeat barlines.

    'start' — measure's left barline is a start-repeat (⌃|:)
    'end'   — measure's right barline is an end-repeat (:|)
    """
    try:
        from music21 import bar as m21bar
    except ImportError:
        return {}

    result: dict[int, dict[str, bool]] = {}
    measures = list(part.getElementsByClass('Measure'))
    if len(measures) > MAX_SANE_BARS:
        measures = measures[:MAX_SANE_BARS]

    for i, measure in enumerate(measures):
        lb = getattr(measure, 'leftBarline', None)
        rb = getattr(measure, 'rightBarline', None)
        has_start = isinstance(lb, m21bar.Repeat) and getattr(lb, 'direction', None) == 'start'
        has_end   = isinstance(rb, m21bar.Repeat) and getattr(rb, 'direction', None) == 'end'
        if has_start or has_end:
            result[i] = {'start': has_start, 'end': has_end}

    return result


def extract_jianpu_measures(score, key_tonic_semitone: int = 0,
                             _part=None, _voice_id: str = '1',
                             _multi_voice_mode: bool = False,
                             _is_primary_voice: bool = True,
                             *, follow_key_changes: bool = False) -> tuple[list[list[JianpuNote]], str]:
    """
    Extract jianpu measure list and time signature from a music21 score.
    Merges polyphony by offset; fills gaps and overflows with rests at bar boundaries.

    Parameters
    ----------
    _part              : If provided, use this Part stream directly instead of score.parts[0].
    _voice_id          : Which voice ID to extract when a measure contains multiple voices.
                         Defaults to '1' (main melody).
    _multi_voice_mode  : When True (multi-voice extraction), a measure that lacks the
                         requested voice emits a whole-measure rest instead of falling
                         back to another voice's content.  This prevents secondary voices
                         from duplicating the primary melody in monophonic measures.
    _is_primary_voice  : When *_multi_voice_mode* is True, the primary voice still gets the
                         flat measure content when the measure has no explicit voice
                         containers.  Secondary voices (``_is_primary_voice=False``) emit
                         rests in such measures.
    follow_key_changes : Apply source key changes at measure boundaries and
                         expose their headers through ``measures.key_changes``.
    """
    part = _part if _part is not None else (score.parts[0] if score.parts else score.flatten())
    time_signature = '4/4'
    nominal_measure_length = 4.0
    time_sig = part.recurse().getElementsByClass('TimeSignature')
    if time_sig:
        time_signature = time_sig[0].ratioString
        nominal_measure_length = float(getattr(time_sig[0].barDuration, 'quarterLength', 4.0) or 4.0)

    measures: MeasureBarUnits = MeasureBarUnits()
    initial_tonic, initial_header = _get_score_key_tonic(score)
    active_header = initial_header if initial_tonic == key_tonic_semitone else None
    tonic_step = _reference_step_from_header(active_header, key_tonic_semitone)
    measure_streams = list(part.getElementsByClass(stream.Measure))
    if not measure_streams:
        fallback_notes = [note_to_jianpu(element, key_tonic_semitone, tonic_step=tonic_step)
                          for element in part.flatten().notesAndRests]
        fallback = MeasureBarUnits([fallback_notes] if fallback_notes else [])
        if fallback:
            fallback.bar_units.append(_parse_bar_units(time_signature))
            fallback.time_signatures.append(time_signature.split(',')[0])
        return fallback, time_signature

    if len(measure_streams) > MAX_SANE_BARS:
        log_message(
            f'[jianpu] 注意：MusicXML 包含 {len(measure_streams)} 个小节（超过安全限制 {MAX_SANE_BARS}），'
            f'可能是 OMR 解析异常。将截尾处理前 {MAX_SANE_BARS} 个小节。',
            _logging.WARNING,
        )
        measure_streams = measure_streams[:MAX_SANE_BARS]

    # ── Pickup (anacrusis / 弱起) detection ───────────────────────────────────
    # A pickup measure has paddingLeft > 0 in music21 (ported from MusicXML's implicit="yes").
    # Actual pickup duration = nominal_measure_length - paddingLeft.
    _first_measure_length_override: Optional[float] = None
    _first_m = measure_streams[0]
    _padding_left = float(getattr(_first_m, 'paddingLeft', 0.0) or 0.0)
    if _padding_left > 0.01 and _padding_left < nominal_measure_length - 0.01:
        _actual_pickup = nominal_measure_length - _padding_left
        _closest_ql = min(_QL_TO_ANACRUSIS_CODE, key=lambda q: abs(q - _actual_pickup))
        if abs(_closest_ql - _actual_pickup) < 0.25:
            time_signature = f'{time_signature},{_QL_TO_ANACRUSIS_CODE[_closest_ql]}'
            _first_measure_length_override = _closest_ql
    # Fallback: detect pickup from trailing-rest pattern (implicit="no" but notes+filler-rest)
    if _first_measure_length_override is None:
        _fallback_ql = _detect_trailing_rest_pickup(_first_m, nominal_measure_length)
        if _fallback_ql is not None:
            time_signature = f'{time_signature},{_QL_TO_ANACRUSIS_CODE[_fallback_ql]}'
            _first_measure_length_override = _fallback_ql

    tol = 0.01
    # How long the previous measure's *content* actually was.  Only used to give
    # a measure the OMR left completely empty a sane width — see the ``not
    # by_offset`` branch below.
    _previous_content = nominal_measure_length or 4.0
    deferred_key = None
    for _m_idx, measure in enumerate(measure_streams):
        if follow_key_changes:
            boundary_key = deferred_key
            deferred_key = None
            for signature in measure.recurse().getElementsByClass(m21key.KeySignature):
                signature_offset = float(signature.getOffsetInHierarchy(measure))
                try:
                    if isinstance(signature, m21key.Key):
                        ref, header = _tonic_result(signature.tonic, signature.mode)
                        explicit_mode = True
                    else:
                        major = signature.asKey('major')
                        ref, header = _tonic_result(major.tonic, 'major')
                        explicit_mode = False
                except Exception:
                    log_message(f'[jianpu] 小节 {_m_idx + 1} 的调号无法转换，保持之前的唱名；'
                                '请核对。', _logging.WARNING)
                    continue
                candidate = (ref, header, explicit_mode)
                if abs(signature_offset) < 1e-7:
                    boundary_key = candidate
                else:
                    deferred_key = candidate
                    log_message(f'[jianpu] 小节 {_m_idx + 1} 中途变调，保持当前唱名并在下一小节切换；'
                                '请核对调号位置。', _logging.WARNING)
            if boundary_key is not None:
                ref, header, explicit_mode = boundary_key
                changed_spelling = (active_header and active_header.startswith('1=')
                                    and header.startswith('1=') and active_header != header)
                if ref != key_tonic_semitone or (explicit_mode and active_header != header) or changed_spelling:
                    measures.key_changes[_m_idx] = (ref, header)
                    key_tonic_semitone, active_header = ref, header
                    tonic_step = _reference_step_from_header(active_header, key_tonic_semitone)
        # The measure's length is decided *after* its notes are collected: it is
        # wherever the model's last note ends (see ``measure_length`` below).
        # No time signature is consulted — a bar the OMR under-filled keeps its
        # notes and simply ends early (no padding '0'), and a bar it over-filled
        # keeps every note (nothing is dropped or clamped).  The pickup
        # detection above stays only for the signature word shown in the UI.
        by_offset: dict[float, list[object]] = {}
        _offset_seen: dict[int, float] = {}  # Preserve first offset for epsilon-only deduplication.
        slur_boundaries: dict[tuple[float, tuple[int, ...]], list[int]] = {}
        for source_note in measure.recurse().notes:
            if getattr(source_note.duration, 'isGrace', False):
                continue
            pitches = (tuple(sorted(p.ps for p in source_note.pitches))
                       if isinstance(source_note, m21chord.Chord) else (source_note.pitch.ps,))
            key = (round(float(source_note.getOffsetInHierarchy(measure)), 7), pitches)
            for slur in source_note.getSpannerSites(spanner.Slur):
                ends = slur.getSpannedElements()
                # Grace-note curves are already rendered by the grace syntax.
                if len(ends) < 2 or any(getattr(e.duration, 'isGrace', False)
                                        for e in (ends[0], ends[-1])):
                    continue
                flags = slur_boundaries.setdefault(key, [0, 0])
                flags[0] += int(slur.isFirst(source_note))
                flags[1] += int(slur.isLast(source_note))

        # Keep tied fragments separate: merging can create a duration such as
        # 2.5 quarters that one Jianpu token cannot express, causing truncation.
        measure_src = measure

        # 简谱为单声部格式；使用 _voice_id 参数选取目标声部。
        # 单声部模式下 _voice_id='1'，多声部渲染时由调用方传入具体 ID。
        _voices = list(measure_src.voices)
        if _voices:
            _matching_voice = next((v for v in _voices if str(v.id) == _voice_id), None)
            if _matching_voice is not None:
                _flat_all = list(_matching_voice.flatten().notesAndRests)
            elif _multi_voice_mode:
                # Multi-voice mode: this voice doesn't exist in this measure → whole-measure rest.
                # Do NOT fall back to another voice's content.
                _flat_all = []
            else:
                # Single-voice mode fallback: pick the voice with the most non-rest notes.
                # music21 sometimes places a rest-only Voice container alongside other voices
                # that hold the actual notes (e.g. when an OMR-tagged sub-voice steals the
                # measure's content from the nominal main voice); falling back to _voices[0]
                # in that case would silently drop all real notes for the measure.
                _best_voice = max(
                    _voices,
                    key=lambda v: sum(
                        1 for e in v.flatten().notes if isinstance(e, m21note.Note)
                    ),
                )
                _flat_all = list(_best_voice.flatten().notesAndRests)
        else:
            if _multi_voice_mode and not _is_primary_voice:
                # A chord without explicit voices belongs to the primary
                # track.  Copying its lower pitch into a missing secondary
                # voice invents notes that were never assigned to that voice.
                _flat_all = []
            else:
                _flat_all = list(measure_src.flatten().notesAndRests)

        for element in _flat_all:
            offset = _canonical_offset(_offset_seen, float(element.offset))
            if getattr(element.duration, 'isGrace', False):
                grace = note_to_jianpu(element, key_tonic_semitone, tonic_step=tonic_step)
                grace.is_grace = True
                grace.duration = 0.0
                by_offset.setdefault(offset, []).append(grace)
                continue
            # 跨小节 tie-stop/continue：上一小节 tie-start 音符在本小节内的延续。
            # 渲染为延音 dash（'-'），而非独立音符或休止符。
            _tie = getattr(element, 'tie', None)
            if _tie is not None and _tie.type in ('stop', 'continue'):
                cont = note_to_jianpu(element, key_tonic_semitone, tonic_step=tonic_step)
                pitches = (cont.midi, *[p[4] for p in cont.chord_notes])
                starts, stops = slur_boundaries.pop((round(offset, 7), pitches), (0, 0))
                cont.slur_starts, cont.slur_stops = starts, stops
                cont.symbol, cont.accidental = '-', ''
                cont.upper_dots = cont.lower_dots = 0
                cont.midi, cont.chord_notes, cont.lyrics = None, (), {}
                existing = by_offset.get(offset)
                if existing is None:
                    by_offset[offset] = [cont]
                else:
                    existing.append(cont)
                continue
            # No clamping to the barline: the note keeps the duration the model
            # gave it.  (Clamping here silently shortened the last note of any bar
            # the OMR had over-filled, and stopped earlier bars from being "too
            # long" at all.)
            candidate = clone_monophonic_element(element, float(element.duration.quarterLength or 0.25))
            existing = by_offset.get(offset)
            if existing is None:
                by_offset[offset] = [candidate]
            else:
                existing.append(candidate)

        if not by_offset:
            # The model recognised nothing in this measure.  Print one bar of
            # silence, as wide as the previous measure (there is no meter here to
            # take a width from).  Silence is the only thing left to render; this
            # is not padding to fit a signature.
            measure_length = _previous_content
            measures.append([
                note_to_jianpu(_rest_of(piece), key_tonic_semitone, tonic_step=tonic_step)
                for piece in (split_duration_chunks(measure_length) or [measure_length])
            ])
            measures.bar_units.append(int(round(measure_length * 16)))
            measures.time_signatures.append(measure_bar_signature(measure))
            continue

        # ── The measure ends where the model's last note ends ────────────────
        # This is the length everything below lays the measure out against.  It
        # is derived from the durations that will actually be emitted (already
        # normalised onto the jianpu grid by clone_monophonic_element), so the
        # jianpu-ly text builder reproduces it exactly by summing the tokens.
        measure_length = max(
            offset + _element_duration(item)
            for offset, items in by_offset.items() for item in items
        )
        _previous_content = measure_length

        offsets = sorted(by_offset.keys())
        current_offset = 0.0
        measure_notes: list[JianpuNote] = []

        for idx, offset in enumerate(offsets):
            if current_offset >= measure_length - tol:
                break
            if offset > current_offset + tol:
                gap = min(offset - current_offset, measure_length - current_offset)
                for piece in split_duration_chunks(gap):
                    rest_piece = m21note.Rest()
                    rest_piece.duration.quarterLength = piece
                    measure_notes.append(note_to_jianpu(rest_piece, key_tonic_semitone, tonic_step=tonic_step))
                    current_offset += piece

            next_offset = offsets[idx + 1] if idx + 1 < len(offsets) else measure_length
            next_offset = min(next_offset, measure_length)
            element_end = offset
            for element in by_offset[offset]:
                if isinstance(element, JianpuNote):
                    measure_notes.append(element)
                    element_end = max(element_end, offset + element.duration)
                else:
                    jn = note_to_jianpu(element, key_tonic_semitone, tonic_step=tonic_step)
                    if not jn.is_rest:
                        pitches = (jn.midi, *[p[4] for p in jn.chord_notes])
                        starts, stops = slur_boundaries.pop((round(offset, 7), pitches), (0, 0))
                        jn.slur_starts, jn.slur_stops = starts, stops
                    measure_notes.append(jn)
                    element_end = max(element_end, offset + jn.duration)
            # 填充音符实际结束位置到下一个音符开始位置之间的空白（OMR 漏标休止符的常见情形）
            if element_end < next_offset - tol:
                gap = min(next_offset - element_end, measure_length - element_end)
                for piece in split_duration_chunks(gap):
                    rest_piece = m21note.Rest()
                    rest_piece.duration.quarterLength = piece
                    measure_notes.append(note_to_jianpu(rest_piece, key_tonic_semitone, tonic_step=tonic_step))
            current_offset = next_offset

        # No trailing rest: the measure already ends with the model's last note,
        # so there is nothing left to fill.  (This is the step that used to append
        # the stray '0's whenever the OMR's own note total fell short of the
        # signature it had guessed.)
        _mark_tuplet_groups(measure_notes)
        measures.append(measure_notes)
        # Log what this measure's notes add up to, verbatim.  The jianpu-ly text
        # builder recomputes the same sum from the emitted tokens and makes it the
        # measure's bar length (see measure_units / layout_time_word).
        measures.bar_units.append(measure_units(measure_notes))
        measures.time_signatures.append(measure_bar_signature(measure))

    return measures, time_signature


def extract_strict_jianpu_measures(score, key_tonic_semitone: int = 0,
                                    _part=None) -> tuple[list[list[JianpuNote]], str]:
    """
    Extract jianpu measures by re-slicing all notes strictly by bar length,
    ignoring the score's own Measure objects.

    Parameters
    ----------
    _part : If provided, use this Part stream directly instead of score.parts[0].
            All voices are merged (flattened) — use extract_jianpu_measures with
            _voice_id for per-voice extraction.
    """
    part = _part if _part is not None else (score.parts[0] if score.parts else score.flatten())
    initial_tonic, header = _get_score_key_tonic(score)
    tonic_step = _reference_step_from_header(header, key_tonic_semitone) if initial_tonic == key_tonic_semitone else None
    # 合并延音线：同小节 start→stop 对合并为单音符；跨小节 tie 的合并音符由
    # append_element_chunks 正确跨 bar 拆分。
    try:
        part = part.stripTies()
    except Exception:
        pass
    time_signature = '4/4'
    bar_length = 4.0
    time_sig = part.recurse().getElementsByClass('TimeSignature')
    if time_sig:
        time_signature = time_sig[0].ratioString
        bar_length = float(getattr(time_sig[0].barDuration, 'quarterLength', 4.0) or 4.0)

    max_score_offset = MAX_SANE_BARS * bar_length  # hard ceiling on note offsets

    # ── Pickup (anacrusis / 弱起) detection ───────────────────────────────────
    _first_pickup_ql: Optional[float] = None
    _trailing_rest_pickup = False  # was pickup inferred from trailing-rest pattern?
    _m_list = list(part.getElementsByClass(stream.Measure))
    if _m_list:
        _pl = float(getattr(_m_list[0], 'paddingLeft', 0.0) or 0.0)
        if _pl > 0.01 and _pl < bar_length - 0.01:
            _act = bar_length - _pl
            _closest = min(_QL_TO_ANACRUSIS_CODE, key=lambda q: abs(q - _act))
            if abs(_closest - _act) < 0.25:
                time_signature = f'{time_signature},{_QL_TO_ANACRUSIS_CODE[_closest]}'
                _first_pickup_ql = _closest
    # Fallback: detect pickup from trailing-rest pattern (implicit="no" but notes+filler-rest)
    if _first_pickup_ql is None and _m_list:
        _fallback_ql = _detect_trailing_rest_pickup(_m_list[0], bar_length)
        if _fallback_ql is not None:
            time_signature = f'{time_signature},{_QL_TO_ANACRUSIS_CODE[_fallback_ql]}'
            _first_pickup_ql = _fallback_ql
            _trailing_rest_pickup = True

    by_offset: dict[float, object] = {}
    _offset_seen: dict[int, float] = {}  # Preserve first offset for epsilon-only deduplication.
    for element in part.flatten().notesAndRests:
        offset = _canonical_offset(_offset_seen, float(element.offset))
        if offset < 0 or offset >= max_score_offset:
            continue
        candidate = clone_monophonic_element(element, float(element.duration.quarterLength or 0.25))
        existing = by_offset.get(offset)
        if existing is None:
            by_offset[offset] = candidate
        elif isinstance(existing, m21note.Rest) and not isinstance(candidate, m21note.Rest):
            by_offset[offset] = candidate
        elif isinstance(existing, m21note.Note) and isinstance(candidate, m21note.Note) and candidate.pitch.ps > existing.pitch.ps:
            by_offset[offset] = candidate

    # For trailing-rest pickups: rebuild by_offset so that the layout matches
    # what append_element_chunks expects (identical to the implicit="yes" case).
    # Remove filler rests in [pickup_ql, bar_length) and shift all notes at
    # offset >= bar_length by -(bar_length - pickup_ql), so bar 1 starts
    # immediately after the pickup with no silent gap.
    if _trailing_rest_pickup and _first_pickup_ql is not None:
        _shift = bar_length - _first_pickup_ql
        _adjusted: dict[float, object] = {}
        for _off, _elem in by_offset.items():
            if _first_pickup_ql <= _off < bar_length:
                continue  # skip filler rests / elements
            elif _off >= bar_length:
                _adjusted[_off - _shift] = _elem  # shift into correct position
            else:
                _adjusted[_off] = _elem  # keep pickup notes as-is
        by_offset = _adjusted

    offsets = sorted(by_offset.keys())
    if not offsets:
        return [], time_signature

    # 警告：如果最大 offset 超过合理上限
    if offsets[-1] >= max_score_offset * 0.5 and offsets[-1] > bar_length * 30:
        log_message(
            f'[jianpu] 注意：MusicXML 中最大音符偏移量为 {offsets[-1]:.1f} 拍'
            f'（约 {offsets[-1] / bar_length:.0f} 小节），可能是 OMR 解析异常。',
            _logging.WARNING,
        )

    measures: list[list[JianpuNote]] = []
    current_measure: list[JianpuNote] = []
    current_pos = 0.0
    tol = 0.01


    def append_element_chunks(element_template, total_duration: float) -> None:
        """Split an element across bar boundaries and append chunks to the current measure."""
        nonlocal current_measure, current_pos, measures
        remaining = total_duration
        while remaining > tol:
            # 安全检查：超过小节上限时提前退出
            if len(measures) >= MAX_SANE_BARS:
                return
            # 弱起小节：第一小节的边界是 _first_pickup_ql，后续小节按 bar_length 推进
            if not measures and _first_pickup_ql:
                capacity = _first_pickup_ql - current_pos
            elif _first_pickup_ql:
                pos_in_bar = (current_pos - _first_pickup_ql) % bar_length
                capacity = bar_length - pos_in_bar if pos_in_bar > tol else bar_length
            else:
                pos_in_bar = current_pos % bar_length
                capacity = bar_length - pos_in_bar if pos_in_bar > tol else bar_length
            piece_total = min(remaining, max(capacity, 0.0))
            if piece_total <= tol:
                # 容量耗尽但仍有剩余（理论上仅弱起边界浮点漂移时出现）。
                # 旧行为靠 split_duration_chunks(0)→[0.125] 的假音符推进位置，
                # 生成整片垃圾直到 MAX_SANE_BARS；split 改返回 [] 后若不显式
                # 退出，remaining 永不减少会原地死循环。丢弃该元素剩余时值。
                log_message(
                    f'[jianpu] append_element_chunks: 小节容量耗尽仍剩 {remaining:.3f} 拍，'
                    '已丢弃该元素剩余时值（疑似弱起边界异常）。',
                    _logging.WARNING,
                )
                return
            for piece in split_duration_chunks(piece_total):
                element_piece = clone_monophonic_element(element_template, piece)
                current_measure.append(note_to_jianpu(element_piece, key_tonic_semitone, tonic_step=tonic_step))
                current_pos += piece
            remaining -= piece_total
            # 判断当前位置是否恰好是小节边界
            if _first_pickup_ql:
                at_bar_end = (
                    (not measures and abs(current_pos - _first_pickup_ql) < tol)
                    or (measures and abs((current_pos - _first_pickup_ql) % bar_length) < tol)
                )
            else:
                at_bar_end = abs(current_pos % bar_length) < tol
            if at_bar_end:
                measures.append(current_measure)
                current_measure = []

    for idx, offset in enumerate(offsets):
        if len(measures) >= MAX_SANE_BARS:
            break
        if offset > current_pos + tol:
            gap = offset - current_pos
            # 跳过异常大的间隙（可能是 OMR 输出中的宫位异常）而不是用休止符充填
            if gap > bar_length * MAX_SANE_BARS:
                current_pos = offset
            else:
                rest = m21note.Rest()
                append_element_chunks(rest, gap)

        next_offset = offsets[idx + 1] if idx + 1 < len(offsets) else offset + float(by_offset[offset].duration.quarterLength or 0.25)  # type: ignore[union-attr]
        # 单个音符时寄不得超过 MAX_SANE_BARS 个小节（防止 OMR 输出中的超长音符）
        duration = max(min(next_offset - offset, bar_length * MAX_SANE_BARS), 0.125)
        append_element_chunks(by_offset[offset], duration)

    if current_measure:
        if _first_pickup_ql:
            if not measures:
                pos_in_bar = current_pos  # still in pickup bar
            else:
                pos_in_bar = (current_pos - _first_pickup_ql) % bar_length
        else:
            pos_in_bar = current_pos % bar_length
        if pos_in_bar > tol:
            rest = m21note.Rest()
            append_element_chunks(rest, bar_length - pos_in_bar)
        elif current_measure:
            measures.append(current_measure)

    repaired_measures = MeasureBarUnits(
        repair_jianpu_measure(m, (_first_pickup_ql if ri == 0 and _first_pickup_ql else bar_length))
        for ri, m in enumerate(measures)
    )
    # This path re-slices the whole part at a single bar_length, so every measure
    # it emits has that length; recording it keeps the text builder from padding
    # against a different value.
    repaired_measures.bar_units = [
        int(round((_first_pickup_ql if ri == 0 and _first_pickup_ql else bar_length) * 16))
        for ri in range(len(repaired_measures))
    ]
    repaired_measures.time_signatures = [time_signature.split(',')[0]] * len(repaired_measures)
    return repaired_measures, time_signature
