"""Conservative PDF notehead evidence and multi-measure rest metadata.

This is a review aid, not a recognition oracle: it does not verify accidentals,
rhythm, rests, unrecognised fonts, or the global staff/measure assignment.
It never changes the score or guesses a correction.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import pymupdf
from music21 import clef

from core.notation.musicxml_wedges import parse_musicxml_compatible
from insitu.align import align_from_files
from .vector_tempos import _matched_multirest_info, _matched_note_events, _page_marks


def audit_pdf_noteheads(pdf_path: Path, mxl_path: Path) -> dict:
    score = parse_musicxml_compatible(mxl_path)
    limits = ['supported_vector_glyphs_only', 'diatonic_staff_position_only',
              'accidentals_not_checked', 'durations_and_other_rest_symbols_not_checked',
              'system_and_measure_alignment_not_independently_verified',
              'not_a_measure_of_omr_accuracy']
    if len(score.parts) != 1:
        return {'status': 'unsupported', 'reason': 'requires_one_musicxml_part',
                'part_count': len(score.parts), 'limitations': limits, 'measures': []}
    measures = list(score.parts[0].getElementsByClass('Measure'))
    layouts, slices = align_from_files(pdf_path, mxl_path)
    if not layouts or len(layouts) != len(slices):
        return {'status': 'unsupported', 'reason': 'no_aligned_staves',
                'limitations': limits, 'measures': []}
    decisions = []
    with pymupdf.open(pdf_path) as doc:
        for page_index, page in enumerate(doc):
            _, heads, numbers = _page_marks(page)
            for layout, sl in zip(layouts, slices):
                if layout.staff.page != page_index + 1:
                    continue
                staff = layout.staff
                edges = [staff.x0, *layout.barlines]
                for slot, (left, right) in enumerate(zip(edges, edges[1:])):
                    item = {'page': page_index + 1, 'staff_index': sl.staff_index + 1,
                            'slot_index': slot + 1,
                            'bounds_points': [round(left, 2), round(staff.top, 2),
                                              round(right, 2), round(staff.bottom, 2)]}
                    index = sl.omr_start + slot
                    if sl.omr_measures != sl.pdf_measures or index >= sl.omr_end or index >= len(measures):
                        decisions.append({**item, 'status': 'ambiguous_measure_alignment'})
                        continue
                    measure = measures[index]
                    events = list(measure.recurse().notes)
                    unsupported_events = any(not getattr(event, 'pitches', ()) for event in events)
                    expected = sum(len(getattr(event, 'pitches', ())) for event in events)
                    observed = [(x, y) for x, y, size in heads
                                if left <= x < right
                                and staff.top - 6 * staff.line_spacing <= y <= staff.bottom + 6 * staff.line_spacing
                                and 2.4 * staff.line_spacing <= size <= 4.6 * staff.line_spacing]
                    item.update({'musicxml_measure_index': index + 1,
                                 'musicxml_measure_number': str(measure.number),
                                 'musicxml_pitched_heads': expected,
                                 'detected_supported_heads': len(observed)})
                    if unsupported_events:
                        item['status'] = 'unsupported_nonpitched_note'
                    elif len(measure.voices) > 1:
                        item['status'] = 'unsupported_multiple_voices'
                    elif not events and not observed:
                        info = _matched_multirest_info(page, measure, staff, left, right, numbers)
                        if info is not None:
                            item.update({'status': 'printed_multirest_count_detected',
                                         'printed_rest_measures': info[1],
                                         'count_evidence': 'unique_number_above_middle_line_stroke'})
                        else:
                            item['status'] = 'no_pitched_heads_to_compare'
                    elif events and not observed:
                        item['status'] = 'no_supported_noteheads_detected'
                    else:
                        own = list(measure.recurse().getElementsByClass(clef.Clef))
                        current = own[0] if own else measure.getContextByClass(clef.Clef)
                        if type(current) is not clef.TrebleClef or getattr(current, 'octaveChange', 0):
                            item['status'] = 'unsupported_clef'
                        elif len(observed) != expected:
                            item['status'] = 'head_count_differs_review'
                        elif _matched_note_events(measure, staff, left, right, heads) is None:
                            item['status'] = 'head_positions_or_grouping_differ_review'
                        else:
                            item['status'] = 'supported_head_positions_agree'
                    decisions.append(item)
    if not decisions:
        return {'status': 'unsupported', 'reason': 'no_measure_slots',
                'musicxml_measure_count': len(measures), 'limitations': limits, 'measures': []}
    counts = dict(Counter(item['status'] for item in decisions))
    expected_indices = set(range(1, len(measures) + 1))
    mapped = [item['musicxml_measure_index'] for item in decisions if 'musicxml_measure_index' in item]
    missing = sorted(expected_indices - set(mapped))
    duplicate = sorted(index for index, count in Counter(mapped).items() if count > 1)
    return {'status': 'partial_evidence', 'limitations': limits,
            'musicxml_measure_count': len(measures), 'mapped_measure_count': len(mapped),
            'unmapped_musicxml_indices': missing, 'duplicate_musicxml_indices': duplicate,
            'summary': counts, 'measures': decisions}
