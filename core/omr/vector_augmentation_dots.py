"""Conservative positive evidence for printed augmentation dots.

A dot must be a uniquely identified music-font glyph to the right of a
matched notehead and in the expected staff space. Absence of a recognized dot
is not proof that none is printed; rests and other fonts are out of scope.
"""
from __future__ import annotations

from .vector_note_values import verified_music_glyphs


def verified_dot_glyphs(doc, page):
    return verified_music_glyphs(doc, page, {ord('.')})


def compare_augmentation_dots(measure, staff, left, right, matched, noteheads, dots):
    events = list(measure.recurse().notes)
    if matched is None or len(events) != len(matched) or staff.line_spacing <= 0:
        return {'status': 'unavailable', 'reason': 'notehead_positions_not_matched'}
    spacing = staff.line_spacing
    heads = [g for g in noteheads if left <= g['x'] < right
             and staff.top - 6 * spacing <= g['y'] <= staff.bottom + 6 * spacing
             and 2.4 * spacing <= g['size'] <= 4.6 * spacing]
    event_heads = []
    missing_dotted = 0
    for index, (event, (xs, offset, grace)) in enumerate(zip(events, matched)):
        selected = [head for head in heads if any(abs(head['x'] - x) < .04 for x in xs)]
        if len(selected) != len(xs):
            if event.duration.dots:
                missing_dotted += len(xs)
            continue
        for head in selected:
            step_float = (staff.bottom - head['y']) / (spacing / 2)
            step = round(step_float)
            if abs(step_float - step) > .35:
                if event.duration.dots:
                    missing_dotted += 1
                continue
            # Standard augmentation dots sit in the note's space, or in the
            # space immediately above a note on a line.
            expected_y = head['y'] + .1 * spacing - (spacing / 2 if step % 2 == 0 else 0)
            event_heads.append({'event_index': index, 'offset': offset, 'grace': grace,
                                'head': head, 'expected_y': expected_y, 'dots': []})
    seen, eligible_dots = set(), []
    for dot in dots:
        if not (left <= dot['x'] < right and staff.top - 7 * spacing <= dot['y'] <= staff.bottom + 7 * spacing):
            continue
        # Repeated PDF text paths at exactly the same point are one dot.
        signature = (round(dot['x'], 2), round(dot['y'], 2), round(dot['size'], 2))
        if signature not in seen:
            eligible_dots.append(dot)
            seen.add(signature)
    unassigned, ambiguous = [], []
    for dot in eligible_dots:
        choices = []
        for index, item in enumerate(event_heads):
            head = item['head']
            gap = dot['x'] - head['bbox'][2]
            if (not .15 * spacing <= gap <= 2 * spacing
                    or abs(dot['size'] - head['size']) > .15 * head['size']
                    or abs(dot['y'] - item['expected_y']) > .22 * spacing):
                continue
            choices.append((abs(dot['y'] - item['expected_y']), gap, index))
        if not choices:
            unassigned.append({'x': round(dot['x'], 2), 'y': round(dot['y'], 2)})
            continue
        choices.sort()
        if (len(choices) > 1 and abs(choices[1][0] - choices[0][0]) < .2 * spacing
                and abs(choices[1][1] - choices[0][1]) < .6 * spacing):
            ambiguous.append({'x': round(dot['x'], 2), 'y': round(dot['y'], 2),
                              'candidate_event_indices': sorted({event_heads[choice[2]]['event_index'] + 1
                                                                 for choice in choices})})
            continue
        event_heads[choices[0][2]]['dots'].append(dot)
    consistent = 0
    reviews, associations = [], []
    for item in event_heads:
        expected = events[item['event_index']].duration.dots
        detected = len(item['dots'])
        if not detected:
            if expected:
                missing_dotted += 1
            continue
        detail = {'event_index': item['event_index'] + 1,
                  'offset_quarters': str(item['offset']), 'is_grace': bool(item['grace']),
                  'musicxml_dot_count': expected, 'uniquely_assigned_pdf_dots': detected,
                  'head_x': round(item['head']['x'], 2), 'head_y': round(item['head']['y'], 2),
                  'dot_points': [[round(dot['x'], 2), round(dot['y'], 2)] for dot in item['dots']]}
        if detected == expected:
            consistent += 1
            associations.append(detail)
        else:
            reviews.append({**detail, 'reason': 'detected_dot_count_differs_review'})
    return {'status': 'partial_evidence', 'detected_pdf_dot_glyphs': len(eligible_dots),
            'consistent_dotted_noteheads': consistent, 'unverified_dotted_noteheads': missing_dotted,
            'unassigned_pdf_dot_glyphs': len(unassigned), 'ambiguous_pdf_dot_glyphs': len(ambiguous),
            'unassigned_dot_points': unassigned, 'ambiguous_dot_points': ambiguous,
            'consistent_associations': associations, 'reviews': reviews}
