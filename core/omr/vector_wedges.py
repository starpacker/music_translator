"""Conservatively match isolated vector hairpins to MusicXML note boundaries."""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import pymupdf
from music21 import converter

from insitu.align import align_from_files
from .multirest_collapse import page_system_slot_kinds
from .vector_tempos import _matched_multirest_x, _matched_note_events, _page_marks, _read_root


def _two_stroke_hairpin(drawing, staff, *, closed_limit=.35):
    """Classify a single open V-shaped path below the staff, or reject it."""
    rect = drawing['rect']
    width = drawing.get('width') or 0
    if (drawing['type'] != 's' or not (.1 <= width <= 1.2)
            or len(drawing['items']) != 2 or not all(item[0] == 'l' for item in drawing['items'])
            or not (8 <= rect.width <= staff.x1 - staff.x0) or not (.2 <= rect.height <= 12)
            or not (staff.bottom + 1 <= rect.y0 and rect.y1 <= staff.bottom + 40)
            or rect.x0 < staff.x0 or rect.x1 > staff.x1):
        return None
    lines = []
    for _, a, b in drawing['items']:
        left, right = (a, b) if a.x <= b.x else (b, a)
        lines.append((left, right))
    (a, b), (c, d) = lines
    if abs(a.x - c.x) > .8 or abs(b.x - d.x) > .8:
        return None
    left_gap, right_gap = abs(a.y - c.y), abs(b.y - d.y)
    if abs((a.y + c.y) / 2 - (b.y + d.y) / 2) > .5 * staff.line_spacing:
        return None
    if left_gap <= closed_limit * staff.line_spacing and .8 * staff.line_spacing <= right_gap <= 2.2 * staff.line_spacing:
        kind = 'crescendo'
    elif right_gap <= closed_limit * staff.line_spacing and .8 * staff.line_spacing <= left_gap <= 2.2 * staff.line_spacing:
        kind = 'diminuendo'
    else:
        return None
    return float(rect.x0), float(rect.x1), float((rect.y0 + rect.y1) / 2), kind


def _split_stroke_hairpins(drawings, staff):
    """Detect split-line V shapes for review, without assuming continuity."""
    singles = [d for d in drawings if d['type'] == 's' and len(d['items']) == 1
               and d['items'][0][0] == 'l' and .1 <= (d.get('width') or 0) <= 1.2
               and d['rect'].width >= 8 and staff.x0 <= d['rect'].x0 < d['rect'].x1 <= staff.x1
               and staff.bottom + 1 <= d['rect'].y0 and d['rect'].y1 <= staff.bottom + 40]
    result = []
    for i, first in enumerate(singles):
        for second in singles[i + 1:]:
            if abs(first['rect'].x0 - second['rect'].x0) > .8 or abs(first['rect'].x1 - second['rect'].x1) > .8:
                continue
            merged = {'type': 's', 'width': max(first.get('width') or 0, second.get('width') or 0),
                      'rect': first['rect'] | second['rect'], 'items': [first['items'][0], second['items'][0]]}
            shape = _two_stroke_hairpin(merged, staff, closed_limit=.6)
            if shape and shape not in result:
                result.append(shape)
    return result


def _nearest_boundary(x, positions, spacing):
    """Find a unique musical boundary, grouping noteheads and barlines."""
    distances = sorted((min(abs(x - position) for position in xs), key)
                       for key, xs in positions.items())
    if not distances or distances[0][0] > 2 * spacing:
        return None
    if len(distances) > 1 and distances[1][0] - distances[0][0] < .8 * spacing:
        return None
    return distances[0][1], distances[0][0]


def discover_pdf_wedges(pdf_path: Path, mxl_path: Path):
    score = converter.parse(str(mxl_path))
    if len(score.parts) != 1:
        return [], [{'reason': 'multiple_parts'}]
    measures = list(score.parts[0].getElementsByClass('Measure'))
    staves, slices = align_from_files(pdf_path, mxl_path)
    candidates, report = [], []
    with pymupdf.open(pdf_path) as doc:
        for page_number, page in enumerate(doc, 1):
            _, heads, numbers = _page_marks(page)
            kinds = page_system_slot_kinds(page)
            rows = [(layout, sl) for layout, sl in zip(staves, slices) if layout.staff.page == page_number]
            drawings = page.get_drawings()
            for row, (layout, sl) in enumerate(rows):
                staff = layout.staff
                shapes = [shape for drawing in drawings if (shape := _two_stroke_hairpin(drawing, staff))]
                for x0, x1, y, kind in _split_stroke_hairpins(drawings, staff):
                    report.append({'page': page_number, 'staff_index': sl.staff_index + 1,
                                   'x0': round(x0, 2), 'x1': round(x1, 2), 'y': round(y, 2),
                                   'kind': kind, 'reason': 'split_strokes_or_possible_continuation'})
                if not shapes:
                    continue
                edges = [staff.x0, *layout.barlines]
                slots = []
                positions = {}
                for i, (left, right) in enumerate(zip(edges, edges[1:])):
                    index = sl.omr_start + i
                    if index >= len(measures) or index >= sl.omr_end:
                        slots.append(False)
                        continue
                    measure = measures[index]
                    events = _matched_note_events(measure, staff, left, right, heads)
                    matched = events is not None
                    if events:
                        for xs, offset, _ in events:
                            time = Fraction(offset).limit_denominator(1000000)
                            positions.setdefault((index, time), []).extend(xs)
                    elif row < len(kinds) and i < len(kinds[row]) and kinds[row][i] == 'M':
                        matched = _matched_multirest_x(page, measure, staff, left, right, numbers) is not None
                    slots.append(matched)
                for i, edge in enumerate(edges):
                    index = sl.omr_start + i
                    # A boundary must touch at least one geometrically verified box.
                    if index < len(measures) and ((i > 0 and slots[i - 1]) or (i < len(slots) and slots[i])):
                        positions.setdefault((index, Fraction(0)), []).append(edge)
                for x0, x1, y, kind in shapes:
                    rec = {'page': page_number, 'staff_index': sl.staff_index + 1,
                           'x0': round(x0, 2), 'x1': round(x1, 2), 'y': round(y, 2), 'kind': kind}
                    if sl.omr_measures != sl.pdf_measures:
                        report.append({**rec, 'reason': 'ambiguous_measure_alignment'})
                        continue
                    # A line at a system edge may continue on the next line;
                    # do not split it into an invented independent hairpin.
                    if x0 - staff.x0 <= 2 * staff.line_spacing or staff.x1 - x1 <= 2 * staff.line_spacing:
                        report.append({**rec, 'reason': 'possible_system_continuation'})
                        continue
                    crossing = [i for i, (left, right) in enumerate(zip(edges, edges[1:]))
                                if right > x0 and left < x1]
                    if not crossing or any(i >= len(slots) or not slots[i] for i in crossing):
                        report.append({**rec, 'reason': 'unverified_measures'})
                        continue
                    start, end = _nearest_boundary(x0, positions, staff.line_spacing), _nearest_boundary(x1, positions, staff.line_spacing)
                    if start is None or end is None or start[0] >= end[0]:
                        report.append({**rec, 'reason': 'ambiguous_endpoints'})
                        continue
                    (a, ta), da = start
                    (b, tb), db = end
                    item = {**rec, 'start_measure_index': a + 1, 'start_offset_quarters': str(ta),
                            'end_measure_index': b + 1, 'end_offset_quarters': str(tb),
                            'start_distance_points': round(da, 2), 'end_distance_points': round(db, 2),
                            'evidence': 'vector_shape_and_notehead_sequence'}
                    candidates.append(item)
    # Do not insert overlapping hairpins: LilyPond has a single active wedge.
    overlap = set()
    for i, first in enumerate(candidates):
        a = (first['start_measure_index'], Fraction(first['start_offset_quarters']))
        b = (first['end_measure_index'], Fraction(first['end_offset_quarters']))
        for j, second in enumerate(candidates[i + 1:], i + 1):
            c = (second['start_measure_index'], Fraction(second['start_offset_quarters']))
            d = (second['end_measure_index'], Fraction(second['end_offset_quarters']))
            if a < d and c < b:
                overlap.update((i, j))
    for i, item in enumerate(candidates):
        report.append({**item, 'reason': 'overlapping_candidates' if i in overlap else 'verified_candidate'})
    return [item for i, item in enumerate(candidates) if i not in overlap], report


def _insertion_points(measure, ns, divisions):
    """Map exact note/rest boundaries to XML insertion indices (single voice)."""
    if not divisions or divisions <= 0:
        return {}
    children = list(measure)
    cursor = Fraction(0)
    points = {}
    timed = False
    for index, child in enumerate(children):
        if child.tag in (f'{ns}backup', f'{ns}forward'):
            return {}
        if child.tag == f'{ns}attributes' and timed and child.find(f'{ns}divisions') is not None:
            return {}
        if child.tag != f'{ns}note' or child.find(f'{ns}chord') is not None:
            continue
        points.setdefault(cursor, index)
        if child.find(f'{ns}grace') is not None:
            continue
        try:
            duration = int(child.findtext(f'{ns}duration'))
        except (ValueError, TypeError):
            return {}
        if duration <= 0:
            return {}
        cursor += Fraction(duration, divisions)
        timed = True
    # End-of-measure directions must precede the right barline, if present.
    barline = next((i for i, child in enumerate(children)
                    if child.tag == f'{ns}barline' and child.get('location', 'right') == 'right'), len(children))
    points.setdefault(cursor, barline)
    return points


def write_enriched_wedges(mxl_path: Path, output: Path, candidates):
    """Write exact note-boundary wedges to a copy, never overriding source wedges."""
    root = _read_root(mxl_path)
    ns = root.tag.split('}', 1)[0] + '}' if root.tag.startswith('{') else ''
    parts = root.findall(f'{ns}part')
    if len(parts) != 1:
        return [], [{**item, 'reason': 'multiple_parts'} for item in candidates]
    part = parts[0]
    if any(el.tag == f'{ns}wedge' for el in part.iter()):
        return [], [{**item, 'reason': 'existing_musicxml_wedges'} for item in candidates]
    measures = part.findall(f'{ns}measure')
    divisions = None
    points = []
    for measure in measures:
        initial = divisions
        timed = False
        for child in measure:
            if child.tag in (f'{ns}note', f'{ns}forward', f'{ns}backup'):
                timed = True
            if child.tag == f'{ns}attributes' and child.findtext(f'{ns}divisions'):
                try:
                    divisions = int(child.findtext(f'{ns}divisions'))
                except ValueError:
                    divisions = None
                if not timed:
                    initial = divisions
        points.append(_insertion_points(measure, ns, initial))
    boundaries, inserted, skipped = {}, [], []
    for item in candidates:
        try:
            start = (item['start_measure_index'] - 1, Fraction(item['start_offset_quarters']))
            end = (item['end_measure_index'] - 1, Fraction(item['end_offset_quarters']))
            valid = (item['kind'] in ('crescendo', 'diminuendo') and start < end
                     and all(0 <= i < len(points) and offset in points[i] for i, offset in (start, end)))
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            valid = False
        if not valid:
            skipped.append({**item, 'reason': 'unrepresentable_boundary'})
            continue
        boundaries.setdefault(start[0], []).append((points[start[0]][start[1]], 1, item['kind']))
        boundaries.setdefault(end[0], []).append((points[end[0]][end[1]], 0, 'stop'))
        inserted.append(item)
    for index, events in boundaries.items():
        measure = measures[index]
        # Reverse insertion makes earlier boundaries and stop-before-start order
        # stable even when multiple directions share the same index.
        for location, priority, kind in sorted(events, key=lambda e: (e[0], e[1]), reverse=True):
            direction = ET.Element(f'{ns}direction', {'placement': 'below'})
            dtype = ET.SubElement(direction, f'{ns}direction-type')
            ET.SubElement(dtype, f'{ns}wedge', {'type': kind, 'number': '1'})
            measure.insert(location, direction)
    if inserted:
        if output.resolve() == mxl_path.resolve():
            raise ValueError('不能覆盖原始 MusicXML')
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix='.pdf-wedges-', suffix='.musicxml', delete=False) as f:
            temporary = Path(f.name)
        try:
            ET.ElementTree(root).write(temporary, encoding='utf-8', xml_declaration=True)
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
    return inserted, skipped
