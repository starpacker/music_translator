"""Recover a narrow class of PDF tempo marks without modifying the source XML.

Only quarter-note metronome text at a verified measure onset is considered.
For pitched measures every engraved notehead must match the MusicXML pitch
sequence and its treble-staff position.  Multi-measure rests require both a
matching thick rest stroke and its printed count.  Ambiguity is reported.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
import tempfile
import xml.etree.ElementTree as ET
import zipfile

import pymupdf
from music21 import clef, note
from core.notation.musicxml_wedges import parse_musicxml_compatible

from insitu.align import align_from_files
from .multirest_collapse import page_system_slot_kinds


_TEMPO = re.compile(r'^\s*(?:q|♩)\s*=\s*([1-9][0-9]{0,3})\s*$', re.IGNORECASE)
_HEADS = {'': frozenset({'Ï', 'ú', 'w'}),
          'Maestro': frozenset({'\uf0cf', '\uf0fa', '\uf077'})}


def _page_marks(page):
    """Extract text positions, not their unusually tall font bounding boxes."""
    tempos, heads, numbers = [], [], []
    for block in page.get_text('rawdict')['blocks']:
        for line in block.get('lines', []):
            for span in line.get('spans', []):
                text = ''.join(c['c'] for c in span['chars']).strip()
                x, y = span['origin']
                match = _TEMPO.fullmatch(text)
                if match:
                    tempos.append((float(x), float(y), int(match.group(1))))
                if re.fullmatch(r'[0-9]+', text) and int(text) >= 2:
                    numbers.append((float(x), float(y), int(text)))
                supported = _HEADS.get(span['font'])
                if supported is not None:
                    for char in span['chars']:
                        if char['c'] in supported:
                            cx, cy = char['origin']
                            heads.append((float(cx), float(cy), float(span['size'])))
    return tempos, heads, numbers


def _matched_note_events(measure, staff, left, right, heads):
    """Match every notehead to its event and offset, or return None."""
    own_clefs = list(measure.recurse().getElementsByClass(clef.Clef))
    current = own_clefs[0] if own_clefs else measure.getContextByClass(clef.Clef)
    if type(current) is not clef.TrebleClef or getattr(current, 'octaveChange', 0):
        return None
    events = list(measure.recurse().notes)
    if not events:
        return None
    predicted = []
    sizes = []
    for event in events:
        event_pitches = getattr(event, 'pitches', ())
        if not event_pitches or any(p.octave is None or p.step not in ('C', 'D', 'E', 'F', 'G', 'A', 'B')
                                    for p in event_pitches):
            return None
        pitches = sorted(event_pitches, key=lambda p: -(p.octave * 7 + 'CDEFGAB'.index(p.step)))
        sizes.append(len(pitches))
        for pitch in pitches:
            diatonic = pitch.octave * 7 + 'CDEFGAB'.index(pitch.step)
            predicted.append(staff.bottom - (diatonic - 30) * staff.line_spacing / 2)
    if not predicted or min(predicted) < staff.top - 6 * staff.line_spacing or max(predicted) > staff.bottom + 6 * staff.line_spacing:
        return None
    observed = sorted((x, y) for x, y, size in heads
                      if left <= x < right and staff.top - 6 * staff.line_spacing <= y <= staff.bottom + 6 * staff.line_spacing
                      and 2.4 * staff.line_spacing <= size <= 4.6 * staff.line_spacing)
    if len(observed) != len(predicted):
        return None
    if any(abs(y - target) > max(.8, staff.line_spacing * .2)
           for (_, y), target in zip(observed, predicted)):
        return None
    result = []
    cursor = 0
    for event, count in zip(events, sizes):
        group = observed[cursor:cursor + count]
        if count > 1 and group[-1][0] - group[0][0] > 2 * staff.line_spacing:
            return None
        result.append((tuple(x for x, _ in group), event.getOffsetInHierarchy(measure),
                       bool(event.duration.isGrace)))
        cursor += count
    return result


def _matched_first_note_x(measure, staff, left, right, heads):
    events = _matched_note_events(measure, staff, left, right, heads)
    if not events or abs(float(events[0][1])) > 1e-7:
        # A mark on the first pitched event after a leading rest is not
        # necessarily a measure-start mark.
        return None
    return events[0][0][0], events[1][0][0] if len(events) > 1 else None


def _matched_multirest_x(page, measure, staff, left, right, numbers):
    """Require a whole rest, a thick middle-line stroke and a printed count."""
    events = list(measure.recurse().notesAndRests)
    if len(events) != 1 or not isinstance(events[0], note.Rest) or events[0].fullMeasure is not True:
        return None
    if abs(float(events[0].getOffsetInHierarchy(measure))) > 1e-7:
        return None
    middle = (staff.top + staff.bottom) / 2
    strokes = [d['rect'] for d in page.get_drawings()
               if d['type'] == 's' and (d.get('width') or 0) >= 1.5
               and d['rect'].height <= 1 and d['rect'].width >= 2 * staff.line_spacing
               and abs(d['rect'].y0 - middle) <= 1
               and left <= d['rect'].x0 < d['rect'].x1 <= right]
    if len(strokes) != 1:
        return None
    stroke = strokes[0]
    counts = [n for x, y, n in numbers if stroke.x0 <= x <= stroke.x1
              and staff.top - 3 * staff.line_spacing <= y < staff.top]
    if len(counts) != 1:
        return None
    return float(stroke.x0), None


def discover_pdf_tempos(pdf_path: Path, mxl_path: Path):
    """Return candidates and per-candidate decisions for a single-part score."""
    with pymupdf.open(pdf_path) as doc:
        page_marks = [_page_marks(page) for page in doc]
    if not any(tempos for tempos, _, _ in page_marks):
        return [], []
    score = parse_musicxml_compatible(mxl_path)
    if len(score.parts) != 1:
        return [], [{'page': page_index, 'x': round(x, 2), 'y': round(y, 2),
                     'bpm': bpm, 'reason': 'multiple_parts'}
                    for page_index, (tempos, _, _) in enumerate(page_marks, 1)
                    for x, y, bpm in tempos]
    measures = list(score.parts[0].getElementsByClass('Measure'))
    staves, slices = align_from_files(pdf_path, mxl_path)
    report, candidates = [], []
    with pymupdf.open(pdf_path) as doc:
        for page_number, page in enumerate(doc, 1):
            tempos, heads, numbers = page_marks[page_number - 1]
            if not tempos:
                continue
            rows = [(layout, sl) for layout, sl in zip(staves, slices) if layout.staff.page == page_number]
            kinds = page_system_slot_kinds(page)
            for x, y, bpm in tempos:
                rec = {'page': page_number, 'x': round(x, 2), 'y': round(y, 2), 'bpm': bpm}
                nearby = [(i, layout, sl) for i, (layout, sl) in enumerate(rows)
                          if 4 <= layout.staff.top - y <= 40]
                if len(nearby) != 1:
                    report.append({**rec, 'reason': 'ambiguous_staff'})
                    continue
                row, layout, sl = nearby[0]
                edges = [layout.staff.x0, *layout.barlines]
                slots = [i for i, (a, b) in enumerate(zip(edges, edges[1:])) if a <= x < b]
                if len(slots) != 1 or sl.omr_measures != sl.pdf_measures:
                    report.append({**rec, 'reason': 'ambiguous_measure_alignment'})
                    continue
                slot = slots[0]
                index = sl.omr_start + slot
                if index >= sl.omr_end or index >= len(measures):
                    report.append({**rec, 'reason': 'missing_measure'})
                    continue
                left, right = edges[slot:slot + 2]
                measure = measures[index]
                first = _matched_first_note_x(measure, layout.staff, left, right, heads)
                evidence = 'notehead_sequence'
                if first is None and row < len(kinds) and slot < len(kinds[row]) and kinds[row][slot] == 'M':
                    first = _matched_multirest_x(page, measure, layout.staff, left, right, numbers)
                    evidence = 'multi_measure_rest'
                    if first is not None and sl.pdf_measures > 1:
                        # Rest boxes alone cannot establish where an OMR stream
                        # starts after a missing or duplicated bar. Require a
                        # matched pitched neighbour on the same system.
                        verified_neighbour = any(
                            i != slot and sl.omr_start + i < len(measures)
                            and _matched_first_note_x(measures[sl.omr_start + i], layout.staff,
                                                      a, b, heads) is not None
                            for i, (a, b) in enumerate(zip(edges, edges[1:])))
                        if not verified_neighbour:
                            first = None
                if first is None:
                    report.append({**rec, 'measure_index': index + 1, 'reason': 'no_verified_onset'})
                    continue
                first_x, next_x = first
                # Also require the first onset to be strictly closer than any
                # later onset, so a mid-measure tempo is never moved to zero.
                if (abs(x - first_x) > 2 * layout.staff.line_spacing
                        or (next_x is not None and abs(x - next_x) <= abs(x - first_x) + .5)):
                    report.append({**rec, 'measure_index': index + 1, 'reason': 'not_at_first_onset'})
                    continue
                candidate = {**rec, 'measure_index': index + 1, 'offset': 0, 'evidence': evidence}
                candidates.append(candidate)
    # Two markers in one source measure are ambiguous even when near an onset.
    counts = Counter(item['measure_index'] for item in candidates)
    for item in candidates:
        if counts[item['measure_index']] > 1:
            report.append({**item, 'reason': 'multiple_tempos_in_measure'})
        else:
            report.append({**item, 'reason': 'verified_candidate'})
    return [item for item in candidates if counts[item['measure_index']] == 1], report


def _read_root(path: Path):
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            container = ET.fromstring(archive.read('META-INF/container.xml'))
            entry = next((el.get('full-path') for el in container.iter()
                          if el.tag.rsplit('}', 1)[-1] == 'rootfile' and el.get('full-path')), None)
            if not entry:
                raise ValueError('MXL 缺少根 MusicXML 文件')
            return ET.fromstring(archive.read(entry))
    return ET.parse(path).getroot()


def write_enriched_tempos(mxl_path: Path, output: Path, candidates):
    """Write a derived MusicXML; never overwrite explicit tempo directions."""
    root = _read_root(mxl_path)
    namespace = root.tag.split('}', 1)[0] + '}' if root.tag.startswith('{') else ''
    parts = root.findall(f'{namespace}part')
    if len(parts) != 1:
        return [], [{'reason': 'multiple_parts'}]
    measures = parts[0].findall(f'{namespace}measure')
    inserted, skipped = [], []
    for item in candidates:
        index = item['measure_index'] - 1
        if index < 0 or index >= len(measures):
            skipped.append({**item, 'reason': 'missing_measure'})
            continue
        measure = measures[index]
        existing = any(el.tag == f'{namespace}metronome' or
                       (el.tag == f'{namespace}sound' and el.get('tempo')) for el in measure.iter())
        if existing:
            skipped.append({**item, 'reason': 'existing_musicxml_tempo'})
            continue
        direction = ET.Element(f'{namespace}direction', {'placement': 'above'})
        dtype = ET.SubElement(direction, f'{namespace}direction-type')
        metro = ET.SubElement(dtype, f'{namespace}metronome')
        ET.SubElement(metro, f'{namespace}beat-unit').text = 'quarter'
        ET.SubElement(metro, f'{namespace}per-minute').text = str(item['bpm'])
        ET.SubElement(direction, f'{namespace}sound', {'tempo': str(item['bpm'])})
        # Keep existing attributes (divisions, clef, key) before the direction.
        children = list(measure)
        insertion = next((i for i, child in enumerate(children)
                          if child.tag in {f'{namespace}note', f'{namespace}forward', f'{namespace}backup'}), len(children))
        measure.insert(insertion, direction)
        inserted.append(item)
    if inserted:
        if output.resolve() == mxl_path.resolve():
            raise ValueError('不能覆盖原始 MusicXML')
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix='.pdf-tempos-', suffix='.musicxml',
                                         delete=False) as f:
            temporary = Path(f.name)
        try:
            ET.ElementTree(root).write(temporary, encoding='utf-8', xml_declaration=True)
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
    return inserted, skipped
