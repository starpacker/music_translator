"""Recover a small, font-verified set of dynamic marks from vector PDFs.

Only embedded EngraverFontSet glyphs whose notehead sequence and staff position
match MusicXML are candidates.  Ambiguous markings are reported, never guessed.
"""
from __future__ import annotations

from collections import Counter
from fractions import Fraction
from pathlib import Path
import re
import tempfile
import xml.etree.ElementTree as ET

import pymupdf
from core.notation.musicxml_wedges import parse_musicxml_compatible

from insitu.align import align_from_files
from .vector_tempos import _matched_note_events, _page_marks, _read_root


# EngraverFontSet's encoded compound dynamics, confirmed against rendered
# glyphs.  Unsupported symbols and other music fonts are deliberately ignored.
_DYNAMICS = {ord('P'): 'mp', ord('F'): 'mf', ord('p'): 'p', ord('f'): 'f',
             ord('¹'): 'pp', ord('ì'): 'fff'}
_ANONYMOUS = re.compile(r'^[A-Z]{6}\+$')


def _is_engraver(font):
    name = font.name.lower()
    return name == 'engraverfontset' or name.startswith('engraverfontset ')


def _dynamic_glyphs(doc, page):
    """Resolve PDF text to a unique embedded Engraver music-font glyph."""
    fonts = []
    seen = set()
    for entry in page.get_fonts(full=True):
        xref, base_name = entry[0], entry[3]
        if xref in seen or not (_ANONYMOUS.fullmatch(base_name or '')
                                or 'engraverfontset' in (base_name or '').lower()):
            continue
        seen.add(xref)
        try:
            data = doc.extract_font(xref)[3]
            if not data:
                return []
            fonts.append(pymupdf.Font(fontbuffer=data))
        except Exception:
            # An unknown anonymous font makes character-to-font identity
            # ambiguous; do not reinterpret its text as musical dynamics.
            return []
    if not any(_is_engraver(font) for font in fonts):
        return []
    out = []
    for span in page.get_texttrace():
        name = (span.get('font') or '').lower()
        if name and not name.startswith('engraverfontset'):
            continue
        for codepoint, glyph_id, origin, bbox in span['chars']:
            code = codepoint - 0xF000 if 0xF000 <= codepoint <= 0xF0FF else codepoint
            if code not in _DYNAMICS:
                continue
            matching = [font for font in fonts if font.has_glyph(0xF000 + code) == glyph_id and glyph_id]
            if len(matching) != 1 or not _is_engraver(matching[0]):
                continue
            out.append((float(origin[0]), float(origin[1]), float(bbox[2] - bbox[0]),
                        float(span['size']), _DYNAMICS[code]))
    return out


def _closest_event(anchor, events, spacing):
    """Match to a unique time position, grouping grace notes with equal offsets."""
    groups = {}
    for xs, offset, _ in events:
        time = Fraction(offset).limit_denominator(1000000)
        groups.setdefault(time, []).extend(xs)
    distances = sorted((min(abs(anchor - x) for x in positions), time)
                       for time, positions in groups.items())
    if not distances or distances[0][0] > max(1.5, spacing):
        return None
    if len(distances) > 1 and distances[1][0] - distances[0][0] < .8 * spacing:
        return None
    return distances[0][1]


def discover_pdf_dynamics(pdf_path: Path, mxl_path: Path):
    """Return verified-onset candidates and a per-mark review report."""
    with pymupdf.open(pdf_path) as doc:
        marks_by_page = [_dynamic_glyphs(doc, page) for page in doc]
    if not any(marks_by_page):
        return [], []
    score = parse_musicxml_compatible(mxl_path)
    if len(score.parts) != 1:
        return [], [{'page': page, 'x': round(x, 2), 'y': round(y, 2), 'value': value,
                     'reason': 'multiple_parts'}
                    for page, marks in enumerate(marks_by_page, 1)
                    for x, y, _, _, value in marks]
    measures = list(score.parts[0].getElementsByClass('Measure'))
    staves, slices = align_from_files(pdf_path, mxl_path)
    report, candidates = [], []
    with pymupdf.open(pdf_path) as doc:
        for page_number, page in enumerate(doc, 1):
            marks = marks_by_page[page_number - 1]
            if not marks:
                continue
            _, heads, _ = _page_marks(page)
            rows = [(layout, sl) for layout, sl in zip(staves, slices) if layout.staff.page == page_number]
            for x, y, width, size, value in marks:
                # Exclude glyphs that are not below a staff; e.g. unrelated
                # font symbols elsewhere on the page.
                nearby = [(layout, sl) for layout, sl in rows
                          if layout.staff.bottom + 3 <= y <= layout.staff.bottom + 35
                          and layout.staff.x0 <= x <= layout.staff.x1
                          and 2.4 * layout.staff.line_spacing <= size <= 4.6 * layout.staff.line_spacing]
                if not nearby:
                    continue
                rec = {'page': page_number, 'x': round(x, 2), 'y': round(y, 2), 'value': value}
                if len(nearby) != 1:
                    report.append({**rec, 'reason': 'ambiguous_staff'})
                    continue
                layout, sl = nearby[0]
                edges = [layout.staff.x0, *layout.barlines]
                # Composite glyphs are wider than noteheads and centred on the
                # note.  Adjust by half the difference in glyph advance width.
                anchor = x + (width - 1.4 * layout.staff.line_spacing) / 2
                slots = [i for i, (left, right) in enumerate(zip(edges, edges[1:]))
                         if left <= anchor < right]
                if len(slots) != 1 or sl.omr_measures != sl.pdf_measures:
                    report.append({**rec, 'reason': 'ambiguous_measure_alignment'})
                    continue
                slot = slots[0]
                index = sl.omr_start + slot
                if index >= sl.omr_end or index >= len(measures):
                    report.append({**rec, 'reason': 'missing_measure'})
                    continue
                events = _matched_note_events(measures[index], layout.staff, edges[slot], edges[slot + 1], heads)
                if not events:
                    report.append({**rec, 'measure_index': index + 1, 'reason': 'unverified_noteheads'})
                    continue
                offset = _closest_event(anchor, events, layout.staff.line_spacing)
                if offset is None:
                    nearest = sorted((min(abs(anchor - position) for position in xs),
                                      str(Fraction(event_offset).limit_denominator(1000000)))
                                     for xs, event_offset, _ in events)
                    report.append({**rec, 'measure_index': index + 1, 'anchor_x': round(anchor, 2),
                                   'nearest_onsets': [{'offset_quarters': event_offset,
                                                       'distance_points': round(distance, 2)}
                                                      for distance, event_offset in nearest[:3]],
                                   'reason': 'no_unique_note_onset'})
                    continue
                positions = [position for xs, event_offset, _ in events
                             if Fraction(event_offset).limit_denominator(1000000) == offset for position in xs]
                matched_x = min(positions, key=lambda position: abs(position - anchor))
                candidates.append({**rec, 'measure_index': index + 1,
                                   'offset_quarters': str(offset), 'font': 'EngraverFontSet',
                                   'anchor_x': round(anchor, 2), 'matched_notehead_x': round(matched_x, 2),
                                   'distance_points': round(abs(matched_x - anchor), 2),
                                   'evidence': 'font_and_notehead_sequence'})
    counts = Counter((item['measure_index'], item['offset_quarters']) for item in candidates)
    for item in candidates:
        reason = ('multiple_dynamics_at_onset' if counts[(item['measure_index'], item['offset_quarters'])] > 1
                  else 'verified_candidate')
        report.append({**item, 'reason': reason})
    return [item for item in candidates if counts[(item['measure_index'], item['offset_quarters'])] == 1], report


def write_enriched_dynamics(mxl_path: Path, output: Path, candidates):
    """Add directions to a derived XML at exact, representable offsets."""
    root = _read_root(mxl_path)
    ns = root.tag.split('}', 1)[0] + '}' if root.tag.startswith('{') else ''
    parts = root.findall(f'{ns}part')
    if len(parts) != 1:
        return [], [{'reason': 'multiple_parts'}]
    measures = parts[0].findall(f'{ns}measure')
    existing_measures = {index for index, measure in enumerate(measures)
                         if any(el.tag == f'{ns}dynamics' for el in measure.iter())}
    divisions_by_measure = []
    divisions = None
    for measure in measures:
        initial = divisions
        timed = changed = False
        for child in measure:
            if child.tag in {f'{ns}note', f'{ns}forward', f'{ns}backup'}:
                timed = True
            if child.tag != f'{ns}attributes':
                continue
            value = child.findtext(f'{ns}divisions')
            if value:
                try:
                    new_divisions = int(value)
                except ValueError:
                    new_divisions = None
                if timed and new_divisions != divisions:
                    changed = True
                divisions = new_divisions
                if not timed:
                    initial = divisions
        divisions_by_measure.append(None if changed else initial)
    inserted, skipped = [], []
    for item in candidates:
        index = item['measure_index'] - 1
        if index < 0 or index >= len(measures):
            skipped.append({**item, 'reason': 'missing_measure'})
            continue
        measure, div = measures[index], divisions_by_measure[index]
        if index in existing_measures:
            skipped.append({**item, 'reason': 'existing_musicxml_dynamics'})
            continue
        if item.get('value') not in _DYNAMICS.values():
            skipped.append({**item, 'reason': 'unsupported_dynamic'})
            continue
        try:
            offset = Fraction(item['offset_quarters'])
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            skipped.append({**item, 'reason': 'unrepresentable_offset'})
            continue
        ticks = offset * div if div is not None and div > 0 else None
        if ticks is None or ticks.denominator != 1 or ticks < 0:
            skipped.append({**item, 'reason': 'unrepresentable_offset'})
            continue
        direction = ET.Element(f'{ns}direction', {'placement': 'below'})
        dtype = ET.SubElement(direction, f'{ns}direction-type')
        dynamics = ET.SubElement(dtype, f'{ns}dynamics')
        ET.SubElement(dynamics, f'{ns}{item["value"]}')
        if ticks:
            ET.SubElement(direction, f'{ns}offset').text = str(int(ticks))
        children = list(measure)
        at = next((i for i, child in enumerate(children)
                   if child.tag in {f'{ns}note', f'{ns}forward', f'{ns}backup'}), len(children))
        measure.insert(at, direction)
        inserted.append(item)
    if inserted:
        if output.resolve() == mxl_path.resolve():
            raise ValueError('不能覆盖原始 MusicXML')
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix='.pdf-dynamics-', suffix='.musicxml',
                                         delete=False) as f:
            temporary = Path(f.name)
        try:
            ET.ElementTree(root).write(temporary, encoding='utf-8', xml_declaration=True)
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
    return inserted, skipped
