"""Recover narrowly corroborated early C/G key changes from vector PDFs.

Homr can place an inline key change at the beginning of the next system.
Only two unambiguous treble-staff patterns are considered, and the later
MusicXML key and both system headers must corroborate the change. This is an
inference for review; all other cases are left untouched.
"""
from __future__ import annotations

from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import pymupdf
from music21 import key

from core.notation.musicxml_wedges import parse_musicxml_compatible
from insitu.align import align_from_files
from .vector_dynamics import _ANONYMOUS, _is_engraver
from .vector_tempos import _matched_note_events, _page_marks, _read_root


def _accidentals(doc, page):
    """Resolve sharp/natural glyphs to the uniquely identified music font."""
    fonts, seen = [], set()
    for entry in page.get_fonts(full=True):
        xref, base = entry[0], entry[3]
        if xref in seen or not (_ANONYMOUS.fullmatch(base or '')
                                or 'engraverfontset' in (base or '').lower()):
            continue
        seen.add(xref)
        try:
            data = doc.extract_font(xref)[3]
            if not data:
                return None
            fonts.append(pymupdf.Font(fontbuffer=data))
        except Exception:
            return None
    if not any(_is_engraver(font) for font in fonts):
        return None
    result = []
    for span in page.get_texttrace():
        name = (span.get('font') or '').lower()
        if name and not name.startswith('engraverfontset'):
            continue
        for cp, glyph, origin, _ in span['chars']:
            code = cp - 0xF000 if 0xF000 <= cp <= 0xF0FF else cp
            if code not in (ord('#'), ord('n')):
                continue
            matching = [font for font in fonts if glyph and font.has_glyph(0xF000 + code) == glyph]
            if len(matching) == 1 and _is_engraver(matching[0]):
                result.append((chr(code), float(origin[0]), float(origin[1]), float(span['size'])))
    return result


def _f_line(mark, staff):
    return (abs(mark[2] - staff.top) <= max(.8, staff.line_spacing * .2)
            and 2.4 * staff.line_spacing <= mark[3] <= 4.6 * staff.line_spacing)


def _header_has_sharp(layout, marks):
    staff = layout.staff
    return any(mark[0] == '#' and _f_line(mark, staff)
               and staff.x0 + 2 * staff.line_spacing <= mark[1] <= staff.x0 + 7 * staff.line_spacing
               for mark in marks)


def _corroborated_sharp_header(layout, sl, marks, heads, measures):
    """Do not confuse a first-note accidental with a printed G key."""
    if sl.omr_start >= len(measures) or not layout.barlines:
        return False
    first = _first_event(measures[sl.omr_start], layout, layout.staff.x0, layout.barlines[0], heads)
    if first is None or first[1].step == 'F':
        return False
    hx, _ = first
    staff = layout.staff
    return any(mark[0] == '#' and _f_line(mark, staff)
               and staff.x0 + 2 * staff.line_spacing <= mark[1] <= staff.x0 + 7 * staff.line_spacing
               and hx - mark[1] >= 2 * staff.line_spacing for mark in marks)


def _first_event(measure, layout, left, right, heads):
    if len(measure.voices) > 1:
        return None
    pitched = list(measure.recurse().notes)
    matched = _matched_note_events(measure, layout.staff, left, right, heads)
    if not pitched or matched is None or len(matched) != len(pitched):
        return None
    xs, offset, grace = matched[0]
    pitches = list(getattr(pitched[0], 'pitches', ()))
    if grace or float(offset) != 0 or len(xs) != 1 or len(pitches) != 1:
        return None
    return xs[0], pitches[0]


def discover_pdf_key_changes(pdf_path: Path, mxl_path: Path):
    score = parse_musicxml_compatible(mxl_path)
    if len(score.parts) != 1:
        return [], [{'reason': 'multiple_parts'}]
    measures = list(score.parts[0].getElementsByClass('Measure'))
    layouts, slices = align_from_files(pdf_path, mxl_path)
    slots = {}
    for layout, sl in zip(layouts, slices):
        edges = [layout.staff.x0, *layout.barlines]
        if sl.omr_measures != sl.pdf_measures or len(edges) - 1 != sl.omr_measures:
            continue
        for j, (left, right) in enumerate(zip(edges, edges[1:])):
            slots[sl.omr_start + j] = (layout, sl, j, left, right)
    with pymupdf.open(pdf_path) as doc:
        page_data = {index + 1: (_accidentals(doc, page), _page_marks(page)[1])
                     for index, page in enumerate(doc)}
    changes = []
    previous = None
    for index, measure in enumerate(measures):
        signatures = list(measure.getElementsByClass(key.KeySignature))
        if len(signatures) != 1:
            if signatures:
                previous = None
            continue
        current = signatures[0].sharps
        if previous is not None and (previous, current) in ((0, 1), (1, 0)):
            changes.append((index, previous, current))
        previous = current
    candidates, decisions = [], []
    for later, old, new in changes:
        if later not in slots or later == 0 or later - 1 not in slots:
            continue
        next_layout, next_sl, next_slot, _, _ = slots[later]
        prev_layout, prev_sl, _, _, _ = slots[later - 1]
        if (next_slot != 0 or prev_sl.staff_index + 1 != next_sl.staff_index
                or prev_sl.omr_end != later):
            continue
        prev_marks, prev_heads = page_data[prev_layout.staff.page]
        next_marks, next_heads = page_data[next_layout.staff.page]
        if prev_marks is None or next_marks is None:
            decisions.append({'later_musicxml_measure_index': later + 1, 'old_fifths': old,
                              'new_fifths': new, 'reason': 'unsupported_pdf_music_font'})
            continue
        if _header_has_sharp(prev_layout, prev_marks) != (old == 1):
            decisions.append({'later_musicxml_measure_index': later + 1, 'old_fifths': old,
                              'new_fifths': new, 'reason': 'previous_header_not_corroborated'})
            continue
        if _header_has_sharp(next_layout, next_marks) != (new == 1):
            decisions.append({'later_musicxml_measure_index': later + 1, 'old_fifths': old,
                              'new_fifths': new, 'reason': 'next_header_not_corroborated'})
            continue
        if old == 1 and not _corroborated_sharp_header(prev_layout, prev_sl, prev_marks, prev_heads, measures):
            decisions.append({'later_musicxml_measure_index': later + 1, 'old_fifths': old,
                              'new_fifths': new, 'reason': 'ambiguous_previous_sharp_header'})
            continue
        if new == 1 and not _corroborated_sharp_header(next_layout, next_sl, next_marks, next_heads, measures):
            decisions.append({'later_musicxml_measure_index': later + 1, 'old_fifths': old,
                              'new_fifths': new, 'reason': 'ambiguous_next_sharp_header'})
            continue
        found = []
        search = [later - 1] if new == 1 else range(prev_sl.omr_start, later)
        for index in search:
            if index not in slots or list(measures[index].getElementsByClass(key.KeySignature)):
                continue
            layout, sl, _, left, right = slots[index]
            first = _first_event(measures[index], layout, left, right, prev_heads)
            if first is None:
                continue
            hx, pitch = first
            spacing = layout.staff.line_spacing
            prefix = sorted((mark for mark in prev_marks if left <= mark[1] < hx
                             and _f_line(mark, layout.staff)), key=lambda mark: mark[1])
            if new == 1:
                # Sharp on the F line at a bar start cannot be the accidental
                # of a non-F first note; the next system repeats the key.
                if (len(prefix) == 1 and prefix[0][0] == '#'
                        and prefix[0][1] - left <= 1.5 * spacing
                        and hx - prefix[0][1] <= 6 * spacing and pitch.step != 'F'):
                    found.append((index, prefix, hx))
            else:
                # A cancellation natural followed by a separate F-sharp for
                # the first note is supported by the following C-key header.
                if (len(prefix) == 2 and [mark[0] for mark in prefix] == ['n', '#']
                        and prefix[0][1] - left <= 1.5 * spacing
                        and prefix[1][1] - prefix[0][1] >= 2 * spacing
                        and 0 < hx - prefix[1][1] <= 2 * spacing
                        and pitch.step == 'F' and pitch.octave == 5
                        and pitch.accidental is not None and pitch.accidental.alter == 1):
                    found.append((index, prefix, hx))
        base = {'later_musicxml_measure_index': later + 1, 'old_fifths': old, 'new_fifths': new}
        if len(found) != 1:
            decisions.append({**base, 'reason': 'ambiguous_candidates' if found else 'no_corroborated_early_change'})
            continue
        index, prefix, hx = found[0]
        item = {**base, 'source_measure_index': index + 1,
                'page': prev_layout.staff.page, 'staff_index': prev_sl.staff_index + 1,
                'glyphs': [{'kind': a, 'x': round(x, 2), 'y': round(y, 2)} for a, x, y, _ in prefix],
                'first_note_x': round(hx, 2),
                'evidence': 'verified_font_noteheads_and_adjacent_system_headers'}
        candidates.append(item)
        decisions.append({**item, 'reason': 'candidate'})
    return candidates, decisions


def write_enriched_keys(mxl_path: Path, output: Path, candidates):
    """Move only a verified MusicXML key on a derived copy, never notes."""
    root = _read_root(mxl_path)
    ns = root.tag.split('}', 1)[0] + '}' if root.tag.startswith('{') else ''
    parts = root.findall(f'{ns}part')
    if len(parts) != 1:
        return [], [{**item, 'reason': 'multiple_parts'} for item in candidates]
    measures = parts[0].findall(f'{ns}measure')
    inserted, skipped = [], []
    for item in candidates:
        try:
            earlier = item['source_measure_index'] - 1
            later = item['later_musicxml_measure_index'] - 1
            if not (0 <= earlier < later < len(measures)):
                raise ValueError('invalid indices')
            dest, source = measures[earlier], measures[later]
            dest_attrs = [a for a in dest.findall(f'{ns}attributes')
                          if list(dest).index(a) < next((i for i, e in enumerate(dest)
                                                        if e.tag in (f'{ns}note', f'{ns}backup', f'{ns}forward')), len(dest))]
            source_keys = [(a, k) for a in source.findall(f'{ns}attributes') for k in a.findall(f'{ns}key')]
            prior_keys = [k for m in measures[:earlier] for a in m.findall(f'{ns}attributes')
                          for k in a.findall(f'{ns}key')]
            if (any(a.find(f'{ns}key') is not None for a in dest.findall(f'{ns}attributes'))
                    or len(source_keys) != 1
                    or not prior_keys or prior_keys[-1].findtext(f'{ns}fifths') != str(item['old_fifths'])
                    or list(source).index(source_keys[0][0]) >= next((i for i, e in enumerate(source)
                                if e.tag in (f'{ns}note', f'{ns}backup', f'{ns}forward')), len(source))
                    or any(a.find(f'{ns}key') is not None for m in measures[earlier + 1:later]
                           for a in m.findall(f'{ns}attributes'))
                    or source_keys[0][1].findtext(f'{ns}fifths') != str(item['new_fifths'])):
                raise ValueError('key changed or destination occupied')
        except (KeyError, TypeError, ValueError):
            skipped.append({**item, 'reason': 'unsafe_or_changed_musicxml'})
            continue
        attr, key_element = source_keys[0]
        attr.remove(key_element)
        if dest_attrs:
            dest_attr = dest_attrs[0]
        else:
            dest_attr = ET.Element(f'{ns}attributes')
            before = next((i for i, e in enumerate(dest) if e.tag != f'{ns}print'), len(dest))
            dest.insert(before, dest_attr)
        before = next((i for i, e in enumerate(dest_attr) if e.tag != f'{ns}divisions'), len(dest_attr))
        dest_attr.insert(before, key_element)
        inserted.append(item)
    if inserted:
        if output.resolve() == mxl_path.resolve():
            raise ValueError('不能覆盖原始 MusicXML')
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix='.pdf-keys-', suffix='.musicxml', delete=False) as f:
            temporary = Path(f.name)
        try:
            ET.ElementTree(root).write(temporary, encoding='utf-8', xml_declaration=True)
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
    return inserted, skipped
