"""Rebuild MusicXML slurs from the source PDF's filled crescent paths.

MuseScore engraves a slur as one filled closed path made of exactly two cubic
segments; a tie is a shallower filled shape closed with two short straight
edges.  A slur that crosses a system break is split in two: one fragment ends
at the system's right music boundary, the next starts at the following
system's left boundary.  This module detects the crescents, rejoins the
fragments, matches their endpoints to engraved noteheads and maps those
noteheads to MusicXML notes (measure + onset group + vertical rank).

The stage is deliberately conservative.  It rewrites slurs only when every
involved crescent resolves to a notehead and every involved measure's
noteheads can be paired with its MusicXML notes.  Otherwise the original
MusicXML is returned untouched and the reason is written to the report.  The
source MusicXML is never modified; the enriched copy is a separate file.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
import re
import tempfile
import xml.etree.ElementTree as ET
import zipfile

import pymupdf

from insitu.align import align_from_files


# Music-font glyphs that represent noteheads.  Rest glyphs, stems, flags,
# articulations and accidentals are deliberately excluded.
_NOTEHEAD_GLYPHS = frozenset({'œ', '˙', 'w', 'Ï', 'ú',
                              '', '', ''})
_MUSIC_FONT = re.compile(r'maestro|music|engraver|emmentaler|bravura|opus|petaluma',
                         re.IGNORECASE)

# An endpoint is considered attached to a staff boundary when it sits this
# close (pt) to the right end of the staff rules.
_RIGHT_EDGE = 1.5
# Left fragments start inside the staff indent, before the first note.
_LEFT_ZONE = 25.0
# Slur endpoints are drawn at the notehead (or at the stem for stem-up notes).
_MAX_HEAD_DX = 8.0
# Two noteheads closer than this on the x axis belong to the same chord.
_GROUP_DX = 1.0


@dataclass
class _Curve:
    page: int
    staff: int
    start: tuple[float, float]
    end: tuple[float, float]
    start_kind: str = ''
    end_kind: str = ''
    start_head: tuple[float, float] | None = None
    end_head: tuple[float, float] | None = None
    resolved: bool = True


@dataclass
class _Link:
    start_measure: int          # 0-based measure index
    start_note: int             # index of the <note> child within its measure
    stop_measure: int
    stop_note: int
    start_x: float = 0.0
    stop_x: float = 0.0
    start_pitch: str = ''
    stop_pitch: str = ''
    start_order: int = 0        # absolute note order, for slur numbering
    stop_order: int = 0


@dataclass
class _Staff:
    index: int
    layout: object
    slice: object
    heads: list[tuple[float, float]] = field(default_factory=list)
    boxes: list[list[tuple[float, float]]] = field(default_factory=list)


def _read_root(mxl_path: Path):
    if zipfile.is_zipfile(mxl_path):
        with zipfile.ZipFile(mxl_path) as archive:
            container = ET.fromstring(archive.read('META-INF/container.xml'))
            rootfile = next((entry.get('full-path') for entry in container.iter()
                             if entry.tag.rsplit('}', 1)[-1] == 'rootfile'
                             and entry.get('full-path')), None)
            if rootfile is None:
                raise ValueError(f'MXL 缺少 MusicXML 根文件: {mxl_path}')
            return ET.fromstring(archive.read(rootfile))
    return ET.parse(str(mxl_path)).getroot()


def _tag(name: str, ns: str) -> str:
    return f'{ns}{name}'


def _children(element, ns, name):
    return [item for item in element if item.tag == _tag(name, ns)]


def _first(element, ns, name):
    return next((item for item in element if item.tag == _tag(name, ns)), None)


def _is_music_font(name: str) -> bool:
    """Some engravings embed the music font without a usable name.

    MuseScore exports seen in the wild report an empty font name; the
    notehead glyph set is specific enough to accept those traces while still
    ignoring the text fonts (which never carry these code points).
    """
    return bool(_MUSIC_FONT.search(name)) or not name.strip()


def _page_heads(page):
    """Return (x, y) of every engraved notehead on one page."""
    result = []
    for trace in page.get_texttrace():
        if not _is_music_font(trace.get('font', '')):
            continue
        for char in trace['chars']:
            glyph = chr(char[0])
            if glyph not in _NOTEHEAD_GLYPHS:
                continue
            box = char[3]
            result.append(((box[0] + box[2]) / 2,
                           (box[1] + box[3]) / 2))
    return result


def _page_crescents(page):
    """Yield (start, end) of filled two-cubic shapes (slurs), not ties."""
    for drawing in page.get_drawings():
        if 'f' not in drawing.get('type', ''):
            continue
        items = drawing.get('items', ())
        if len(items) != 2 or any(item[0] != 'c' for item in items):
            continue
        first = items[0]
        start, end = first[1], first[4]
        yield (start.x, start.y), (end.x, end.y)


def _staff_distance(staff, y: float) -> float:
    return max(staff.top - y, 0.0, y - staff.bottom)


def _staff_at(staves, y: float):
    return min(staves, key=lambda staff: _staff_distance(staff, y))


def _match_head(heads, staff, x: float, y: float):
    best = None
    for hx, hy in heads:
        dx = abs(hx - x)
        if dx > _MAX_HEAD_DX:
            continue
        if _staff_distance(staff, hy) > 6 * staff.line_spacing:
            continue
        score = dx + 0.15 * abs(hy - y)
        if best is None or score < best[0]:
            best = (score, (hx, hy))
    return None if best is None else best[1]


def _classify(point, staff, heads):
    """Return ('note'|'left'|'right'|'unresolved', matched head)."""
    x, y = point
    if x >= staff.x1 - _RIGHT_EDGE:
        return 'right', None
    head = _match_head(heads, staff, x, y)
    if head is not None:
        return 'note', head
    if x <= staff.x0 + _LEFT_ZONE:
        return 'left', None
    return 'unresolved', None


def _collect_curves(pdf_path, staves):
    """Detect crescents and classify their endpoints."""
    curves = []
    doc = pymupdf.open(str(pdf_path))
    try:
        heads_by_page = {index + 1: _page_heads(page)
                         for index, page in enumerate(doc)}
        staves_by_page = defaultdict(list)
        for staff in staves:
            staves_by_page[staff.page].append(staff)
        for index, page in enumerate(doc):
            page_number = index + 1
            page_staves = staves_by_page.get(page_number, [])
            if not page_staves:
                continue
            heads = heads_by_page[page_number]
            for start, end in _page_crescents(page):
                staff = _staff_at(page_staves, (start[1] + end[1]) / 2)
                staff_index = staves.index(staff)
                curve = _Curve(page_number, staff_index, start, end)
                curve.start_kind, curve.start_head = _classify(start, staff, heads)
                curve.end_kind, curve.end_head = _classify(end, staff, heads)
                curve.resolved = ('unresolved' not in (curve.start_kind, curve.end_kind))
                curves.append(curve)
    finally:
        doc.close()
    return curves


def _join_chains(curves):
    """Join right-edge and left-edge fragments into complete slurs.

    Returns ``(chains, problems)``; each chain is a list of curves whose first
    start and last end are noteheads.
    """
    problems = []
    if not curves:
        return [], problems
    staff_count = max(curve.staff for curve in curves) + 1
    outs = defaultdict(list)
    ins = defaultdict(list)
    for curve in curves:
        if not curve.resolved:
            problems.append({'reason': 'unresolved_endpoint', 'page': curve.page,
                             'start': list(curve.start), 'end': list(curve.end)})
            continue
        if curve.end_kind == 'right':
            outs[curve.staff].append(curve)
        if curve.start_kind == 'left':
            ins[curve.staff].append(curve)

    following = {}
    for staff in range(staff_count):
        left = sorted(outs.get(staff, []), key=lambda c: c.end[1])
        right = sorted(ins.get(staff + 1, []), key=lambda c: c.start[1])
        if len(left) != len(right):
            problems.append({'reason': 'fragment_count_mismatch',
                             'staff': staff + 1,
                             'right_fragments': len(left),
                             'left_fragments': len(right)})
            continue
        for first, second in zip(left, right):
            following[id(first)] = second

    visited = set()
    chains = []
    for curve in curves:
        if not curve.resolved or curve.start_kind != 'note':
            continue
        chain = [curve]
        visited.add(id(curve))
        current = curve
        while current.end_kind == 'right':
            nxt = following.get(id(current))
            if nxt is None or id(nxt) in visited:
                problems.append({'reason': 'unpaired_fragment', 'page': current.page,
                                 'start': list(current.start), 'end': list(current.end)})
                chain = None
                break
            visited.add(id(nxt))
            chain.append(nxt)
            current = nxt
        if chain is None:
            continue
        if current.end_kind != 'note':
            problems.append({'reason': 'unresolved_chain_end', 'page': current.page,
                             'end': list(current.end)})
            continue
        chains.append(chain)

    for curve in curves:
        if id(curve) not in visited:
            problems.append({'reason': 'orphan_fragment', 'page': curve.page,
                             'start': list(curve.start), 'end': list(curve.end)})
    return chains, problems


class _MeasureIndex:
    """Map measure/onset noteheads to the <note> children of one part."""

    def __init__(self, root: ET.Element):
        self.ns = root.tag.split('}', 1)[0] + '}' if root.tag.startswith('{') else ''
        parts = _children(root, self.ns, 'part')
        if len(parts) != 1:
            raise ValueError('MusicXML 必须恰好包含一个声部')
        self.part = parts[0]
        self.measures = _children(self.part, self.ns, 'measure')
        self.multi_voice = any(
            any(item.tag == _tag(name, self.ns) for item in self.part.iter())
            for name in ('backup', 'forward'))
        voices = {item.text for item in self.part.iter()
                  if item.tag == _tag('voice', self.ns)}
        self.multi_voice = self.multi_voice or len(voices) > 1
        self.groups = []
        self.note_order = {}
        order = 0
        for measure in self.measures:
            groups, current = [], None
            notes = _children(measure, self.ns, 'note')
            for position, note in enumerate(notes):
                if _first(note, self.ns, 'rest') is not None:
                    current = None
                    order += 1
                    continue
                if _first(note, self.ns, 'chord') is not None and current is not None:
                    current.append((position, note))
                else:
                    current = [(position, note)]
                    groups.append(current)
                self.note_order[id(note)] = order
                order += 1
            self.groups.append(groups)

    def pitches(self, note):
        ns = self.ns
        pitch = _first(note, ns, 'pitch')
        if pitch is None:
            return None
        step = _first(pitch, ns, 'step')
        octave = _first(pitch, ns, 'octave')
        if step is None or octave is None or step.text not in 'CDEFGAB' or octave.text is None:
            return None
        try:
            return int(octave.text) * 7 + 'CDEFGAB'.index(step.text)
        except ValueError:
            return None


def _pdf_groups(heads):
    """Group a measure's noteheads by x; chords share one x."""
    groups = []
    for head in sorted(heads, key=lambda item: (item[0], item[1])):
        if groups and abs(head[0] - groups[-1][0][0]) <= _GROUP_DX:
            groups[-1].append(head)
        else:
            groups.append([head])
    return groups


def _target_note(index: _MeasureIndex, measure: int, heads, head):
    """Return (measure, note position) for one matched notehead, or None."""
    if not (0 <= measure < len(index.measures)):
        return None
    groups = _pdf_groups(heads)
    note_groups = index.groups[measure]
    if len(groups) != len(note_groups):
        return None
    ###########################################################
    # Locate the chord group holding this notehead.
    ###########################################################
    group_index = None
    for position, group in enumerate(groups):
        if any(abs(hx - head[0]) <= _GROUP_DX and abs(hy - head[1]) <= 0.6
               for hx, hy in group):
            group_index = position
            break
    if group_index is None:
        return None
    observed = groups[group_index]
    notes = note_groups[group_index]
    if len(observed) != len(notes):
        return None
    if len(notes) == 1:
        position, note = notes[0]
        return measure, position, note
    pitches = [index.pitches(note) for _, note in notes]
    if any(pitch is None for pitch in pitches):
        return None
    if len(set(pitches)) != len(pitches):
        return None
    ordered = sorted(zip(pitches, notes), key=lambda item: -item[0])
    observed = sorted(observed, key=lambda item: item[1])
    rank = min(range(len(observed)),
               key=lambda i: abs(observed[i][0] - head[0]) + abs(observed[i][1] - head[1]))
    position, note = ordered[rank][1]
    return measure, position, note


def _map_links(chains, staves, index: _MeasureIndex):
    """Turn curves into (start note, stop note) links, or report problems."""
    links, problems = [], []
    for chain in chains:
        first, last = chain[0], chain[-1]
        start_head, stop_head = first.start_head, last.end_head
        start_staff = staves[first.staff]
        stop_staff = staves[last.staff]
        if (start_staff.slice.omr_measures != len(start_staff.layout.barlines)
                or stop_staff.slice.omr_measures != len(stop_staff.layout.barlines)):
            problems.append({'reason': 'measure_count_mismatch',
                             'start_measure': start_staff.slice.omr_start,
                             'stop_measure': stop_staff.slice.omr_start})
            continue

        def locate(staff, head):
            edges = [staff.layout.staff.x0, *staff.layout.barlines]
            box = len(edges) - 2
            for position in range(len(edges) - 1):
                if edges[position] <= head[0] < edges[position + 1]:
                    box = position
                    break
            if head[0] < edges[0]:
                box = 0
            if not (0 <= box < len(staff.boxes)):
                return None
            measure = staff.slice.omr_start + box
            target = _target_note(index, measure, staff.boxes[box], head)
            if target is None:
                return None
            measure, position, note = target
            return measure, position, note, index.note_order.get(id(note))

        start = locate(start_staff, start_head)
        stop = locate(stop_staff, stop_head)
        if start is None or stop is None:
            problems.append({'reason': 'notehead_not_matched_to_note',
                             'page': first.page,
                             'start': list(first.start), 'end': list(last.end)})
            continue
        if start[3] is None or stop[3] is None:
            problems.append({'reason': 'note_order_unavailable',
                             'start': list(first.start), 'end': list(last.end)})
            continue
        if start[3] >= stop[3]:
            problems.append({'reason': 'slur_not_left_to_right',
                             'start': list(first.start), 'end': list(last.end)})
            continue
        links.append(_Link(start[0], start[1], stop[0], stop[1],
                           float(start_head[0]), float(stop_head[0]),
                           '', '', start[3], stop[3]))
    return links, problems


def _prepare_staves(pdf_path, mxl_path):
    layouts, slices = align_from_files(pdf_path, mxl_path)
    staves = []
    for index, (layout, slice_) in enumerate(zip(layouts, slices)):
        staves.append(_Staff(index, layout, slice_))
    doc = pymupdf.open(str(pdf_path))
    try:
        heads_by_page = {number + 1: _page_heads(page)
                         for number, page in enumerate(doc)}
    finally:
        doc.close()
    by_page = defaultdict(list)
    for staff in staves:
        by_page[staff.layout.staff.page].append(staff)
    for staff in staves:
        geometry = staff.layout.staff
        page_staves = [item.layout.staff for item in by_page[geometry.page]]
        heads = [head for head in heads_by_page.get(geometry.page, [])
                 if _staff_at(page_staves, head[1]) is geometry]
        staff.heads = heads
        edges = [geometry.x0, *staff.layout.barlines]
        boxes = [[] for _ in range(max(0, len(edges) - 1))]
        for head in heads:
            for position in range(len(edges) - 1):
                if edges[position] <= head[0] < edges[position + 1]:
                    boxes[position].append(head)
                    break
        staff.boxes = boxes
    return staves


def discover_pdf_slurs(pdf_path: Path, mxl_path: Path):
    """Return ``(links, report)`` for the slurs drawn in the PDF.

    Status is ``inferred`` when every curve resolved, ``partial`` when at
    least one complete slur was measured but some fragment did not, and
    ``unverified`` when nothing usable was found.  A single stray fragment
    (an unmatched half-curve, a tie mistaken for a slur) must not throw away
    the slurs that *did* resolve: the OMR's own slur marks are far worse —
    Homr emits ``start`` without ``stop``, so they over-extend to the end of
    the score.
    """
    pdf_path, mxl_path = Path(pdf_path), Path(mxl_path)
    report = {'source_pdf': str(pdf_path), 'source_musicxml': str(mxl_path), 'slurs': []}
    try:
        root = _read_root(mxl_path)
        index = _MeasureIndex(root)
        if index.multi_voice:
            report.update(status='unverified', reason='multiple_voices')
            return [], report
        staves = _prepare_staves(pdf_path, mxl_path)
        curves = _collect_curves(pdf_path, [staff.layout.staff for staff in staves])
        report['curves'] = len(curves)
        chains, problems = _join_chains(curves)
        links, mapping_problems = _map_links(chains, staves, index)
        problems.extend(mapping_problems)
        report['problems'] = problems
        if not chains:
            report.update(status='no_slurs')
            return [], report
        if not links:
            report.update(status='unverified', resolved_slurs=0,
                          curve_slurs=len(chains))
            return [], report
        status = 'inferred' if not problems and len(links) == len(chains) else 'partial'
        report.update(status=status, curve_slurs=len(chains),
                      resolved_slurs=len(links))
        for link in sorted(links, key=lambda item: (item.start_order, item.stop_order)):
            report['slurs'].append({
                'start_measure': link.start_measure + 1,
                'stop_measure': link.stop_measure + 1,
                'start_x': round(link.start_x, 2),
                'stop_x': round(link.stop_x, 2),
            })
        return links, report
    except Exception as exc:                      # noqa: BLE001 - reported, not raised
        report.update(status='error', reason=str(exc))
        return [], report


def _notations(note: ET.Element, ns: str) -> ET.Element:
    existing = _first(note, ns, 'notations')
    if existing is not None:
        return existing
    notations = ET.Element(_tag('notations', ns))
    position = next((i for i, item in enumerate(note)
                     if item.tag in (_tag('lyric', ns), _tag('play', ns))), len(note))
    note.insert(position, notations)
    return notations


def _add_slur(note: ET.Element, ns: str, kind: str, number: int):
    notations = _notations(note, ns)
    element = ET.Element(_tag('slur', ns), {'type': kind, 'number': str(number)})
    position = max([i + 1 for i, item in enumerate(notations)
                    if item.tag == _tag('tied', ns)] or [len(notations)])
    notations.insert(position, element)


def write_enriched_slurs(mxl_path: Path, output: Path, links) -> dict:
    """Write a derived MusicXML with every slur replaced or removed."""
    root = _read_root(mxl_path)
    ns = root.tag.split('}', 1)[0] + '}' if root.tag.startswith('{') else ''
    parts = _children(root, ns, 'part')
    if len(parts) != 1:
        raise ValueError('MusicXML 必须恰好包含一个声部')
    part = parts[0]
    removed = 0
    for note in [item for item in part.iter() if item.tag == _tag('note', ns)]:
        for notations in _children(note, ns, 'notations'):
            for slur in _children(notations, ns, 'slur'):
                notations.remove(slur)
                removed += 1
            if len(notations) == 0:
                note.remove(notations)
    measures = _children(part, ns, 'measure')

    # Assign slur numbers so that overlapping slurs never share one.
    events = []
    for link in links:
        events.append((link.start_order, 0, link))
        events.append((link.stop_order, 1, link))
    events.sort(key=lambda item: (item[0], item[1]))
    open_numbers, numbers = [], {}
    for _, kind, link in events:
        if kind == 0:
            number = next(value for value in range(1, len(links) + 2)
                          if value not in open_numbers)
            open_numbers.append(number)
            numbers[id(link)] = number
        else:
            number = numbers.get(id(link))
            if number in open_numbers:
                open_numbers.remove(number)

    added = 0
    for link in links:
        for measure, position, kind in (
                (link.start_measure, link.start_note, 'start'),
                (link.stop_measure, link.stop_note, 'stop')):
            notes = _children(measures[measure], ns, 'note')
            _add_slur(notes[position], ns, kind, numbers.get(id(link), 1))
            added += 1
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.resolve() == Path(mxl_path).resolve():
        raise ValueError('不能覆盖原始 MusicXML')
    with tempfile.NamedTemporaryFile(dir=output.parent, prefix='.pdf-slurs-',
                                     suffix='.musicxml', delete=False) as handle:
        temporary = Path(handle.name)
    try:
        ET.ElementTree(root).write(temporary, encoding='utf-8', xml_declaration=True)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return {'removed_slurs': removed, 'added_marks': added, 'slurs': len(links)}
