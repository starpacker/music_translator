"""Reflow intact PDF painting objects onto source-sized pages.

No horizontal or column clip is used to separate source systems: complete
text/path/image operations travel with their staff. This matters for slurs,
lyrics and ornaments whose bounding boxes overlap the inter-system whitespace.
The source font resources and painting programs are reused, not rasterized.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re

import pymupdf as fitz


@dataclass
class Graphic:
    program: bytes
    bounds: fitz.Rect
    anchor: float
    text: str = ''


@dataclass
class SourceRow:
    index: int
    programs: list[Graphic]
    top: float
    bottom: float
    original_top: float


def _tokens(data: bytes):
    """Yield PDF token offsets, excluding comments and string contents.

    In particular ``(q)``, escaped parentheses, hex strings and ``/Q`` are
    NOT graphics-state operators. Regex splitting corrupts music-font glyphs.
    """
    i, size = 0, len(data)
    whitespace = b'\x00\t\n\x0c\r '
    delimiters = b'()<>[]{}/%'
    while i < size:
        start, char = i, data[i]
        if char in whitespace:
            i += 1
            continue
        if char == ord('%'):
            while i < size and data[i] not in b'\r\n':
                i += 1
            continue
        if char == ord('('):
            depth = 1
            i += 1
            while i < size and depth:
                if data[i] == ord('\\'):
                    i += 2
                    continue
                if data[i] == ord('('):
                    depth += 1
                elif data[i] == ord(')'):
                    depth -= 1
                i += 1
            if depth:
                raise ValueError('unterminated PDF literal string')
        elif char == ord('<') and data[i:i + 2] != b'<<':
            end = data.find(b'>', i + 1)
            if end < 0:
                raise ValueError('unterminated PDF hex string')
            i = end + 1
        elif data[i:i + 2] in (b'<<', b'>>'):
            i += 2
        elif char == ord('/'):
            i += 1
            while i < size and data[i] not in whitespace + delimiters:
                i += 1
        elif char in delimiters:
            i += 1
        else:
            while i < size and data[i] not in whitespace + delimiters:
                i += 1
        yield start, i, data[start:i]


def _top_level(data: bytes) -> list[bytes]:
    depth, start = 0, 0
    result = []
    for left, right, token in _tokens(data):
        if token == b'q':
            if depth == 0:
                start = left
            depth += 1
        elif token == b'Q':
            depth -= 1
            if depth < 0:
                raise ValueError('unbalanced PDF graphics state')
            if depth == 0:
                result.append(data[start:right])
        elif depth == 0:
            raise ValueError('source PDF has painting outside a saved graphics state')
    if depth:
        raise ValueError('unbalanced PDF graphics state')
    return result


def _programs(data: bytes) -> list[bytes]:
    """Split MuPDF's cleaned, self-contained q…Q painting programs.

    Some producers (MuseScore) wrap the entire page in one extra q…Q, which
    would otherwise make the whole page a single indivisible object. When the
    outer level holds exactly one group, drop that wrapper and split again.
    """
    result = _top_level(data)
    while len(result) == 1:
        tokens = list(_tokens(data))
        if tokens[0][2] != b'q' or tokens[-1][2] != b'Q':
            break
        inner = data[tokens[0][1]:tokens[-1][0]]
        try:
            children = _top_level(inner)
        except ValueError:
            # This q/Q contains painting or shared state, not just a wrapper.
            break
        data, result = inner, children
    return result


_WRAPPERS = {b'q': b'Q', b'BT': b'ET', b'BDC': b'EMC', b'BMC': b'EMC'}
_STATE_SLOTS = {
    b'g': 'fill', b'rg': 'fill', b'k': 'fill', b'sc': 'fill', b'scn': 'fill', b'cs': 'fill',
    b'G': 'stroke', b'RG': 'stroke', b'K': 'stroke', b'SC': 'stroke', b'SCN': 'stroke',
    b'CS': 'stroke', b'w': 'line', b'J': 'line', b'j': 'line', b'M': 'line', b'd': 'line',
    b'ri': 'line', b'i': 'line', b'gs': 'gs', b'cm': 'cm',
    b'Tc': 'text', b'Tw': 'text', b'Tz': 'text', b'TL': 'text', b'Tr': 'text', b'Ts': 'text',
    b'Tf': 'font',
}
_TEXT_STATE_SLOTS = {'text', 'font'}
_PAINT_ENDERS = {
    b'S', b's', b'f', b'F', b'f*', b'B', b'B*', b'b', b'b*', b'n',
    b'Do', b'sh', b'BI', b'EI', b'd0', b'd1', b'Tj', b'TJ', b"'", b'"',
}
_TEXT_PAINT = {b'Tj', b'TJ', b"'", b'"'}
_TEXT_OPERATORS = (set(_STATE_SLOTS) | _PAINT_ENDERS
                   | {b'Tm', b'Td', b'TD', b'T*', b'BX', b'EX', b'q', b'Q',
                      b'BT', b'ET', b'BDC', b'BMC', b'EMC'})


def _match(events, index):
    """Index of the token closing the group opened at ``index``."""
    opener = events[index][2]
    closer = _WRAPPERS[opener]
    depth = 0
    for position in range(index, len(events)):
        token = events[position][2]
        if token == opener:
            depth += 1
        elif token == closer:
            depth -= 1
            if depth == 0:
                return position
    return None


def _statements(data: bytes):
    """Yield (is_group, span, leading_operands, operator) at nesting level 0."""
    events = list(_tokens(data))
    start, index = None, 0
    while index < len(events):
        left, right, token = events[index]
        if start is None:
            start = left
        if token in _WRAPPERS:
            end = _match(events, index)
            if end is None:
                yield False, data[start:], b'', None
                return
            yield True, data[left:events[end][1]], data[start:left], token
            index, start = end + 1, None
            continue
        if token in _STATE_SLOTS or token in _PAINT_ENDERS:
            yield False, data[start:right], b'', token
            index, start = index + 1, None
            continue
        index += 1
    if start is not None and data[start:].strip():
        yield False, data[start:], b'', None


def _render(state: dict) -> bytes:
    return b' '.join(value for value in state.values())


def _paints(body: bytes) -> bool:
    return any(token in _TEXT_PAINT for _, _, token in _tokens(body))


def _text_object(state: dict, body: bytes) -> bytes:
    state = _render(state)
    return b'BT ' + (state + b' ' if state else b'') + body + b' ET'


def _text_runs(prefix: bytes, span: bytes, text_state: dict):
    """Cut a ``BT…ET`` block at absolute text-matrix (``Tm``) boundaries.

    One text block can hold annotations for every system on the page: the
    pypdf-rewritten source does exactly that, leaving a block far taller than
    a staff that cannot travel with a single row.  ``Tm`` starts a new absolute
    position, so it is a safe cut.  Text state (font, size, spacing, render
    mode) is part of the graphics state and survives ``ET``, so it is replayed
    inside every run — including runs whose own block never sets a font.

    Returns the runs plus the text state in force after the block.
    """
    events = list(_tokens(span))
    inner = span[events[0][1]:events[-1][0]]
    current = dict(text_state)
    cuts, cursor = [], 0
    for _, right, token in _tokens(inner):
        if token not in _TEXT_OPERATORS:
            continue
        statement = inner[cursor:right].strip()
        if token in _STATE_SLOTS and _STATE_SLOTS[token] in _TEXT_STATE_SLOTS:
            current[_STATE_SLOTS[token]] = statement
        elif token == b'Tm':
            cuts.append((cursor, dict(current)))
        cursor = right
    final = dict(current)

    if not cuts:
        return [(prefix, _text_object(text_state, inner))], final

    runs = []
    head = cuts[0][0]
    if head > 0 and _paints(inner[:head]):
        runs.append((prefix, _text_object({}, inner[:head])))
    for index, (start, snapshot) in enumerate(cuts):
        end = cuts[index + 1][0] if index + 1 < len(cuts) else len(inner)
        body = inner[start:end]
        if _paints(body):
            runs.append((prefix, _text_object(snapshot, body)))
    if not runs:
        runs = [(prefix, _text_object(text_state, inner))]
    return runs, final


def _state_objects(data: bytes, state: dict, text_state: dict) -> list[tuple[bytes, bytes]]:
    """Movable objects as (prefix, body), in paint order.

    State statements (colour, line width, CTM, resource dictionary) cannot be
    painted alone; they accumulate into the prefix of every following object so
    each object can be painted in isolation.  ``state``/``text_state`` are
    mutated in place so a caller descending into a wrapper keeps them.
    """
    objects = []
    for is_group, span, lead, operator in _statements(data):
        if not is_group and operator in _STATE_SLOTS:
            slot = _STATE_SLOTS[operator]
            if slot in _TEXT_STATE_SLOTS:
                text_state[slot] = span
            elif slot == 'cm':
                # Concatenate: successive `cm` operators compose the CTM.
                state['cm'] = (state.get('cm', b'') + b' ' + span).strip()
            else:
                state[slot] = span
            continue
        prefix = _render(state)
        if lead.strip():
            prefix = (prefix + b' ' if prefix else b'') + lead.strip()
        if is_group and operator == b'BT':
            runs, final = _text_runs(prefix, span, text_state)
            text_state.clear()
            text_state.update(final)
            objects.extend(runs)
        else:
            objects.append((prefix, span))
    return objects


def _wrapper(body: bytes):
    """(opener, inner) when ``body`` is one complete ``q…Q``/``BT…ET``/``BDC…EMC``."""
    events = list(_tokens(body))
    if len(events) < 2:
        return None
    opener = events[0][2]
    if _WRAPPERS.get(opener) != events[-1][2]:
        return None
    if _match(events, 0) != len(events) - 1:
        return None
    return opener, body[events[0][1]:events[-1][0]]


def _split_objects(data: bytes) -> list[tuple[bytes, bytes]]:
    """Descend page-level ``q``/marked-content wrappers to the painting level.

    A whole page wrapped in one ``q…Q`` (and often one ``BDC…EMC``) is a single
    opaque object to ``_programs``; it would be assigned wholesale to one staff
    and exceed the printable page.  Descending keeps every inner object, and
    every object carries the accumulated prefix so it stays self-contained.
    """
    state, text_state = {}, {}
    while True:
        objects = _state_objects(data, state, text_state)
        if len(objects) != 1:
            return objects
        wrapped = _wrapper(objects[0][1])
        if wrapped is None or wrapped[0] == b'BT':
            return objects
        inner = wrapped[1]
        # Marked content is a tagging wrapper, not painting: its tag and
        # property operands go with the `BDC`/`EMC` pair and must not be
        # replayed on their own (a bare `<<…>>` before `BT` is a syntax error).
        # Graphics state accumulated so far is carried by ``state``; probe the
        # children on copies so the next pass does not compose state twice.
        probe = _state_objects(inner, dict(state), dict(text_state))
        if not probe:
            return objects
        if len(probe) > 1 or _wrapper(probe[0][1]) is not None:
            data = inner
            continue
        return objects


def _split_programs(data: bytes) -> list[bytes]:
    """Self-contained painting programs, descending a page-wide wrapper."""
    return [((prefix + b'\n') if prefix else b'') + body
            for prefix, body in _split_objects(data)]


def _bounds(log) -> fitz.Rect:
    boxes = [fitz.Rect(rect) for kind, rect in log
             if kind != 'ignore-text' and all(math.isfinite(v) for v in rect)
             and rect[2] > rect[0] and rect[3] > rect[1]]
    if not boxes:
        return fitz.Rect()
    box = fitz.Rect(boxes[0])
    for other in boxes[1:]:
        box |= other
    return box


def _distance(y: float, staff) -> float:
    return max(staff.top - y, 0.0, y - staff.bottom)


class SourceObjects:
    """An isolated, editable source page with reusable original PDF resources."""

    def __init__(self, source_doc, page_index: int, layouts):
        self.doc = fitz.open()
        self.doc.insert_pdf(source_doc, from_page=page_index, to_page=page_index)
        self.page = self.doc[0]
        # Sanitization changes fill-and-stroke text in some Finale PDFs.
        # Normalization alone retains that text and its original stroke width.
        self.page.clean_contents(sanitize=False)
        self.outlined = False
        self.page_index = page_index
        self.width, self.height = self.page.rect.width, self.page.rect.height
        self._layouts = layouts
        try:
            self.graphics = self._extract()
        except ValueError:
            self._flatten(source_doc[page_index])
        # A form XObject can contain several systems under one q/Q. Flatten
        # only these pages to vector outlines; never crop through that form.
        if any(self._contains_systems(g) for g in self.graphics) and not self.outlined:
            self._flatten(source_doc[page_index])
        if any(self._contains_systems(g) for g in self.graphics):
            raise ValueError('source contains an inseparable multi-system painting object')
        self.rows, self.header, self.footer = self._assign()

    def _flatten(self, source_page):
        svg = source_page.get_svg_image(text_as_path=True).encode('utf-8')
        with fitz.open(stream=svg, filetype='svg') as vector:
            pdf = vector.convert_to_pdf()
        self.doc.close()
        self.doc = fitz.open(stream=pdf, filetype='pdf')
        self.page = self.doc[0]
        self.page.clean_contents(sanitize=False)
        self.outlined = True
        self.graphics = self._extract()

    def _extract(self):
        data = self.page.read_contents()
        contents = self.page.get_contents()
        if len(contents) != 1:
            raise ValueError('PDF cleaning did not create one content stream')
        self.xref = contents[0]
        try:
            objects = self._describe(_programs(data))
            # A page wrapped in one q…Q (plus marked content) is a single
            # opaque object to _programs, and one BT block can span every
            # system.  Both are taller than any staff, so they would be
            # assigned wholesale to one row and overflow the printable page.
            # Descend to the painting level only when the evidence says so.
            tallest = max((g.bounds.height for g in objects if not g.bounds.is_empty),
                          default=0.0)
            if tallest > self.height * .5:
                divided = _split_programs(data)
                if len(divided) > len(objects):
                    objects = self._describe(divided)
        finally:
            self.doc.update_stream(self.xref, data)
        return objects

    def _describe(self, programs):
        """Measure each program's ink box, text and anchor on the live page."""
        objects = []
        for program in programs:
            self.doc.update_stream(self.xref, program)
            box = _bounds(self.page.get_bboxlog())
            if box.is_empty:
                # Preserve nonpainting programs too (e.g. marked content).
                # The cleaned stream makes their state self-contained.
                objects.append(Graphic(program, box, 0))
                continue
            traces = self.page.get_texttrace()
            chars = [char for trace in traces for char in trace['chars']]
            text = ''.join(chr(char[0]) for char in chars if char[0] >= 0)
            # The ink centre works for both high accents and low dynamics;
            # unlike a font ascender box it does not cover the next staff.
            anchor = (box.y0 + box.y1) / 2
            objects.append(Graphic(program, box, anchor, text))
        return objects

    def _contains_systems(self, graphic):
        if graphic.bounds.is_empty:
            return False
        # Only multi-operation forms need flattening, not long intact curves.
        if b'Do' not in graphic.program:
            return False
        return sum(graphic.bounds.y0 <= layout.staff.top
                   and graphic.bounds.y1 >= layout.staff.bottom
                   for _, layout in self._layouts) > 1

    def _assign(self):
        if not self._layouts:
            return [], self.graphics, []
        staffs = [layout.staff for _, layout in self._layouts]
        groups = [[] for _ in staffs]
        header, footer = [], []
        first, last = staffs[0], staffs[-1]
        # Keep actual page furniture, not music above/below the hidden staff.
        header_limit = first.top - max(38.0, first.line_spacing * 7)
        footer_limit = max(last.bottom + 38.0, self.height - 60.0)
        for graphic in self.graphics:
            box = graphic.bounds
            if box.is_empty:
                header.append(graphic)
            elif box.y1 < header_limit:
                header.append(graphic)
            elif box.y0 > footer_limit:
                footer.append(graphic)
            else:
                owner = min(range(len(staffs)), key=lambda i: _distance(graphic.anchor, staffs[i]))
                groups[owner].append(graphic)
        # Lyrics and plain-text annotations stay together as complete text
        # lines. Do not apply this to music fonts: their oversized line boxes
        # can merge notes from different systems.
        if not self.outlined:
            for block in self.page.get_text('dict')['blocks']:
                for line in block.get('lines', []):
                    spans = line.get('spans', [])
                    if not spans or any(re.search(r'maestro|music|engraver|emmentaler|bravura|opus|petaluma',
                                                  span['font'], re.I) for span in spans):
                        continue
                    line_box = fitz.Rect(line['bbox'])
                    matches = [g for group in groups for g in group
                               if g.text.strip() and line_box.contains(g.bounds.tl)
                               and line_box.contains(g.bounds.br)]
                    if len(matches) < 2:
                        continue
                    owner = min(range(len(staffs)),
                                key=lambda i: _distance(sum(g.anchor for g in matches) / len(matches), staffs[i]))
                    matched = {id(g) for g in matches}
                    for i, group in enumerate(groups):
                        groups[i] = [g for g in group if id(g) not in matched]
                    groups[owner].extend(matches)
        rows = []
        for (index, layout), group in zip(self._layouts, groups):
            top = min([layout.staff.top - .5] + [g.bounds.y0 for g in group if not g.bounds.is_empty])
            bottom = max([layout.staff.bottom + .5] + [g.bounds.y1 for g in group if not g.bounds.is_empty])
            rows.append(SourceRow(index, group, top, bottom, layout.staff.top))
        return rows, header, footer

    def place(self, target, graphics, shift: float = 0):
        if not graphics:
            return
        self.doc.update_stream(self.xref, b'\n'.join(g.program for g in graphics))
        # Each group has different contents, but shares immutable resources.
        # MuPDF's page-form cache otherwise replays the first group's stream.
        target.parent.ShownPages.pop((self.doc._graft_id, 0), None)
        # show_pdf_page copies the reusable font/resource dictionary. The
        # outer rectangle translates the WHOLE page; no object is sliced.
        target.show_pdf_page(fitz.Rect(0, shift, self.width, self.height + shift),
                             self.doc, 0, keep_proportion=False)

    def close(self):
        self.doc.close()


def paginate(source_doc, layouts, tiles, row_reports, output, place_tiles, *, gap=9.0):
    """Place each source staff + its jianpu together, adding equal-sized pages.

    ``tiles[index]`` entries expose ink at position 1. ``place_tiles`` receives
    (output_page, index, tiles, top) and returns the used bottom coordinate.
    Oversized multi-voice/wrapped groups may continue onto a new source-sized
    page; a source staff is never split.

    Consecutive source pages with identical dimensions flow continuously: when
    a source page ends with room left, the next source page's staffs continue
    on the same output page instead of starting a blank one.  A new output page
    is created only when the current one is full or the source page size
    changes.  The output page keeps the size (and therefore the height) of the
    source page where it started; it is never stretched.
    """
    page_map = []
    state = None

    def start_page(objects, source_number, width, height, margin, bottom_limit, *, header):
        page = output.new_page(width=width, height=height)
        entry = {'output_page': len(output), 'source_page': source_number,
                 'source_pages': [source_number],
                 'source_text_outlined': objects.outlined}
        page_map.append(entry)
        cursor = margin
        if header:
            objects.place(page, objects.header)
            header_bottom = max([0.] + [g.bounds.y1 for g in objects.header if not g.bounds.is_empty])
            cursor = max(margin, header_bottom + gap)
        objects.place(page, objects.footer)
        return {'page': page, 'entry': entry, 'cursor': cursor,
                'width': width, 'height': height, 'bottom_limit': bottom_limit,
                'previous_bottom': None, 'previous_source_bottom': 0.0,
                'previous_output_page': None, 'previous_source_page': None}

    for source_index, original in enumerate(source_doc):
        page_layouts = [(i, layout) for i, layout in enumerate(layouts)
                        if layout.staff.page == source_index + 1]
        if not page_layouts:
            # A page with no recognised staff cannot be reflowed; copy it as
            # its own output page and let the next page start fresh.
            output.insert_pdf(source_doc, from_page=source_index, to_page=source_index)
            page_map.append({'output_page': len(output), 'source_page': source_index + 1,
                             'source_pages': [source_index + 1]})
            state = None
            continue
        objects = SourceObjects(source_doc, source_index, page_layouts)
        try:
            width, height = objects.width, objects.height
            margin = min(30.0, height * .06)
            footer_top = min([height - margin] + [g.bounds.y0 for g in objects.footer if not g.bounds.is_empty])
            bottom_limit = min(height - margin, footer_top - gap)
            # Reuse the current output page only when the source page has the
            # same dimensions and brings no page header of its own (a title
            # block cannot be moved into the middle of an already-filled page).
            continues = (state is not None and state['width'] == width
                         and state['height'] == height
                         and not any(not g.bounds.is_empty for g in objects.header))
            if continues:
                state['entry']['source_pages'].append(source_index + 1)
                state['bottom_limit'] = min(state['bottom_limit'], bottom_limit)
            else:
                state = start_page(objects, source_index + 1, width, height,
                                   margin, bottom_limit, header=True)
            for row in objects.rows:
                row_tiles = tiles.get(row.index, [])
                source_height = row.bottom - row.top
                group_height = source_height + sum(t[1].height + gap for t in row_tiles)
                if state['previous_bottom'] is None:
                    top = max(state['cursor'], row.top)
                elif (state['previous_output_page'] == len(output)
                      and state['previous_source_page'] == source_index + 1):
                    # Existing whitespace can hold the new Jianpu; do not add
                    # the entire old gap AGAIN after the inserted tiles.
                    original_gap = max(0., row.top - state['previous_bottom'])
                    top = max(state['cursor'] + gap,
                              state['previous_source_bottom'] + original_gap)
                else:
                    # A new source page continues right after the previous
                    # content; its own top coordinate is meaningless here.
                    top = state['cursor'] + gap
                # Keep a source staff and its Jianpu together whenever the
                # whole group fits on a fresh page. Long wrapped/multi-voice
                # groups may continue, but never split a source object/tile.
                reserve = (group_height if group_height <= bottom_limit - margin else
                           source_height + (gap + row_tiles[0][1].height if row_tiles else 0))
                if top + reserve > state['bottom_limit']:
                    state = start_page(objects, source_index + 1, width, height,
                                       margin, bottom_limit, header=False)
                    top = state['cursor']
                if top + source_height > state['bottom_limit']:
                    raise ValueError(f'source staff {row.index + 1} is taller than its original page printable area')
                objects.place(state['page'], row.programs, top - row.top)
                report = row_reports[row.index]
                report.update(output_page=len(output), source_placement='whole_objects',
                              source_text_outlined=objects.outlined,
                              source_bounds=[row.top, row.bottom],
                              output_source_bounds=[top, top + source_height])
                state['cursor'] = top + source_height
                state['previous_source_bottom'] = top + source_height
                state['previous_output_page'] = len(output)
                state['previous_source_page'] = source_index + 1
                state['previous_bottom'] = row.bottom
                for tile in row_tiles:
                    tile_height = tile[1].height
                    if tile_height > bottom_limit - margin:
                        raise ValueError(f'Jianpu for staff {row.index + 1} is taller than the source page printable area')
                    tile_top = state['cursor'] + gap
                    if tile_top + tile_height > state['bottom_limit']:
                        state = start_page(objects, source_index + 1, width, height,
                                           margin, bottom_limit, header=False)
                        report.setdefault('continuation_pages', []).append(len(output))
                        tile_top = state['cursor']
                    state['cursor'] = place_tiles(state['page'], row.index, [tile], tile_top)
        finally:
            objects.close()
    return page_map
