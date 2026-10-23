"""Extract engraved staff geometry and barline positions from the source PDF.

The source PDF is a MuseScore export: every staff is five horizontal rules
drawn as separate line items, and every barline is a vertical rule spanning
(and slightly overhanging) those five lines.  Both are pure vector line
segments, so PyMuPDF reads their exact coordinates back with no OCR guessing.

Geometry alone cannot tell us which staff holds the melody when several
systems appear on one page, so we also record the x-extent of each staff: the
first system on a page is indented (indent on repeat systems, space for
instrument names on the first system) while continuation systems share a
common x0.  The x0 also tells us which vertical lines really are barlines of
*this* staff rather than a neighbouring staff's (e.g. a second instrument) or a
marking such as a rehearsal letter.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pymupdf as fitz

# MuseScore's Maestro font encodes digits in the private use area.
_PUA_DIGIT = {chr(0xF030 + i): str(i) for i in range(10)}
# A printed multi-measure-rest count sits this far above the staff's top rule.
_REST_COUNT_TOP = 33.0
_REST_COUNT_BOTTOM = 40.0

# Two horizontal rules belong to the same staff when they sit no more than this
# far apart.  Real staff line spacing is 5.1-5.2pt; the next staff's first rule
# is >50pt away, so anything in between works; 7.0 leaves comfortable margin
# while still refusing to fuse two distinct systems.
_STAFF_LINE_MAX_GAP = 7.0
# A rule has to be at least this long to be a staff line rather than a ledger
# line, a beam, or a barline fragment.
_MIN_STAFF_LINE_LEN = 200.0
# Same line spacing across all rules of one staff (used to validate 5 lines).
_MIN_RULES_PER_STAFF = 5
# A vertical rule counts as a barline when it spans at least this fraction of
# the staff height, measured from the top rule down.
# Barlines are drawn from exactly the top rule to exactly the bottom rule;
# this tolerance absorbs sub-point rounding in the segment endpoints.
_BARLINE_TOL = 0.6
# MuseScore overshoots the bottom staff rule by exactly this much when stroking
# a barline.  Measured on all three source scores: 507/507 barlines start at
# offset 0.00 from the top rule and end at +0.24 from the bottom rule.
_BARLINE_OVERSHOOT = 0.24
# Two vertical strokes closer than this are one barline (repeat barlines are a
# thin+thick pair ~3-4pt apart).  Real barlines are >40pt apart.
_BARLINE_MERGE = 5.5


@dataclass
class Staff:
    """One engraved five-line staff."""
    page: int            # 1-based
    top: float           # y of the top rule
    bottom: float        # y of the bottom rule
    x0: float            # left edge of the rules
    x1: float            # right edge of the rules
    line_spacing: float  # average spacing between consecutive rules

    @property
    def height(self) -> float:
        return self.bottom - self.top

    @property
    def center(self) -> float:
        return (self.top + self.bottom) / 2.0


def _line_items(page):
    """Yield every straight line segment on the page as (x0, y0, x1, y1)."""
    for drawing in page.get_drawings():
        for item in drawing['items']:
            if item[0] == 'l':
                a, b = item[1], item[2]
                yield a.x, a.y, b.x, b.y


def extract_staves(page, page_number: int) -> list[Staff]:
    """Return the five-line staves of one PDF page, top to bottom."""
    horizontal: list[tuple[float, float, float]] = []
    for x0, y0, x1, y1 in _line_items(page):
        if abs(y0 - y1) > 0.15:
            continue                       # not horizontal
        if abs(x1 - x0) < _MIN_STAFF_LINE_LEN:
            continue                       # too short to be a staff rule
        left, right = min(x0, x1), max(x0, x1)
        if right - left < _MIN_STAFF_LINE_LEN:
            continue
        horizontal.append(((y0 + y1) / 2.0, left, right))

    horizontal.sort()

    staves: list[Staff] = []
    run: list[tuple[float, float, float]] = []

    def flush():
        if len(run) < _MIN_RULES_PER_STAFF:
            return
        ys = [r[0] for r in run]
        spacing = (ys[-1] - ys[0]) / (len(ys) - 1)
        if not 4.0 <= spacing <= 6.5:      # a piano-style brace or a beam cluster
            return
        staves.append(Staff(
            page=page_number,
            top=ys[0],
            bottom=ys[-1],
            x0=run[0][1],
            x1=run[0][2],
            line_spacing=spacing,
        ))

    for item in horizontal:
        if run and item[0] - run[-1][0] <= _STAFF_LINE_MAX_GAP:
            run.append(item)
        else:
            flush()
            run = [item]
    flush()
    return staves


def staff_barlines(page, staff: Staff) -> list[float]:
    """Return the x positions of the barlines belonging to *staff*, sorted.

    Only vertical rules that (a) span most of the staff height and (b) lie
    inside the staff's own x-extent are accepted, which keeps out the rules of
    other staves, dynamics hairpins and rehearsal marks.
    """
    # A real barline is drawn from exactly the top rule to exactly the bottom,
    # so its segment spans the full staff height.  Clef and time-signature
    # strokes reach only part of the way (or start below the top rule) and are
    # rejected by requiring the segment to cover the whole staff within a small
    # tolerance.
    # A barline is drawn from the top rule to the bottom rule and overhangs it a
    # fraction of a point; MuseScore draws it as one segment covering the whole
    # staff.  Beamed stems also run vertically but only span part of the staff
    # (a high stem starts above the staff but stops short of the bottom rule, a
    # low stem starts below the top rule), so requiring the segment to contain
    # BOTH rules inside it — with a small tolerance — excludes every stem.
    xs: list[float] = []
    for x0, y0, x1, y1 in _line_items(page):
        if abs(x0 - x1) > 0.2:
            continue                        # not vertical
        top, bottom = min(y0, y1), max(y0, y1)
        # Endpoint signature, not mere coverage.  A barline starts exactly on
        # the top rule (offset 0.00pt) and overhangs the bottom rule by exactly
        # _BARLINE_OVERSHOOT (0.24pt).  Verifying both ends is what separates a
        # barline from a note stem: stems also reach from above the staff to
        # below it, but they never land on that exact pair of offsets.  The old
        # "covers the whole staff" test counted 306 barlines where the engraving
        # has 168 (stems, ledger-line strokes and repeat-barline pairs each
        # added phantom measure boxes), so every measure count downstream was
        # wrong.
        if abs(top - staff.top) > _BARLINE_TOL:
            continue                        # does not start on the top rule
        if abs(bottom - (staff.bottom + _BARLINE_OVERSHOOT)) > _BARLINE_TOL:
            continue                        # does not end where a barline ends
        if not (staff.x0 - 2.0 <= x0 <= staff.x1 + 2.0):
            continue                        # outside this staff's extent
        xs.append(x0)

    xs.sort()
    # Collapse near-duplicates.  A repeat barline is drawn as a thin line and a
    # thick one ~3-4pt apart; at 1.5pt that pair survived as two barlines and
    # invented a zero-width measure box (the "5 boxes in a 3-measure system"
    # mismatches against MuseScore's measure-number ruler).  Real barlines are
    # >40pt apart, so 5.5pt is safely below the closest genuine spacing.
    merged: list[float] = []
    for x in xs:
        if merged and x - merged[-1] < _BARLINE_MERGE:
            continue
        merged.append(x)
    return merged


def _maestro_digits(page) -> list[tuple[float, float, int]]:
    """Return (x0, y0, value) for every whole-number Maestro digit run.

    MuseScore prints measure numbers and multi-measure-rest counts in the
    private-use area of its Maestro font (U+F030+digit), which PyMuPDF reads
    back as plain text with no OCR involved.
    """
    out: list[tuple[float, float, int]] = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                if not span["font"].startswith("Maestro"):
                    continue
                text = "".join(_PUA_DIGIT.get(ch, ch) for ch in span["text"]).strip()
                if text and text.isdigit():
                    x0, y0, _, _ = span["bbox"]
                    out.append((x0, y0, int(text)))
    return out


def box_measure_counts(page, staff: Staff, barlines: list[float]) -> list[int]:
    """Real measure count of each engraved box on *staff*.

    A multi-measure rest is engraved as ONE box with the collapsed count
    printed above it.  Reading that count is what reconciles the engraved boxes
    with the real measure count: 曲笛 has 165 boxes but 314 real measures, and
    without the printed counts the two sides of the alignment disagree for the
    whole piece.  The printed count sits 33-40pt above the staff's top rule,
    while time signatures and other digits sit 14-25pt above, so the window is
    unambiguous (measured over all three source scores).
    """
    digits = _maestro_digits(page)
    edges = [staff.x0, *barlines]
    counts: list[int] = []
    for index in range(len(barlines)):
        x0, x1 = edges[index], edges[index + 1]
        hits = [
            value for dx, dy, value in digits
            if _REST_COUNT_TOP <= staff.top - dy <= _REST_COUNT_BOTTOM
            and x0 + 4.0 <= dx <= x1 - 2.0
        ]
        counts.append(max(hits) if hits else 1)
    return counts


def measure_numbers(page, staff: Staff) -> Optional[int]:
    """The measure number MuseScore prints in the left margin for *staff*.

    This is ground truth for the measure-number ruler: it lets us check that
    consecutive staves' spans abut (`start + count == next start`) instead of
    trusting the box count alone.  The number is Times New Roman Italic at
    x < 50, 5-50pt above the staff's top rule.
    """
    best: Optional[int] = None
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                if not span["font"].startswith("TimesNewRomanPS-Italic"):
                    continue
                x0, y0, _, y1 = span["bbox"]
                text = span["text"].strip()
                if x0 > 50 or not text.isdigit():
                    continue
                if 0.0 <= staff.top - y1 <= 60.0:
                    best = int(text)
    return best


def document_staves(pdf_path) -> list[Staff]:
    """Return every staff in the document, in reading order (page, then top-down)."""
    doc = fitz.open(str(pdf_path))
    try:
        out: list[Staff] = []
        for index in range(doc.page_count):
            out.extend(extract_staves(doc[index], index + 1))
        return out
    finally:
        doc.close()


@dataclass
class StaffLayout:
    staff: Staff
    barlines: list[float] = field(default_factory=list)
    # Real measure count of each engraved box, read off the printed
    # multi-measure-rest count.  1 for a normal box, N for a compressed rest.
    box_counts: list[int] = field(default_factory=list)

    @property
    def spans(self) -> list[tuple[float, float]]:
        """Consecutive barline pairs — one per engraved measure."""
        return list(zip(self.barlines, self.barlines[1:]))

    @property
    def boxes(self) -> list[tuple[float, float, int]]:
        """One entry per engraved box: (left, right, real_measure_count)."""
        edges = [self.staff.x0, *self.barlines]
        counts = self.box_counts or [1] * len(self.barlines)
        return [
            (edges[i], edges[i + 1], counts[i] if i < len(counts) else 1)
            for i in range(len(self.barlines))
        ]

    @property
    def measure_count(self) -> int:
        """Real measures under this staff, multi-measure rests expanded."""
        return sum(c for _, _, c in self.boxes)


def layout_for_document(pdf_path) -> list[StaffLayout]:
    doc = fitz.open(str(pdf_path))
    try:
        out: list[StaffLayout] = []
        for index in range(doc.page_count):
            page = doc[index]
            for staff in extract_staves(page, index + 1):
                barlines = staff_barlines(page, staff)
                out.append(StaffLayout(staff, barlines,
                                       box_measure_counts(page, staff, barlines)))
        return out
    finally:
        doc.close()
