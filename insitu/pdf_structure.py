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

Engraving scale
---------------
A source can be engraved at any size; the bundled 梆笛 score uses a 5.13pt
staff space while the same music re-engraved at 72% has 3.72pt.  Every
tolerance below is a distance on the page, so a tolerance tuned for one size
silently rejects the other: with a point-valued barline test the 3.72pt score
yielded *zero* staves and the in-situ PDF was an empty copy of the source.

All scale-dependent values therefore live in `StaffMetrics`, derived from the
measured `Staff.line_spacing`.  The reference engraving (`REFERENCE_SPACING`)
is the 梆笛 score; multipliers are its calibration values expressed per staff
space, following the same convention as `core/omr/vector_*.py`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pymupdf as fitz

# The staff space of the engraving every multiplier below was calibrated on.
REFERENCE_SPACING = 5.13

# Accepted staff space range.  A piano-style brace, a beam cluster or a row of
# ledger lines also produces horizontal rules, but their spacing falls outside
# this band.  The 3.72pt source sits comfortably inside it; the old 4.0pt floor
# was an artefact of calibrating on the 5.13pt score only.
_STAFF_LINE_SPACING_MIN = 2.0
_STAFF_LINE_SPACING_MAX = 6.5
# Two horizontal rules belong to the same staff when they sit no more than this
# far apart.  Must stay above the largest accepted staff space (so one staff is
# never split) and below the distance between systems (always > 50pt).
_STAFF_LINE_MAX_GAP = _STAFF_LINE_SPACING_MAX * 1.35
# A rule has to be at least this long to be a staff line rather than a ledger
# line, a beam, or a barline fragment.  This is a page-scale measure, not a
# staff-scale one, so it stays in points.
_MIN_STAFF_LINE_LEN = 200.0
# Same line spacing across all rules of one staff (used to validate 5 lines).
_MIN_RULES_PER_STAFF = 5

# MuseScore's music fonts encode digits in the private use area; Maestro is
# MuseScore 3, Opus is MuseScore 4.  Only digit runs inside the multi-measure
# rest-count band are used, so time signatures and note-adjacent numbers are
# ignored regardless of the font.
_PUA_DIGIT = {chr(0xF030 + i): str(i) for i in range(10)}
_MUSIC_FONT_PREFIXES = ("Maestro", "Opus")
# Measure numbers use Times New Roman Italic in both MuseScore 3 and 4 exports.
_MEASURE_NUMBER_FONT = "TimesNewRomanPS-Italic"


@dataclass(frozen=True)
class StaffMetrics:
    """Every staff-scale-dependent tolerance, derived from one staff space.

    Multipliers are written relative to the staff space (`s`).  The comment on
    each field gives the value measured on the reference 5.13pt engraving.
    """
    line_spacing: float

    # Barline endpoints.  A barline is drawn from exactly the top rule to the
    # bottom rule plus a small overshoot; sub-point rounding is proportional to
    # the engraving scale.
    barline_top_tol: float        # 0.20pt
    barline_tol: float            # 0.60pt
    barline_overshoot: float      # 0.24pt
    barline_x_margin: float       # 2.0pt  beyond the staff's x-extent
    barline_merge: float          # 5.5pt  thin+thick repeat pair is one barline

    # Printed multi-measure-rest count above the staff.  Time signatures sit
    # 14-25pt above the reference staff, rest counts 33-40pt.
    rest_count_top: float         # 33.0pt
    rest_count_bottom: float      # 40.0pt
    rest_inset_left: float        # 4.0pt  keep the count inside its own box
    rest_inset_right: float       # 2.0pt

    # Measure-number ruler in the left margin.
    measure_number_top: float     # 60.0pt above the staff top
    measure_number_margin: float  # 2.0pt  around the staff's left edge

    # System-opening repeat bar: a thick stroke just right of the clef.
    repeat_stroke_max_width: float   # 1.2pt
    repeat_stroke_min_height: float  # 8.0pt
    repeat_top_tol: float            # 1.0pt (stroke starts on the top rule)
    repeat_margin: float             # 45.0pt from the staff's left edge
    repeat_merge: float              # 8.0pt  near the first barline

    # Space above the first staff / below the last one that belongs to page
    # furniture (title block, page number) rather than to the music.
    page_furniture_margin: float  # 38.0pt

    @classmethod
    def from_spacing(cls, line_spacing: float) -> "StaffMetrics":
        """Derive the tolerances for a staff with the given line spacing."""
        s = line_spacing if line_spacing and line_spacing > 0 else REFERENCE_SPACING
        return cls(
            line_spacing=s,
            barline_top_tol=0.04 * s,
            barline_tol=0.12 * s,
            barline_overshoot=0.047 * s,
            barline_x_margin=0.4 * s,
            barline_merge=1.07 * s,
            rest_count_top=6.43 * s,
            rest_count_bottom=7.80 * s,
            rest_inset_left=0.78 * s,
            rest_inset_right=0.39 * s,
            measure_number_top=11.7 * s,
            measure_number_margin=max(1.0, 0.4 * s),
            repeat_stroke_max_width=0.23 * s,
            repeat_stroke_min_height=1.56 * s,
            repeat_top_tol=0.20 * s,
            repeat_margin=8.8 * s,
            repeat_merge=1.56 * s,
            page_furniture_margin=7.4 * s,
        )


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

    @property
    def metrics(self) -> StaffMetrics:
        """Tolerances scaled to this staff's own engraving size."""
        return StaffMetrics.from_spacing(self.line_spacing)


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
    # A long annotation can lie on top of a staff rule. Treat coincident
    # strokes as one rule, retaining the full-width staff line.
    unique: list[tuple[float, float, float]] = []
    for line in horizontal:
        if unique and abs(line[0] - unique[-1][0]) < 0.15:
            if line[2] - line[1] > unique[-1][2] - unique[-1][1]:
                unique[-1] = line
        else:
            unique.append(line)
    horizontal = unique

    staves: list[Staff] = []
    run: list[tuple[float, float, float]] = []

    def flush():
        if len(run) < _MIN_RULES_PER_STAFF:
            return
        ys = [r[0] for r in run]
        spacing = (ys[-1] - ys[0]) / (len(ys) - 1)
        if not _STAFF_LINE_SPACING_MIN <= spacing <= _STAFF_LINE_SPACING_MAX:
            return                         # a brace or a beam cluster
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
    metrics = staff.metrics
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
        # _BARLINE_OVERSHOOT (0.24pt at the reference scale).  Verifying both
        # ends is what separates a barline from a note stem: stems also reach
        # from above the staff to below it, but they never land on that exact
        # pair of offsets.  The old "covers the whole staff" test counted 306
        # barlines where the engraving has 168 (stems, ledger-line strokes and
        # repeat-barline pairs each added phantom measure boxes), so every
        # measure count downstream was wrong.
        if abs(top - staff.top) > metrics.barline_top_tol:
            continue                        # does not start on the top rule
        if abs(bottom - (staff.bottom + metrics.barline_overshoot)) > metrics.barline_tol:
            continue                        # does not end where a barline ends
        if not (staff.x0 - metrics.barline_x_margin
                <= x0 <= staff.x1 + metrics.barline_x_margin):
            continue                        # outside this staff's extent
        xs.append(x0)

    xs.sort()
    # Collapse near-duplicates.  A repeat barline is drawn as a thin line and a
    # thick one ~3-4pt apart at the reference scale; at 1.5pt that pair
    # survived as two barlines and invented a zero-width measure box (the
    # "5 boxes in a 3-measure system" mismatches against MuseScore's
    # measure-number ruler).  Real barlines are >40pt apart, so the merged
    # threshold is safely below the closest genuine spacing.
    merged: list[float] = []
    for x in xs:
        if merged and x - merged[-1] < metrics.barline_merge:
            continue
        merged.append(x)
    # A system-opening repeat bar follows the clef but is not a measure.
    thick_repeat = [d['rect'].x0 for d in page.get_drawings()
                    if d['type'] == 's' and (d.get('width') or 0) >= 1.5
                    and d['rect'].width < metrics.repeat_stroke_max_width
                    and d['rect'].height > metrics.repeat_stroke_min_height
                    and abs(d['rect'].y0 - staff.top) < metrics.repeat_top_tol]
    if (len(merged) > 1 and merged[0] - staff.x0 < metrics.repeat_margin
            and any(abs(x - merged[0]) < metrics.repeat_merge for x in thick_repeat)):
        merged.pop(0)
    return merged


def _music_digits(page) -> list[tuple[float, float, int]]:
    """Return (x0, y0, value) for every whole-number digit run in a music font.

    MuseScore prints measure numbers and multi-measure-rest counts in the
    private-use area of its music font (U+F030+digit in Maestro), which
    PyMuPDF reads back as plain text with no OCR involved.  MuseScore 4 uses
    Opus instead.  The caller restricts these to the rest-count band, so
    digits elsewhere (time signatures, note-adjacent numbers) do no harm.
    """
    out: list[tuple[float, float, int]] = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                if not span["font"].startswith(_MUSIC_FONT_PREFIXES):
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
    whole piece.  The printed count sits 33-40pt above the reference staff's
    top rule, while time signatures and other digits sit 14-25pt above, so the
    window is unambiguous (measured over all three source scores).
    """
    metrics = staff.metrics
    digits = _music_digits(page)
    edges = [staff.x0, *barlines]
    counts: list[int] = []
    for index in range(len(barlines)):
        x0, x1 = edges[index], edges[index + 1]
        hits = [
            value for dx, dy, value in digits
            if metrics.rest_count_top <= staff.top - dy <= metrics.rest_count_bottom
            and x0 + metrics.rest_inset_left <= dx <= x1 - metrics.rest_inset_right
        ]
        counts.append(max(hits) if hits else 1)
    return counts


def measure_numbers(page, staff: Staff) -> Optional[int]:
    """The measure number MuseScore prints in the left margin for *staff*.

    This is ground truth for the measure-number ruler: it lets us check that
    consecutive staves' spans abut (`start + count == next start`) instead of
    trusting the box count alone.  The number is Times New Roman Italic at the
    staff's left edge (indented first systems included), 5-50pt above the
    staff's top rule.
    """
    metrics = staff.metrics
    best: Optional[int] = None
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                if not span["font"].startswith(_MEASURE_NUMBER_FONT):
                    continue
                x0, y0, _, y1 = span["bbox"]
                text = span["text"].strip()
                if not text.isdigit():
                    continue
                if abs(x0 - staff.x0) > metrics.measure_number_margin:
                    continue                 # not the left-margin ruler
                if 0.0 <= staff.top - y1 <= metrics.measure_number_top:
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