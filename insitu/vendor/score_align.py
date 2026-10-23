#!/usr/bin/env python3
"""Align transparent jianpu measure crops beneath corresponding staff measures.

This script is tuned for the supplied score PDFs. It detects measure divisions
from PDF vector lines, keeps the original staff notation at its native size,
and inserts a transparent jianpu strip after each staff system.
"""
from __future__ import annotations

import argparse
import math
from collections import defaultdict
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image


@dataclass(frozen=True)
class StaffSystem:
    page_index: int
    row_index: int
    top: float
    bottom: float
    left: float
    bars: tuple[float, ...]


@dataclass(frozen=True)
class JianpuCell:
    page_index: int
    clip: pymupdf.Rect
    # y of the row's own baseline anchor: the y where the barline dividers of
    # this jianpu line start.  Ink *above* the anchor (octave dots over the
    # upper digit of a chord, accidentals) has a negative offset, ink below
    # (digits, underlines) a positive one.  Cells are pasted by aligning this
    # anchor, not by centring the crop, so a tall chord cell and a short rest
    # cell keep exactly the vertical relation they have in the reference PDF.
    anchor_y: float = 0.0


@dataclass(frozen=True)
class ExtraCellPlacement:
    cell_index: int
    x_pt: float
    scale: float


@dataclass(frozen=True)
class Config:
    render_scale: float = 4.0
    # The inserted band is no longer a fixed height.  It is measured from the
    # jianpu ink actually present: every cell is cropped to its own ink, cells
    # are pasted by aligning their row baseline, and the band is sized so that
    #   gap_above_pt + (tallest ink above the baseline)
    #   + (deepest ink below the baseline) + gap_below_pt
    # exactly fits.  That is what the "extra staff spacing + tolerance" asks
    # for: the interval between two engraved staves becomes
    #   original interval + band height,
    # so the jianpu can never run into either staff.
    gap_above_pt: float = 9.0
    gap_below_pt: float = 9.0
    # Floor for the band height, used only if no ink was measured at all.
    insert_height_pt: float = 40.0
    jianpu_first_left_pt: float = 80.0
    jianpu_other_left_pt: float = 29.5
    page_margin_pt: float = 26.0
    jianpu_gap_pt: float = 6.0
    system_label_clearance_pt: float = 31.5
    final_system_search_pt: float = 46.0
    # Vertical window around a row's baseline anchor in which ink is attributed
    # to that row.  Jianpu rows sit 61pt apart and the tallest cell measured is
    # 45pt (23.6 above / 21.7 below the anchor), so 30pt never reaches a
    # neighbouring row; it also excludes the page title / "1=C" header.
    row_band_pt: float = 30.0
    # Small slack added around a cell's measured ink so antialiasing and the
    # outermost stroke width are not clipped.
    cell_pad_pt: float = 0.8
    cell_top_pad_pt: float = 2.5
    cell_bottom_pad_pt: float = 21.5


def line_segments(page: pymupdf.Page):
    """Yield line endpoints from a page's vector drawings."""
    for drawing in page.get_drawings():
        for item in drawing["items"]:
            if item[0] == "l":
                yield item[1], item[2]


def cluster_sorted(values: list[float], tolerance: float) -> list[list[float]]:
    """Cluster nearby sorted scalar values, retaining each cluster's members."""
    clusters: list[list[float]] = []
    for value in sorted(values):
        if not clusters or value - clusters[-1][-1] > tolerance:
            clusters.append([value])
        else:
            clusters[-1].append(value)
    return clusters


def detect_staff_systems(document: pymupdf.Document) -> list[StaffSystem]:
    """Find five-line staff systems and count only full-height bar separators.

    The strict endpoint checks intentionally reject note stems that do not meet
    both outer staff rules. Thresholds are tuned to the supplied vector PDF.
    """
    systems: list[StaffSystem] = []
    for page_index, page in enumerate(document):
        horizontal_rules: list[float] = []
        segments = list(line_segments(page))
        for a, b in segments:
            if abs(a.y - b.y) < 0.35 and abs(a.x - b.x) > 200:
                horizontal_rules.append(round((a.y + b.y) / 2, 2))

        rule_groups: list[list[float]] = []
        for y in sorted(set(horizontal_rules)):
            if not rule_groups or y - rule_groups[-1][-1] > 10:
                rule_groups.append([y])
            else:
                rule_groups[-1].append(y)

        for row_index, rules in enumerate(rule_groups):
            top, bottom = rules[0], rules[-1]
            left_edges: list[float] = []
            raw_bars: list[float] = []
            for a, b in segments:
                if abs(a.y - b.y) < 0.35 and abs(a.x - b.x) > 200:
                    rule_y = (a.y + b.y) / 2
                    if any(abs(rule_y - y) < 0.1 for y in rules):
                        left_edges.append(min(a.x, b.x))
                elif abs(a.x - b.x) < 0.4:
                    y0, y1 = sorted((a.y, b.y))
                    # Barlines touch the top staff rule and extend 0.24 pt
                    # beyond the bottom rule in these source PDFs.
                    if abs(y0 - top) < 0.10 and abs((y1 - bottom) - 0.24) < 0.10:
                        raw_bars.append(round(a.x, 2))

            bars = tuple(
                sum(cluster) / len(cluster)
                for cluster in cluster_sorted(raw_bars, tolerance=5.5)
            )
            systems.append(
                StaffSystem(
                    page_index=page_index,
                    row_index=row_index,
                    top=top,
                    bottom=bottom,
                    left=min(left_edges) if left_edges else 36.0,
                    bars=bars,
                )
            )
    return systems


def _ink_boxes(page: pymupdf.Page) -> list[tuple[float, float, float, float]]:
    """Every inked rectangle on the page: glyph boxes and vector strokes.

    Tall thin vertical strokes are dropped — they are the barline dividers that
    delimit the cells, and are accounted for separately so that a cell's crop
    always reaches the full height of its barlines while its *ink* measurement
    is not inflated by a neighbour's divider.
    """
    boxes: list[tuple[float, float, float, float]] = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                if not span["text"].strip() and not span["font"].startswith("Emmentaler"):
                    continue
                x0, y0, x1, y1 = span["bbox"]
                # The measure number printed in the left margin ("137", "2")
                # is set in C059 and sits left of the first cell.  It is not
                # jianpu ink: including it made the first cell of a line look
                # 11pt taller than it is and inflated the whole band.
                if (span["font"].startswith("C059")
                        and span["text"].strip().isdigit()
                        and x1 < 60.0):
                    continue
                boxes.append((x0, y0, x1, y1))
    for drawing in page.get_drawings():
        rect = drawing["rect"]
        if rect.width < 0.6 and rect.height > 12.0:
            continue                      # a cell divider, handled by the row
        boxes.append((rect.x0, rect.y0, rect.x1, rect.y1))
    return boxes


def detect_jianpu_cells(document: pymupdf.Document, config: Config) -> list[JianpuCell]:
    """Crop one jianpu cell per visible vertical divider, in reading order.

    Each cell is cropped to its **own ink**, not to a fixed-height strip.  A
    chord with octave dots stacked above the upper digit is ~45pt tall while a
    plain rest measure is ~20pt, so a single fixed pad either clips the chord
    (the dots and the upper digit vanish) or leaves the rest cell surrounded by
    empty band.  Cropping to the measured ink and remembering the row's baseline
    anchor lets the pasting step reproduce the reference PDF's vertical relations
    exactly, and lets the band be sized from the real height.
    """
    cells: list[JianpuCell] = []
    for page_index, page in enumerate(document):
        segments = list(line_segments(page))
        dividers_raw: list[tuple[float, float, float]] = []      # (top_y, bottom_y, x)
        for a, b in segments:
            if abs(a.x - b.x) < 0.12 and abs(a.y - b.y) > 18:
                dividers_raw.append((min(a.y, b.y), max(a.y, b.y), (a.x + b.x) / 2))
        if not dividers_raw:
            continue

        # One row per distinct divider top.  The tops are exact PDF coordinates
        # (66.31, 127.45, ...), so they must be clustered without rounding to an
        # integer - the anchor is used to align pasted cells and 0.3pt shows.
        rows: list[tuple[float, list[float], float]] = []
        for cluster in cluster_sorted([d[0] for d in dividers_raw], tolerance=0.6):
            y = sum(cluster) / len(cluster)
            members = [
                d for d in dividers_raw if abs(d[0] - y) <= 0.6
            ]
            xs = [sum(c) / len(c) for c in cluster_sorted([m[2] for m in members], tolerance=5.5)]
            divider_bottom = max(m[1] for m in members)
            rows.append((y, xs, divider_bottom))

        ink = _ink_boxes(page)
        band = config.row_band_pt

        for row_index, (y, dividers, divider_bottom) in enumerate(rows):
            # Window with which ink is attributed to this row: never reach into
            # the neighbouring row, and (first row) never reach up into the
            # page title or the "1=C" header.
            lo = y - band if row_index == 0 else (rows[row_index - 1][0] + y) / 2.0
            hi = y + band if row_index == len(rows) - 1 else (y + rows[row_index + 1][0]) / 2.0

            if page_index == 0 and row_index == 0:
                previous_x = config.jianpu_first_left_pt
            else:
                previous_x = config.jianpu_other_left_pt

            for x in dividers:
                if x - previous_x > 2:
                    x0 = max(0.0, previous_x + 0.35)
                    x1 = min(page.rect.width, x + 0.15)
                    # Ink whose centre falls inside this cell's column.
                    own = [
                        (bx0, by0, bx1, by1)
                        for bx0, by0, bx1, by1 in ink
                        if x0 <= (bx0 + bx1) / 2 <= x1 and lo < (by0 + by1) / 2 < hi
                    ]
                    ys = [b[1] for b in own] + [b[3] for b in own]
                    top = min(ys) if ys else y
                    bottom = max(ys) if ys else y + 19.9
                    # The cell's own barlines always run the full row height.
                    top = min(top, y)
                    bottom = max(bottom, divider_bottom)
                    # A fermata / slur / accidental can reach a few points past
                    # the barline the cell ends at.  Widening the crop to that
                    # ink (capped at the neighbour's halfway point so two cells
                    # can never claim the same glyph) keeps it from being cut.
                    left_x, right_x = x0, x1
                    for bx0, by0, bx1, by1 in own:
                        left_x = min(left_x, bx0 - config.cell_pad_pt)
                        right_x = max(right_x, bx1 + config.cell_pad_pt)
                    neighbours = [d for d in dividers if d < x0 - 1 or d > x1 + 1]
                    if neighbours:
                        left_lim = max([d for d in neighbours if d < x0] or [0.0])
                        right_lim = min([d for d in neighbours if d > x1] or [page.rect.width])
                        left_x = max(left_x, (left_lim + x0) / 2.0)
                        right_x = min(right_x, (x1 + right_lim) / 2.0)
                    clip = pymupdf.Rect(
                        left_x,
                        max(0.0, top - config.cell_pad_pt),
                        right_x,
                        min(page.rect.height, bottom + config.cell_pad_pt),
                    )
                    cells.append(JianpuCell(page_index=page_index, clip=clip,
                                            anchor_y=float(y)))
                previous_x = x
    return cells


def cell_has_notes(
    cell_index: int,
    cells: list[JianpuCell],
    document: pymupdf.Document,
) -> bool:
    """Return True when a jianpu cell carries any sounding digit (not just rests).

    The jianpu renderer prints ``0`` for every rest and ``1``–``7`` (with ``#``/
    ``b`` accidentals) for sounding notes, so a cell is a rest measure exactly
    when it contains a ``0`` and no ``1``–``7`` digit.  This is read from the
    cell's text layer, which is exactly what the standalone jianpu PDF carries.
    """
    text = document[cells[cell_index].page_index].get_text("text", clip=cells[cell_index].clip)
    return any(ch in "1234567" for ch in text)


_PUA_DIGIT = {chr(0xF030 + i): str(i) for i in range(10)}


def _decode_pua(text: str) -> str:
    return "".join(_PUA_DIGIT.get(ch, ch) for ch in text)


def system_measure_counts(
    document: pymupdf.Document,
    systems: list[StaffSystem],
) -> list[int]:
    """Read MuseScore's multi-measure-rest counts off the engraved PDF.

    Every box the engraver drew is one *engraved* measure, but a multi-measure
    rest collapses several real measures into one box; MuseScore prints the
    collapsed count above that box (in the private-use Maestro digit font, which
    PyMuPDF reads back as ``\\uf030 + digit``).  The returned list is indexed the
    same way as the flattened box sequence (staff by staff, box by box): 1 for an
    ordinary box, or the printed count when the box is a multi-measure rest.
    """
    # Collect every Maestro digit run with its bounding box, per page.
    digits: list[tuple[int, float, float, int]] = []
    for page_index, page in enumerate(document):
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    if not span["font"].startswith("Maestro"):
                        continue
                    decoded = _decode_pua(span["text"])
                    if decoded and all(ch.isdigit() for ch in decoded):
                        box = span["bbox"]
                        digits.append((page_index, box[0], box[1], int(decoded)))
    # A count is printed just above the staff, inside the staff's x-extent.
    counts: list[int] = []
    for system in systems:
        boundaries = [system.left, *system.bars]
        for index in range(len(system.bars)):
            x0, x1 = boundaries[index], boundaries[index + 1]
            hits = [
                value
                for page_index, dx, dy, value in digits
                if page_index == system.page_index
                and x0 + 4 <= dx <= x1 - 2
                and system.top - 44 <= dy <= system.top - 5
            ]
            counts.append(max(hits) if hits else 1)
    return counts


def longest_zero_run(values: np.ndarray) -> tuple[int, int] | None:
    """Return the longest inclusive range of zero-valued entries."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(values):
        if value == 0 and start is None:
            start = index
        elif value != 0 and start is not None:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, len(values) - 1))
    if not runs:
        return None
    return max(runs, key=lambda run: (run[1] - run[0] + 1, run[1]))


def choose_insertion_cut(
    alpha: np.ndarray,
    sy: float,
    x_start_px: int,
    x_end_px: int,
    low_pt: float,
    high_pt: float,
) -> int:
    """Choose a blank horizontal scanline after a system for the new strip."""
    low = max(0, int(math.ceil(low_pt * sy)))
    high = min(alpha.shape[0] - 1, int(math.floor(high_pt * sy)))
    if high <= low:
        return max(0, min(alpha.shape[0] - 1, round((low_pt + high_pt) * 0.5 * sy)))

    row_ink = np.count_nonzero(alpha[low : high + 1, x_start_px:x_end_px], axis=1)
    blank_run = longest_zero_run(row_ink)
    if blank_run is not None:
        return low + (blank_run[0] + blank_run[1]) // 2

    # A few source gaps contain a tiny isolated mark at every scanline.
    # Use the least-ink row, preferring one near the centre of the search range.
    min_ink = int(row_ink.min())
    least_ink_rows = np.flatnonzero(row_ink == min_ink)
    middle = (high - low) // 2
    return low + int(min(least_ink_rows, key=lambda index: abs(int(index) - middle)))


def transparent_cell_image(
    cell_index: int,
    cells: list[JianpuCell],
    document: pymupdf.Document,
    config: Config,
    cache: dict[int, Image.Image],
) -> Image.Image:
    """Render a cell to RGBA and make near-white paper transparent."""
    if cell_index not in cache:
        cell = cells[cell_index]
        pixmap = document[cell.page_index].get_pixmap(
            matrix=pymupdf.Matrix(config.render_scale, config.render_scale),
            clip=cell.clip,
            alpha=True,
        )
        image = Image.frombytes("RGBA", (pixmap.width, pixmap.height), pixmap.samples)
        pixels = np.asarray(image).copy()
        near_white = np.all(pixels[:, :, :3] > 245, axis=2) & (pixels[:, :, 3] > 0)
        pixels[near_white] = (0, 0, 0, 0)
        cache[cell_index] = Image.fromarray(pixels, mode="RGBA")
    return cache[cell_index]


def pack_extra_cells(
    cells: list[JianpuCell],
    first_index: int,
    page_width_pt: float,
    config: Config,
) -> list[list[ExtraCellPlacement]]:
    """Pack unpaired trailing cells into one or more natural-size rows."""
    max_width = page_width_pt - 2 * config.page_margin_pt
    rows: list[list[ExtraCellPlacement]] = []
    current: list[ExtraCellPlacement] = []
    x = config.page_margin_pt

    for index in range(first_index, len(cells)):
        natural_width = cells[index].clip.width
        scale = min(1.0, max_width / natural_width) if natural_width > 0 else 1.0
        width = natural_width * scale
        gap = config.jianpu_gap_pt if current else 0.0
        if current and x + gap + width > page_width_pt - config.page_margin_pt:
            rows.append(current)
            current = []
            x = config.page_margin_pt
            gap = 0.0
        x += gap
        current.append(ExtraCellPlacement(index, x, scale))
        x += width
    if current:
        rows.append(current)
    return rows


def cell_vertical_extents(cells: list[JianpuCell]) -> tuple[float, float]:
    """How far the jianpu ink reaches above / below each row's baseline anchor.

    Returns the worst case over the whole document: a chord cell with octave
    dots stacked on its upper digit reaches ~24pt up, and the underlines of the
    lowest octave reach ~22pt down.  Sizing the band from these measured values
    (instead of a fixed 29/40pt) is what guarantees no more overlap.
    """
    above = max((c.anchor_y - c.clip.y0 for c in cells), default=23.6)
    below = max((c.clip.y1 - c.anchor_y for c in cells), default=21.7)
    return above, below


def place_cell(
    target: Image.Image,
    cell_image: Image.Image,
    cell: JianpuCell,
    x_center_pt: float,
    baseline_px: float,
    sx: float,
    sy: float,
    scale: float = 1.0,
) -> None:
    """Paste a proportionally scaled transparent cell, aligning its baseline.

    The cell is placed so that the y of its row's baseline anchor lands on
    ``baseline_px``.  Aligning the baseline instead of centring the crop is what
    keeps a tall chord cell and a short rest cell in the same vertical relation
    they have in the reference jianpu PDF — centring would push the chord's
    octave dots upward, straight into the staff above.
    """
    natural_width = cell.clip.width * sx
    natural_height = cell.clip.height * sy
    width_px = max(1, round(natural_width * scale))
    height_px = max(1, round(natural_height * scale))
    if cell_image.size != (width_px, height_px):
        rendered = cell_image.resize((width_px, height_px), Image.Resampling.LANCZOS)
    else:
        rendered = cell_image
    x_px = round(x_center_pt * sx - width_px / 2)
    above_px = (cell.anchor_y - cell.clip.y0) * sy * scale
    y_px = round(baseline_px - above_px)
    x_px = max(0, min(target.width - width_px, x_px))
    y_px = max(0, min(target.height - height_px, y_px))
    target.alpha_composite(rendered, (x_px, y_px))


def build_output(
    score_path: Path,
    jianpu_path: Path,
    output_path: Path,
    config: Config,
) -> tuple[int, int]:
    """Create the extended-page PDF and return western/jianpu cell counts."""
    score_document = pymupdf.open(score_path)
    jianpu_document = pymupdf.open(jianpu_path)
    systems = detect_staff_systems(score_document)
    cells = detect_jianpu_cells(jianpu_document, config)
    western_count = sum(len(system.bars) for system in systems)
    jianpu_count = len(cells)

    if western_count == 0:
        raise ValueError("No staff measures were detected. Check the score PDF and detection settings.")
    # Either side may have extra cells; the surplus is simply ignored rather
    # than aborting: if the score has more measures than the jianpu, the extra
    # measures at the end get no jianpu strip; if the jianpu has more cells,
    # pack_extra_cells() below appends them after the last measure.

    extra_rows = pack_extra_cells(
        cells,
        first_index=western_count,
        page_width_pt=score_document[-1].rect.width,
        config=config,
    )
    image_cache: dict[int, Image.Image] = {}
    output_document = pymupdf.open()
    cell_cursor = 0

    # Band geometry, measured from the jianpu ink itself:
    #     gap_above + above_the_anchor + below_the_anchor + gap_below
    # This is the requested rule — the interval between two engraved staves
    # becomes (original interval) + (jianpu height) + tolerance — instead of the
    # fixed 29/40pt that clipped chord cells and let their octave dots run into
    # the staff above.
    ink_above_pt, ink_below_pt = cell_vertical_extents(cells)
    band_height_pt = max(
        config.insert_height_pt,
        ink_above_pt + ink_below_pt + config.gap_above_pt + config.gap_below_pt,
    )
    # Where the row baseline sits inside the band.
    baseline_in_band_pt = config.gap_above_pt + ink_above_pt

    for page_index, source_page in enumerate(score_document):
        page_systems = [system for system in systems if system.page_index == page_index]
        pixmap = source_page.get_pixmap(
            matrix=pymupdf.Matrix(config.render_scale, config.render_scale),
            alpha=True,
        )
        original = Image.frombytes("RGBA", (pixmap.width, pixmap.height), pixmap.samples)
        alpha = np.asarray(original)[:, :, 3]
        sx = pixmap.width / source_page.rect.width
        sy = pixmap.height / source_page.rect.height
        search_x0 = max(0, int(config.page_margin_pt * sx))
        search_x1 = min(pixmap.width, int((source_page.rect.width - config.page_margin_pt) * sx))

        cuts: list[int] = []
        for row_index, system in enumerate(page_systems):
            if row_index + 1 < len(page_systems):
                low_pt = system.bottom + 2.0
                high_pt = page_systems[row_index + 1].top - config.system_label_clearance_pt
            else:
                low_pt = system.bottom + 2.0
                high_pt = min(
                    source_page.rect.height - 24.0,
                    system.bottom + config.final_system_search_pt,
                )
            cuts.append(
                choose_insertion_cut(alpha, sy, search_x0, search_x1, low_pt, high_pt)
            )

        band_height_px = max(1, round(band_height_pt * sy))
        baseline_in_band_px = round(baseline_in_band_pt * sy)
        page_extra_rows = extra_rows if page_index == len(score_document) - 1 else []
        band_count = len(page_systems) + len(page_extra_rows)
        output_height_px = pixmap.height + band_count * band_height_px
        score_layer = Image.new("RGBA", (pixmap.width, output_height_px), (0, 0, 0, 0))
        jianpu_layer = Image.new("RGBA", (pixmap.width, output_height_px), (0, 0, 0, 0))

        source_y = 0
        destination_y = 0
        for system, cut in zip(page_systems, cuts):
            cut = max(source_y, min(pixmap.height - 1, cut))
            segment = original.crop((0, source_y, pixmap.width, cut))
            score_layer.alpha_composite(segment, (0, destination_y))
            destination_y += cut - source_y
            band_top = destination_y

            measure_count = len(system.bars)
            boundaries = [system.left, *system.bars]
            centers = [
                (boundaries[index] + boundaries[index + 1]) / 2
                for index in range(measure_count)
            ]
            for measure_index, center in enumerate(centers):
                # The jianpu may run short of the score's measures.  When it
                # does, the remaining measures stay blank — nothing is placed
                # under them — rather than aborting the whole document.
                if cell_cursor >= jianpu_count:
                    break
                cell = cells[cell_cursor]
                cell_image = transparent_cell_image(
                    cell_cursor, cells, jianpu_document, config, image_cache
                )
                neighbor_gaps_px: list[float] = []
                if measure_index > 0:
                    neighbor_gaps_px.append((center - centers[measure_index - 1]) * sx)
                if measure_index + 1 < len(centers):
                    neighbor_gaps_px.append((centers[measure_index + 1] - center) * sx)
                fit_scale = 1.0
                natural_width_px = cell.clip.width * sx
                if neighbor_gaps_px and natural_width_px > 0:
                    fit_scale = min(1.0, 0.90 * min(neighbor_gaps_px) / natural_width_px)
                place_cell(
                    jianpu_layer,
                    cell_image,
                    cell,
                    center,
                    band_top + baseline_in_band_px,
                    sx,
                    sy,
                    fit_scale,
                )
                cell_cursor += 1

            destination_y += band_height_px
            source_y = cut

        # Place unpaired jianpu cells after all matched cells, before the source footer.
        if page_extra_rows:
            for row_index, row in enumerate(page_extra_rows):
                band_top = destination_y + row_index * band_height_px
                for placement in row:
                    cell = cells[placement.cell_index]
                    image = transparent_cell_image(
                        placement.cell_index, cells, jianpu_document, config, image_cache
                    )
                    natural_width_pt = cell.clip.width * placement.scale
                    center_pt = placement.x_pt + natural_width_pt / 2
                    place_cell(
                        jianpu_layer,
                        image,
                        cell,
                        center_pt,
                        band_top + baseline_in_band_px,
                        sx,
                        sy,
                        placement.scale,
                    )
            cell_cursor = jianpu_count
            destination_y += len(page_extra_rows) * band_height_px

        tail = original.crop((0, source_y, pixmap.width, pixmap.height))
        score_layer.alpha_composite(tail, (0, destination_y))

        final_page_image = Image.new(
            "RGBA", (pixmap.width, output_height_px), (255, 255, 255, 255)
        )
        # Transparent jianpu first; the unchanged staff artwork is composited on top.
        final_page_image.alpha_composite(jianpu_layer)
        final_page_image.alpha_composite(score_layer)
        image_stream = BytesIO()
        final_page_image.convert("RGB").save(image_stream, format="PNG", optimize=True)

        output_height_pt = output_height_px / sy
        output_page = output_document.new_page(
            width=source_page.rect.width,
            height=output_height_pt,
        )
        output_page.insert_image(output_page.rect, stream=image_stream.getvalue())

        cut_points = [round(cut / sy, 2) for cut in cuts]
        print(
            f"Page {page_index + 1}: {len(page_systems)} systems, "
            f"cuts={cut_points}, output height={output_height_pt:.1f} pt"
        )

    if cell_cursor != jianpu_count:
        raise RuntimeError(f"Placed {cell_cursor} of {jianpu_count} jianpu cells.")

    output_document.set_metadata({"title": "梆笛：按竖线逐格对应（加宽行距）"})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_document.save(output_path, garbage=4, deflate=True)
    score_document.close()
    jianpu_document.close()
    output_document.close()
    return western_count, jianpu_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将简谱格按竖线顺序放到五线谱系统下方，并加长页面以扩展行距。"
    )
    parser.add_argument("--score", required=True, type=Path, help="五线谱 PDF")
    parser.add_argument("--jianpu", required=True, type=Path, help="简谱 PDF")
    parser.add_argument("--output", required=True, type=Path, help="输出 PDF")
    parser.add_argument("--scale", type=float, default=4.0, help="渲染倍率；4 约为 288 dpi")
    parser.add_argument("--insert-height", type=float, default=29.0, help="每条插入带高度（pt）")
    parser.add_argument("--page-margin", type=float, default=26.0, help="页面左右留白（pt）")
    parser.add_argument("--jianpu-first-left", type=float, default=80.0, help="简谱首行首格左界（pt）")
    parser.add_argument("--jianpu-other-left", type=float, default=29.5, help="简谱其余行首格左界（pt）")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config(
        render_scale=args.scale,
        insert_height_pt=args.insert_height,
        page_margin_pt=args.page_margin,
        jianpu_first_left_pt=args.jianpu_first_left,
        jianpu_other_left_pt=args.jianpu_other_left,
    )
    western_count, jianpu_count = build_output(args.score, args.jianpu, args.output, config)
    surplus = jianpu_count - western_count
    print(f"Western cells: {western_count}; jianpu cells: {jianpu_count}; "
          + (f"extra at end: {surplus}" if surplus >= 0 else f"unpaired measures at end: {-surplus}"))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
