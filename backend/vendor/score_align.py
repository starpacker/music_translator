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


@dataclass(frozen=True)
class ExtraCellPlacement:
    cell_index: int
    x_pt: float
    scale: float


@dataclass(frozen=True)
class Config:
    render_scale: float = 4.0
    insert_height_pt: float = 29.0
    jianpu_first_left_pt: float = 80.0
    jianpu_other_left_pt: float = 29.5
    page_margin_pt: float = 26.0
    jianpu_gap_pt: float = 6.0
    system_label_clearance_pt: float = 31.5
    final_system_search_pt: float = 46.0
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


def detect_jianpu_cells(document: pymupdf.Document, config: Config) -> list[JianpuCell]:
    """Crop one jianpu cell per visible vertical divider, in reading order."""
    cells: list[JianpuCell] = []
    for page_index, page in enumerate(document):
        by_top: dict[int, list[float]] = defaultdict(list)
        for a, b in line_segments(page):
            if abs(a.x - b.x) < 0.12 and abs(a.y - b.y) > 18:
                by_top[round(min(a.y, b.y))].append((a.x + b.x) / 2)

        rows: list[tuple[int, list[float]]] = []
        for y, xs in sorted(by_top.items()):
            clusters = cluster_sorted(xs, tolerance=5.5)
            rows.append((y, [sum(cluster) / len(cluster) for cluster in clusters]))

        for row_index, (y, dividers) in enumerate(rows):
            if page_index == 0 and row_index == 0:
                previous_x = config.jianpu_first_left_pt
            else:
                previous_x = config.jianpu_other_left_pt

            for x in dividers:
                if x - previous_x > 2:
                    clip = pymupdf.Rect(
                        max(0, previous_x + 0.35),
                        max(0, y - config.cell_top_pad_pt),
                        min(page.rect.width, x + 0.15),
                        min(page.rect.height, y + config.cell_bottom_pad_pt),
                    )
                    cells.append(JianpuCell(page_index=page_index, clip=clip))
                previous_x = x
    return cells


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


def place_cell(
    target: Image.Image,
    cell_image: Image.Image,
    cell: JianpuCell,
    x_center_pt: float,
    band_top_px: int,
    band_height_px: int,
    sx: float,
    sy: float,
    scale: float = 1.0,
) -> None:
    """Center a proportionally scaled transparent cell in its insertion strip."""
    natural_width = cell.clip.width * sx
    natural_height = cell.clip.height * sy
    width_px = max(1, round(natural_width * scale))
    height_px = max(1, round(natural_height * scale))
    if cell_image.size != (width_px, height_px):
        rendered = cell_image.resize((width_px, height_px), Image.Resampling.LANCZOS)
    else:
        rendered = cell_image
    x_px = round(x_center_pt * sx - width_px / 2)
    y_px = band_top_px + (band_height_px - height_px) // 2
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
    if jianpu_count < western_count:
        raise ValueError(
            f"Not enough jianpu cells: detected {jianpu_count}, but the score has {western_count}."
        )

    extra_rows = pack_extra_cells(
        cells,
        first_index=western_count,
        page_width_pt=score_document[-1].rect.width,
        config=config,
    )
    image_cache: dict[int, Image.Image] = {}
    output_document = pymupdf.open()
    cell_cursor = 0

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

        band_height_px = round(config.insert_height_pt * sy)
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
                if cell_cursor >= western_count:
                    raise RuntimeError("Internal cell assignment exceeded the staff-measure count.")
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
                    band_top,
                    band_height_px,
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
                        band_top,
                        band_height_px,
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
    print(f"Western cells: {western_count}; jianpu cells: {jianpu_count}; extra at end: {jianpu_count - western_count}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
