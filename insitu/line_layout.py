"""Split one source staff/voice into Jianpu lines and align them.

``single`` renders all measures of the staff as one unbroken, natural-width
system; ``place_line`` then compresses the spaces between symbols (and only
as a last resort the measure itself) to the source staff width, so one staff
line always gets exactly one Jianpu line.

``wrap`` keeps the original glyph size and breaks at measure boundaries, so a
staff line may receive several Jianpu lines.  A single measure that cannot
fit is a hard error: cropping, shrinking the font or widening the page would
silently lose music.
"""
from __future__ import annotations

from pathlib import Path

import pymupdf as fitz


class LineTooWideError(ValueError):
    """A single measure is wider than the available Jianpu line width."""


def _natural_layout(ly_text: str) -> str:
    """Ask LilyPond for a one-system page whose width is the system's ink.

    The marker makes ``_render_ly_tile`` skip its legacy fixed 540pt paper.
    ``one-line-breaking`` computes the paper width from the actual music, so
    wide systems are never wrapped or cropped behind our back.
    """
    return (ly_text.rstrip() + '\n% insitu-line-layout: single-system\n'
            '\\paper { page-breaking = #ly:one-line-breaking tagline = ##f }\n')


def _workdir(parent: Path, label: str) -> Path:
    path = parent / label
    path.mkdir(parents=True, exist_ok=True)
    return path


def _repeat_slice(local_repeats, start: int, end: int):
    return {offset - start: info for offset, info in local_repeats.items()
            if start <= offset < end}


def _render(render_source, render_tile, chunk, line_key, time_sig, title,
            workdir, local_repeats, label, source_chunk, start, end):
    workdir = _workdir(workdir, label)
    part = source_chunk(start, end) if source_chunk is not None else chunk[start:end]
    repeats = _repeat_slice(local_repeats, start, end)
    ly_text = render_source(part, line_key, time_sig, title, workdir, repeats)
    return render_tile(_natural_layout(ly_text), workdir)


def render_lines(chunk, line_key: str, time_sig: str, title: str, workdir: Path,
                 local_repeats, available_width: float, layout_mode: str, *,
                 render_source, render_tile, source_chunk=None,
                 tile_barlines=None):
    """Return ``[(pdf_path, ink_rect, measure_start, measure_count), ...]``.

    Measure offsets are zero-based within ``chunk``.  ``source_chunk`` may
    rebuild a sub-range with cross-line slur/tie continuation and key changes;
    without it the chunk is sliced directly.
    """
    if layout_mode not in ('single', 'wrap'):
        raise ValueError(f'unsupported insitu layout: {layout_mode}')
    if not chunk:
        return []
    workdir = Path(workdir)
    if layout_mode == 'single':
        pdf, ink = _render(render_source, render_tile, chunk, line_key, time_sig,
                           title, workdir, local_repeats, 'single', source_chunk,
                           0, len(chunk))
        return [(pdf, ink, 0, len(chunk))]

    # One unbroken natural-width system gives both the cheapest one-line
    # result and accurate measure widths for planning the wrapped lines.
    full_pdf, full_ink = _render(render_source, render_tile, chunk, line_key,
                                 time_sig, title, workdir, local_repeats, 'full',
                                 source_chunk, 0, len(chunk))
    if full_ink.width <= available_width + 0.01:
        return [(full_pdf, full_ink, 0, len(chunk))]

    count = len(chunk)
    widths, prefix, suffix = None, 0.0, 0.0
    if tile_barlines is not None:
        with fitz.open(str(full_pdf)) as doc:
            bars = tile_barlines(doc[0], full_ink, count)
        if bars is not None and len(bars) == count + 1:
            widths = [right - left for left, right in zip(bars, bars[1:])]
            prefix, suffix = bars[0] - full_ink.x0, full_ink.x1 - bars[-1]
    if widths is None:
        weights = [max(1.0, float(len(measure))) for measure in chunk]
        scale = full_ink.width / max(1.0, sum(weights))
        widths = [weight * scale for weight in weights]

    lines = []
    start, line_number = 0, 0
    while start < count:
        # Plan a group from measured widths, then render and verify.  The
        # plan only saves renders; the rendered ink is the source of truth.
        used = prefix if start == 0 else 0.0
        end = start
        while end < count:
            extra = widths[end] + (suffix if end == count - 1 else 0.0)
            if end > start and used + extra > available_width:
                break
            used += widths[end]
            end += 1
            if used > available_width and end > start + 1:
                break
        while True:
            label = f'wrap_{line_number}_{start}_{end}'
            pdf, ink = _render(render_source, render_tile, chunk,
                               line_key if start == 0 else '', time_sig, title,
                               workdir, local_repeats, label, source_chunk,
                               start, end)
            if ink.width <= available_width + 0.01:
                break
            if end - start == 1:
                raise LineTooWideError(
                    f'Jianpu measure {start + 1} needs {ink.width:.1f}pt but only '
                    f'{available_width:.1f}pt is available; cannot wrap it without '
                    'cropping or shrinking the glyphs')
            end -= 1
        lines.append((pdf, ink, start, end - start))
        start, line_number = end, line_number + 1
    return lines


def _mark_overfilled(page, target: fitz.Rect) -> None:
    """Draw a small review marker above an over-filled measure's cell.

    The marker is a red asterisk bracketed by two short vertical ticks, centred
    just above the measure, so a human reviewer can spot the over-filled bar
    without the measure being blanked or the line being dropped.
    """
    mid = (target.x0 + target.x1) / 2
    y = target.y0
    red = (1.0, 0.0, 0.0)
    # Two corner ticks delimiting the measure.
    for tick_x in (target.x0, target.x1):
        page.draw_line(fitz.Point(tick_x, y - 6), fitz.Point(tick_x, y - 2),
                       color=red, width=1.0)
    # A centred asterisk between them.
    page.insert_text(fitz.Point(mid - 3, y - 1), '*', fontsize=8, color=red)


def place_line(page, line, x: float, top: float, available_width: float,
               layout_mode: str, *, target_barlines, tile_barlines, place_cell,
               measure_report=None, overfilled_measures=()) -> str:
    """Draw one rendered Jianpu line below a source staff.

    Returns ``natural`` (wrap), ``aligned`` (per-measure alignment),
    ``line`` (whole line at natural width) or ``scaled`` (whole line
    squeezed because it has no usable barlines — a real fallback).

    Per-measure alignment stays ``aligned`` even when one measure had to be
    narrowed horizontally: that is the documented behaviour of ``single``
    layout, not a degraded fallback.  The individual measures that were
    squeezed are recorded in ``measure_report``.

    ``overfilled_measures`` is an optional set of *absolute* measure indices
    whose OMR durations exceed their metre.  Those measures are still rendered
    in full; a small marker is drawn above each one so the over-fill is visible
    for review instead of being silently blanked or skipped.
    """
    tile, ink, start, count = line
    height = ink.height
    with fitz.open(str(tile)) as src:
        if layout_mode == 'wrap':
            page.show_pdf_page(fitz.Rect(x, top, x + ink.width, top + height),
                               src, 0, clip=ink, overlay=True)
            return 'natural'

        bars = tile_barlines(src[0], ink, count)
        targets = list(target_barlines)
        # Barlines must describe the whole line and stay inside the staff's
        # available width; otherwise fall back to whole-line fitting.
        if (bars is None or len(targets) != count + 1
                or targets[-1] - x > available_width + 0.01):
            natural = ink.width
            width = min(natural, available_width)
            page.show_pdf_page(fitz.Rect(x, top, x + width, top + height),
                               src, 0, clip=ink, keep_proportion=False,
                               overlay=True)
            return 'line' if natural <= available_width + 0.01 else 'scaled'

        # Key label (first line only) sits just before the opening barline.
        leading = bars[0] - ink.x0
        if leading > 0.2:
            left = max(0.0, x - leading)
            page.show_pdf_page(fitz.Rect(left, top, left + leading, top + height),
                               src, 0, clip=fitz.Rect(ink.x0, ink.y0, bars[0], ink.y1),
                               keep_proportion=False, overlay=True)
        for offset, (left_bar, right_bar, target_left, target_right) in enumerate(
                zip(bars, bars[1:], targets, targets[1:])):
            clip = fitz.Rect(left_bar, ink.y0, right_bar, ink.y1)
            target = fitz.Rect(target_left, top, target_right, top + height)
            cell_unscaled = place_cell(page, src, clip, target)
            if measure_report is not None:
                measure_report.append({'measure_offset': start + offset,
                                       'placement': 'unscaled' if cell_unscaled else 'scaled',
                                       'source_width_points': round(clip.width, 3),
                                       'target_width_points': round(target.width, 3)})
            if (start + offset) in overfilled_measures:
                _mark_overfilled(page, target)
        trailing = ink.x1 - bars[-1]
        if trailing > 0.2:
            left = targets[-1]
            right = min(left + trailing, x + available_width)
            page.show_pdf_page(fitz.Rect(left, top, right, top + height), src, 0,
                               clip=fitz.Rect(bars[-1], ink.y0, ink.x1, ink.y1),
                               keep_proportion=False, overlay=True)
        return 'aligned'
