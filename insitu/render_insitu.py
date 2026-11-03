"""Typeset Jianpu beneath each engraved staff using jianpu-ly and LilyPond.

``single`` keeps one source staff/voice on exactly one Jianpu line, fitting
horizontal spacing to the source measure boundaries when detectable. ``wrap``
keeps the reference glyph size (staff size 20) and breaks at measure boundaries.
The default reflow engine moves whole source painting objects, adding pages of
the original size instead of lengthening the MediaBox or slicing through ink.
Legacy overlay placement is available with ``expand=False``; it never resizes
pages and reports rows that do not fit rather than silently clipping them.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import shutil
import copy
import hashlib
import json
from pathlib import Path
from typing import Optional

import pymupdf as fitz

sys.path.insert(0, str(Path(__file__).resolve().parent))        # align.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # _env.py
from _env import ensure_sumisora, jianpu_ly_script, lilypond_exe, python_exe  # noqa: E402

ensure_sumisora()

from core.notation.jianpu import (  # noqa: E402
    jianpu_note_token,
    layout_time_tokens,
    _with_time_signatures,
    extract_jianpu_measures,
    _get_score_key_tonic,
    _extract_part_repeat_barlines,
)
from core.render.jianpu_runner import _insert_repeat_bar_commands  # noqa: E402
from core.notation.musicxml_wedges import parse_musicxml_compatible  # noqa: E402

from align import align_from_files  # noqa: E402

JIANPU_LY = jianpu_ly_script()
PYTHON = python_exe()
LILYPOND = lilypond_exe()

# Where the top of the jianpu ink sits below `staff.bottom`, in page points.
Y_OFFSET = 20.0

# LilyPond staff size.  梆笛.jianpu.pdf is produced at jianpu-ly's default 20.
STAFF_SIZE = 20

# Comfortably wider than any real jianpu line (~185mm) yet small enough that
# ragged/ragged-last leave the system at its *natural* width instead of
# stretching measures out to fill the page.
_LINE_WIDTH_PT = 540


# ── per-measure tokens ───────────────────────────────────────────────────────

def _measure_texts(measures, time_sig: str) -> list[str]:
    """Return one jianpu-ly token string per measure, exactly as the library does."""
    out = []
    for m in measures:
        out.append(' '.join(jianpu_note_token(n) for n in m))
    return out


def _line_slur_continuations(measures, start: int, end: int):
    """Close/reopen a cross-system slur at the edges of independently rendered lines."""
    active = 0
    for measure in measures[:start]:
        for event in measure:
            active = max(0, active - event.slur_stops) + event.slur_starts
    entering = active
    line = [[copy.copy(event) for event in measure] for measure in measures[start:end]]
    # A tile is a separate LilyPond score. Keep only individual-pitch ties
    # whose endpoints both occur within this tile. Draw separate short
    # continuation arcs for endpoints on different rows and retain review.
    starts = {tie_id for measure in line for event in measure
              for _, tie_id in event.pitch_tie_starts}
    stops = {tie_id for measure in line for event in measure
             for _, tie_id in event.pitch_tie_stops}
    local_ties = starts & stops
    for measure in line:
        for event in measure:
            event.pitch_tie_carry_out = [(index, tie_id) for index, tie_id in event.pitch_tie_starts
                                         if tie_id not in local_ties]
            event.pitch_tie_carry_in = [(index, tie_id) for index, tie_id in event.pitch_tie_stops
                                        if tie_id not in local_ties]
            event.pitch_tie_starts = [(index, tie_id) for index, tie_id in event.pitch_tie_starts
                                      if tie_id in local_ties]
            event.pitch_tie_stops = [(index, tie_id) for index, tie_id in event.pitch_tie_stops
                                     if tie_id in local_ties]
    for measure in line:
        for event in measure:
            active = max(0, active - event.slur_stops) + event.slur_starts
    audible = [event for measure in line for event in measure
               if not event.is_rest and not event.is_grace]
    if audible:
        audible[0].slur_carry_in += entering
        audible[-1].slur_carry_out += active
    elif entering or active:
        print('  ! 跨行连音线没有可连接的音符，请校对')
    return line


def _jianpu_ly_source(lines: list[str], key_header: str, time_sig: str, title: str) -> str:
    """Wrap measure lines in a jianpu-ly header; one entry in `lines` = one system.

    `key_header` (e.g. ``1=C``) is emitted only when truthy.  In-situ we print it
    on the first line only, mirroring `梆笛.jianpu.pdf`, which prints ``1=C`` once
    rather than above every system.
    """
    head = ['% jianpu-ly.py', f'title={title}']
    if key_header:
        head.append(key_header)
    # Chord cells are written `,135`; a `'`/`,` inside them must know which
    # digit it belongs to or jianpu-ly aborts with "Ambiguous octave marks".
    # The library's own header sets this too (see core/notation/jianpu).
    # Each tile starts a new LilyPond score. Its automatically restarted bar
    # numbers would therefore label the fifth bar of every source staff "5",
    # regardless of the actual measure number, and can resemble musical ink.
    head += ['OctavesAfter', 'NoBarNums']
    # Layout time signatures are inserted per measure and hidden below.
    head += ['']
    body = [' | '.join(ln) + ' |' for ln in lines if ln]
    return '\n'.join(head + body) + '\n'


# ── LilyPond ─────────────────────────────────────────────────────────────────

def _run_jianpu_ly(txt: str, workdir: Path) -> str:
    """Run the vendored jianpu-ly; return the generated LilyPond source."""
    txt_path = workdir / 'line.jianpu.txt'
    txt_path.write_text(txt, encoding='utf-8')
    env = os.environ.copy()
    env['j2ly_sloppy_bars'] = '1'
    proc = subprocess.run(
        [PYTHON, str(JIANPU_LY), str(txt_path)],
        capture_output=True, text=True, env=env, cwd=str(workdir), timeout=300,
    )
    if proc.returncode != 0:
        raise RuntimeError(f'jianpu-ly failed: {proc.stderr[-800:]}')
    return proc.stdout


def _strip_title_and_tagline(ly_text: str) -> str:
    """Remove per-score titles and the LilyPond tagline from generated source.

    jianpu-ly copies a top-level ``title=<...>`` into per-score ``\\header{}``
    blocks, which would print a title above *every* stamped line and collide
    with the engraved staff below it.  It also leaves the "Music engraving by
    LilyPond" tagline enabled, whose glyph sits ~815pt down the page and
    inflates the ink bbox to a full A4 sheet.  Both are dropped here so the
    ink bbox is exactly the notes.
    """
    # Drop title / subtitle / composer header lines (whole ``\header{...}``
    # blocks that only carry a title are collapsed to empty).
    ly_text = re.sub(r'^\s*title\s*=.*$', '', ly_text, flags=re.MULTILINE)
    ly_text = re.sub(
        r'\\header\s*\{[^{}]*\}',
        lambda m: '' if ('title' in m.group(0) or 'subtitle' in m.group(0)) else m.group(0),
        ly_text,
    )
    # Force the tagline off inside the existing \paper{} block.  jianpu-ly only
    # mentions tagline in a comment, so always inject into \paper{}.
    if '\\paper {' in ly_text:
        ly_text = ly_text.replace('\\paper {', '\\paper {\n  tagline = ##f', 1)
    else:
        ly_text += '\n\\paper { tagline = ##f }\n'
    return ly_text


def _force_single_system(ly_text: str) -> str:
    """Pin the paper so a one-line jianpu source renders as one unbroken system.

    With the default (A4) line-width a dense measure can overflow and LilyPond
    silently breaks it onto a second system, which then collides with the next
    staff.  A generous line-width keeps everything on one system, while
    ``ragged-right = ##t`` / ``ragged-last = ##t`` make the system take its
    *natural* width (no justification), so measures stay compact instead of
    being stretched across the page.  The ``| |`` sequence jianpu-ly emits for
    an empty bar looks like a break hint to LilyPond, so it is collapsed first.
    """
    ly_text = ly_text.replace('| |', '|')
    # LilyPond prints a default 4/4 time signature on the jianpu staff even
    # when the source has no \time, which shows up as an Emmentaler "44" glyph
    # under every engraved staff.  Blank it out on every staff.
    ly_text = ly_text.replace(
        '\\override Staff.Stem #\'transparent = ##t',
        "\\override Staff.Stem #'transparent = ##t\n"
        "    \\override Staff.TimeSignature #'stencil = ##f")
    paper = (f'\\paper {{ line-width = {_LINE_WIDTH_PT}\\pt '
             'ragged-last = ##t ragged-right = ##t indent = 0 '
             'tagline = ##f }')
    return ly_text + '\n' + paper + '\n'


def _normal_line_end_barline(ly_text: str) -> str:
    """Replace jianpu-ly's synthetic final barline on each in-situ line.

    Run after repeat injection. A repeat already closing the line must not
    be overridden by another bar command at the same musical moment.
    Full-page jianpu rendering is left unchanged.
    """
    ly_text = re.sub(
        r'(\\bar\s*"(?:\.\|:|:\|\.(?:\|:)?)"\s*\|)\s*\\bar\s*"\|\."',
        r'\1', ly_text,
    )
    return re.sub(r'\\bar\s*"\|\."', r'\\bar "|"', ly_text)


def _render_ly_tile(ly_text: str, workdir: Path) -> tuple[Path, fitz.Rect]:
    """Render `ly_text` with LilyPond; return (pdf path, ink bbox) of the system."""
    ly_text = re.sub(r'#\(set-global-staff-size \d+\)',
                     f'#(set-global-staff-size {STAFF_SIZE})', ly_text)
    ly_text = _normal_line_end_barline(ly_text)
    ly_text = _strip_title_and_tagline(ly_text)
    if '% insitu-line-layout: single-system' not in ly_text:
        ly_text = _force_single_system(ly_text)
    ly_path = workdir / 'line.ly'
    ly_path.write_text(ly_text, encoding='utf-8')
    proc = subprocess.run(
        [LILYPOND, '-dno-point-and-click', '-o', str(workdir / 'line'), str(ly_path)],
        capture_output=True, text=True, timeout=300, cwd=str(workdir),
    )
    if proc.returncode != 0:
        raise RuntimeError(f'lilypond failed: {proc.stderr[-800:]}')
    pdf_path = workdir / 'line.pdf'
    doc = fitz.open(str(pdf_path))
    if len(doc) != 1:
        doc.close()
        raise RuntimeError('LilyPond tile contains multiple pages; refusing to omit later systems')
    page = doc[0]
    xs: list[float] = []
    ys: list[float] = []
    for blk in page.get_text('dict')['blocks']:
        for ln in blk.get('lines', []):
            for sp in ln['spans']:
                b = sp['bbox']
                xs += [b[0], b[2]]
                ys += [b[1], b[3]]
    for dr in page.get_drawings():
        r = dr['rect']
        xs += [r.x0, r.x1]
        ys += [r.y0, r.y1]
    doc.close()
    if not xs:
        raise RuntimeError('LilyPond produced an empty system')
    return pdf_path, fitz.Rect(min(xs), min(ys), max(xs), max(ys))


# ── main ─────────────────────────────────────────────────────────────────────

def _source_ink_boxes(page, low: float, high: float, raster=None):
    """Bounding boxes of indivisible visible glyphs and vector paths in a band."""
    import numpy as np
    if raster is None:
        raster = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False, colorspace=fitz.csGRAY)
    pixels = np.frombuffer(raster.samples, dtype=np.uint8).reshape(raster.height, raster.stride)
    boxes = []
    # Trace boxes follow the glyph's actual font size. The raw text extractor
    # can return a much taller line box for music fonts, falsely covering
    # entire gaps and preventing otherwise safe placement.
    for span in page.get_texttrace():
        for codepoint, _, _, bounds in span['chars']:
            if codepoint not in (9, 10, 13, 32):
                box = fitz.Rect(bounds)
                if box.y1 <= low or box.y0 >= high:
                    continue
                x0 = max(0, int(np.floor(box.x0 * 2)) - raster.x)
                x1 = min(raster.width, int(np.ceil(box.x1 * 2)) - raster.x)
                y0 = max(0, int(np.floor(box.y0 * 2)) - raster.y)
                y1 = min(raster.height, int(np.ceil(box.y1 * 2)) - raster.y)
                if x1 <= x0 or y1 <= y0:
                    continue
                ink_rows = np.flatnonzero(np.any(pixels[y0:y1, x0:x1] < 245, axis=1))
                if len(ink_rows):
                    # Font boxes frequently have large unused ascenders or
                    # descenders. Keep both disconnected parts of a glyph,
                    # but do not classify those blank margins as ink.
                    box.y0 = (raster.y + y0 + int(ink_rows[0])) / 2
                    box.y1 = (raster.y + y0 + int(ink_rows[-1]) + 1) / 2
                    boxes.append(box)
    for drawing in page.get_drawings():
        box = fitz.Rect(drawing['rect'])
        if box.x0 == box.x1 and box.y1 > box.y0:
            # Vertical paths can have zero-width bounding boxes even when
            # drawn with a visible stroke or a dashed pattern.
            box.x0 -= .25
            box.x1 += .25
        boxes.append(box)
    return [box for box in boxes if box.y1 > low and box.y0 < high and box.x1 > box.x0]


def _blank_cut(page, low: float, high: float) -> float:
    """Choose a low-ink horizontal cut in the gap between adjacent systems."""
    low = max(0.0, min(low, page.rect.height - 1))
    high = max(low + 0.5, min(high, page.rect.height))
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False, colorspace=fitz.csGRAY)
    import numpy as np
    pixels = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.stride)
    start, end = int(low * 2), min(pix.height, int(high * 2))
    counts = np.count_nonzero(pixels[start:end, :pix.width] < 245, axis=1)
    if not len(counts):
        return low
    # An empty pixel row may still pass between disconnected parts of one
    # glyph or path (for example the dot and stem of a letter). Prefer an
    # empty row that is also outside every source object's bounding box.
    rows = (start + np.arange(len(counts))) / 2
    clear_objects = np.ones(len(counts), dtype=bool)
    for box in _source_ink_boxes(page, low, high, pix):
        clear_objects &= ~((rows > box.y0 + .05) & (rows < box.y1 - .05))
    safe_mask = (counts == 0) & clear_objects
    # Clip-edge rasterization can differ slightly from the full-page raster.
    # Prefer a blank half-point on both sides of the boundary when available.
    padded = safe_mask.copy()
    if len(padded) > 2:
        padded[1:] &= counts[:-1] == 0
        padded[:-1] &= counts[1:] == 0
        padded[0] = padded[-1] = False
    safe = np.flatnonzero(padded)
    if not len(safe):
        safe = np.flatnonzero(safe_mask)
    if len(safe):
        middle = (len(counts) - 1) / 2
        row = min(safe, key=lambda i: abs(i - middle))
        blank = np.flatnonzero(counts == 0)
        nearest = min(blank, key=lambda i: abs(i - middle))
        # A distant straight cut may move an annotation to the wrong staff.
        # Let the tempo-aware / piecewise validation handle a much closer
        # candidate before selecting such a distant cut.
        if abs(row - middle) <= abs(nearest - middle) + 4:
            return (start + int(row)) / 2
    best = int(counts.min())
    choices = np.flatnonzero(counts == best)
    middle = (len(counts) - 1) / 2
    row = min(choices, key=lambda i: abs(i - middle))
    if best:
        print(f'  ! 无完全空白的插入位置，尝试按列寻找安全切线（最少 {best} 像素）')
    return (start + int(row)) / 2


def _cut_has_ink(page, cut: float, overrides=()) -> bool:
    """Reject ink or split objects along the whole possibly piecewise cut."""
    import numpy as np
    boundaries = sorted({0.0, page.rect.width,
                         *(max(0.0, min(page.rect.width, x))
                           for left, right, _ in overrides for x in (left, right))})
    segments = []
    for left, right in zip(boundaries, boundaries[1:]):
        middle = (left + right) / 2
        level = min([cut] + [y for x0, x1, y in overrides if x0 <= middle <= x1])
        segments.append((left, right, level))
    lowest = min([cut] + [y for _, _, y in overrides])
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False, colorspace=fitz.csGRAY)
    pixels = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.stride)
    boxes = _source_ink_boxes(page, lowest - .1, cut + .1, pix)
    for left, right, level in segments:
        row = int(np.floor(level * 2)) - pix.y
        x0 = max(0, int(np.floor(left * 2)) - pix.x)
        x1 = min(pix.width, int(np.ceil(right * 2)) - pix.x)
        if not 0 <= row < pix.height or np.any(pixels[row, x0:x1] < 245):
            return True
        if any(box.x0 < right and box.x1 > left and box.y0 + .05 < level < box.y1 - .05
               for box in boxes):
            return True
    for first, second in zip(segments, segments[1:]):
        seam, lower, upper = first[1], min(first[2], second[2]), max(first[2], second[2])
        if lower < upper and any(box.x0 < seam < box.x1 and box.y0 < upper and box.y1 > lower
                                 for box in boxes):
            return True
    return False


def _piecewise_blank_cut(page, low: float, high: float):
    """Find a blank cut per column region, changing height only in blank columns.

    A region spanning the entire gap without a blank row cannot safely be
    divided; in that case return None rather than tear original score ink.
    """
    import numpy as np
    scale = 2
    low = max(0.0, min(low, page.rect.height - 1))
    high = max(low + .5, min(high, page.rect.height))
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale),
                          clip=fitz.Rect(0, low, page.rect.width, high),
                          alpha=False, colorspace=fitz.csGRAY)
    pixels = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.stride)[:, :pix.width]
    if not pix.height or not pix.width:
        return None
    occupied = np.any(pixels < 245, axis=0)
    # Treat each source glyph and vector drawing as indivisible even if its
    # ink contains empty columns or rows. Otherwise separate components of a
    # single symbol could be translated to opposite sides of the new gap.
    boxes = _source_ink_boxes(page, low, high)
    for box in boxes:
        left = max(0, int(box.x0 * scale) - pix.x)
        right = min(pix.width, int(np.ceil(box.x1 * scale)) - pix.x)
        occupied[left:right] = True
    indices = np.flatnonzero(occupied)
    if not len(indices):
        return None
    regions = np.split(indices, np.where(np.diff(indices) > 1)[0] + 1)
    chosen = []
    center = (pix.height - 1) / 2
    for region in regions:
        left, right = int(region[0]), int(region[-1]) + 1
        clear_rows = np.flatnonzero(np.all(pixels[:, left:right] >= 245, axis=1))
        region_left = (pix.x + left) / scale
        region_right = (pix.x + right) / scale
        overlapping = [box for box in boxes if box.x0 < region_right and box.x1 > region_left]
        clear_rows = [y for y in clear_rows
                      if all(not box.y0 - .25 <= (pix.y + y) / scale <= box.y1 + .25
                             for box in overlapping)]
        if not len(clear_rows):
            return None
        row = int(min(clear_rows, key=lambda y: abs(y - center)))
        chosen.append((region_left, region_right, (pix.y + row) / scale))
    base = max(y for _, _, y in chosen)
    overrides = [(left, right, y) for left, right, y in chosen if y < base]
    return base, overrides


def _tempo_cut_overrides(page, low: float, base_cut: float, next_top: float):
    """Find safe column-local cuts above text tempos belonging to the next staff.

    Return (left, right, cut) regions only when the surrounding pixel columns
    are blank across the entire affected strip.  This avoids tearing source ink
    where the insertion boundary changes height.  Other source annotations and
    ambiguous cases keep the ordinary straight cut.
    """
    import numpy as np

    pattern = re.compile(r'^\s*(?:q|h|e|♩|♪)\s*=\s*\d+\s*$', re.IGNORECASE)
    spans = [span for block in page.get_text('dict')['blocks']
             for line in block.get('lines', []) for span in line.get('spans', [])
             if pattern.match(span.get('text', ''))
             and low < float(span['origin'][1]) <= base_cut + 2
             and 3 <= next_top - float(span['origin'][1]) <= 40]
    if not spans:
        return []
    scale = 2
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False, colorspace=fitz.csGRAY)
    pixels = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.stride)[:, :pix.width]
    found = []
    for span in spans:
        origin_y = float(span['origin'][1])
        x0, x1 = float(span['bbox'][0]), float(span['bbox'][2])
        xa, xb = max(0, int((x0 - 2) * scale)), min(pix.width, int((x1 + 2) * scale))
        ya, yb = max(0, int(low * scale)), min(pix.height, int((origin_y - 2) * scale))
        if xb <= xa or yb <= ya:
            continue
        ink_rows = np.any(pixels[ya:yb, xa:xb] < 245, axis=1)
        # Work upward from the tempo baseline; the first blank row is
        # immediately above the connected rows of its printed mark.
        ink_indices = np.flatnonzero(ink_rows)
        if not len(ink_indices):
            continue
        top = int(ink_indices[-1])
        while top > 0 and ink_rows[top - 1]:
            top -= 1
        if top == 0:
            continue
        cut = (ya + top - 0.5) / scale
        if cut >= base_cut - 1:
            continue
        # Do not introduce a vertical seam through any source ink.
        seam_top = max(0, int(cut * scale) - 1)
        seam_bottom = min(pix.height, int(base_cut * scale) + 2)
        clear_columns = np.all(pixels[seam_top:seam_bottom] >= 245, axis=0)
        left_options = np.flatnonzero(clear_columns[max(0, xa - 40):xa])
        right_options = np.flatnonzero(clear_columns[xb:min(pix.width, xb + 40)])
        if not len(left_options) or not len(right_options):
            continue
        left = (max(0, xa - 40) + int(left_options[-1])) / scale
        right = (xb + int(right_options[0])) / scale
        cut_row = max(0, min(pix.height - 1, int(cut * scale)))
        if (left < right and np.all(pixels[cut_row, int(left * scale):
                                           int(right * scale)] >= 245)):
            found.append((left, right, cut))
    if len(found) < len(spans):
        print(f'  ! 第 {page.number + 1} 页速度标记附近无法安全分区，请校对其所属谱表。')
    return found


def _copy_with_column_cuts(output_page, source_doc, page_index: int, insertions):
    """Copy vectors, limiting column clipping to the affected inter-staff gap."""
    page = source_doc[page_index]
    width, height = page.rect.width, page.rect.height
    def copy_region(x0, y0, x1, y1, shift):
        if x1 <= x0 or y1 <= y0:
            return
        clip = fitz.Rect(x0, y0, x1, y1)
        target = fitz.Rect(x0, y0 + shift, x1, y1 + shift)
        output_page.show_pdf_page(target, source_doc, page_index, clip=clip, keep_proportion=False)

    previous = shift = 0.0
    for base_cut, added, _, _, _, _, _, overrides in insertions:
        first_cut = min([base_cut] + [y for _, _, y in overrides])
        copy_region(0, previous, width, first_cut, shift)
        boundaries = {0.0, width}
        for left, right, _ in overrides:
            boundaries.update((max(0.0, left), min(width, right)))
        columns = sorted(boundaries)
        for x0, x1 in zip(columns, columns[1:]):
            mid = (x0 + x1) / 2
            cut = min([base_cut] + [y for left, right, y in overrides if left <= mid <= right])
            copy_region(x0, first_cut, x1, cut, shift)
            copy_region(x0, cut, x1, base_cut, shift + added)
        previous, shift = base_cut, shift + added
    copy_region(0, previous, width, height, shift)


def _tile_barlines(page, ink: fitz.Rect, count: int) -> list[float] | None:
    """Find one rendered row's opening and closing measure dividers."""
    candidates = []
    for drawing in page.get_drawings():
        rect = drawing['rect']
        if (rect.width < 2.5 and rect.height > 12 and ink.y0 - 1 <= rect.y0
                and rect.y1 <= ink.y1 + 1 and ink.x0 - 1 <= rect.x0 <= ink.x1 + 1):
            candidates.append((rect.x0, rect.y0, rect.y1))
    groups: list[list[tuple[float, float, float]]] = []
    for candidate in candidates:
        group = next((g for g in groups if abs(g[0][1] - candidate[1]) < 0.6
                      and abs(g[0][2] - candidate[2]) < 0.6), None)
        if group is None:
            groups.append([candidate])
        else:
            group.append(candidate)
    possible = []
    for group in groups:
        xs = []
        for x, _, _ in sorted(group):
            if not xs or x - xs[-1] > 5.5:
                xs.append(x)
        if len(xs) == count + 1:
            possible.append(xs)
    return possible[0] if len(possible) == 1 else None


def _blank_column_runs(page, clip: fitz.Rect) -> list[tuple[float, float]]:
    """Find empty columns where horizontal spacing can change without cutting ink."""
    import numpy as np
    scale = 2
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=clip,
                          alpha=False, colorspace=fitz.csGRAY)
    pixels = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.stride)
    clear = np.all(pixels[:, :pix.width] >= 245, axis=0)
    runs = []
    start = None
    for index, blank in enumerate(clear):
        if blank and start is None:
            start = index
        elif not blank and start is not None:
            if index - start >= 3 and start > 0 and index < len(clear):
                # Pixmaps are aligned to the device pixel grid, which may
                # start slightly before the requested floating-point clip.
                left = (pix.x + start + 0.5) / scale
                right = (pix.x + index - 0.5) / scale
                if right > left:
                    runs.append((left, right))
            start = None
    return runs


def _curve_column_runs(page, clip: fitz.Rect) -> list[tuple[float, float]]:
    """Find columns containing only long slur curves, suitable for elastic fitting.

    Text bounding boxes are protected even where their raster happens to
    overlap a curve. Other artwork remains protected by the raster mask.
    """
    import numpy as np

    drawings = page.get_drawings()
    curves = [d for d in drawings if d['rect'].intersects(clip)
              and d['items'] and all(item[0] == 'c' for item in d['items'])
              and d['rect'].width >= clip.width * .35
              and d['rect'].height <= min(15, d['rect'].width * .3)
              and d['type'] in ('s', 'f')]
    if not curves:
        return []
    scale = 4
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=clip,
                          alpha=False, colorspace=fitz.csGRAY)
    pixels = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.stride)[:, :pix.width]
    mask_doc = fitz.open()
    try:
        mask_page = mask_doc.new_page(width=page.rect.width, height=page.rect.height)
        for drawing in curves:
            shape = mask_page.new_shape()
            for _, a, b, c, d in drawing['items']:
                shape.draw_bezier(a, b, c, d)
            shape.finish(color=(0, 0, 0), fill=(0, 0, 0) if drawing['type'] == 'f' else None,
                         width=max(1.0, drawing.get('width') or 0), closePath=False)
            shape.commit()
        mpix = mask_page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=clip,
                                    alpha=False, colorspace=fitz.csGRAY)
        mask = np.frombuffer(mpix.samples, dtype=np.uint8).reshape(mpix.height, mpix.stride)[:, :mpix.width] < 245
        if mask.shape != pixels.shape:
            return []
        expanded = mask.copy()
        expanded[1:] |= mask[:-1]
        expanded[:-1] |= mask[1:]
        expanded[:, 1:] |= mask[:, :-1]
        expanded[:, :-1] |= mask[:, 1:]
        clear = ~np.any((pixels < 245) & ~expanded, axis=0)
        curve_ids = {id(d) for d in curves}
        for drawing in drawings:
            box = drawing['rect']
            if id(drawing) not in curve_ids and box.intersects(clip):
                left = max(0, int(box.x0 * scale - pix.x) - 1)
                right = min(pix.width, int(box.x1 * scale - pix.x) + 2)
                clear[left:right] = False
        for info in page.get_image_info():
            box = fitz.Rect(info['bbox'])
            if box.intersects(clip):
                left = max(0, int(box.x0 * scale - pix.x) - 1)
                right = min(pix.width, int(box.x1 * scale - pix.x) + 2)
                clear[left:right] = False
        for block in page.get_text('rawdict')['blocks']:
            for line in block.get('lines', []):
                for span in line.get('spans', []):
                    for char in span['chars']:
                        box = fitz.Rect(char['bbox'])
                        if box.intersects(clip):
                            left = max(0, int(box.x0 * scale - pix.x) - 1)
                            right = min(pix.width, int(box.x1 * scale - pix.x) + 2)
                            clear[left:right] = False
        runs = []
        start = None
        for index, blank in enumerate(clear):
            if blank and start is None:
                start = index
            elif not blank and start is not None:
                if index - start >= 6 and start > 0:
                    left = (pix.x + start + .5) / scale
                    right = (pix.x + index - .5) / scale
                    if right > left:
                        runs.append((left, right))
                start = None
        return runs
    finally:
        mask_doc.close()


def _place_curve_gaps(target_page, source_doc, clip: fitz.Rect, target: fitz.Rect) -> bool:
    """Resize curve-only columns continuously, leaving all other ink unchanged."""
    gaps = _curve_column_runs(source_doc[0], clip)
    delta = target.width - clip.width
    if not gaps:
        return False
    capacity = [max(0.0, right - left - 1.5) for left, right in gaps]
    if delta < 0 and sum(capacity) < -delta + .01:
        return False
    changes = ([delta / len(gaps)] * len(gaps) if delta >= 0 else
               [delta * value / sum(capacity) for value in capacity])
    previous, dest_x = clip.x0, target.x0
    for (left, right), change in zip(gaps, changes):
        for a, b, width in ((previous, left, left - previous),
                            (left, right, right - left + change)):
            if b > a and width > 0:
                target_page.show_pdf_page(fitz.Rect(dest_x, target.y0, dest_x + width, target.y1),
                                         source_doc, 0, clip=fitz.Rect(a, clip.y0, b, clip.y1),
                                         keep_proportion=False)
                dest_x += width
        previous = right
    if clip.x1 > previous:
        target_page.show_pdf_page(fitz.Rect(dest_x, target.y0, target.x1, target.y1), source_doc, 0,
                                 clip=fitz.Rect(previous, clip.y0, clip.x1, clip.y1),
                                 keep_proportion=False)
    return True


def _slur_seam(page, clip: fitz.Rect, target_width: float):
    """Find an empty separator below long curves, if they are the only upper ink.

    The upper band can then be scaled while the lower band retains unscaled
    digits. Reject ambiguous artwork, cuts through text, and misaligned curve
    endpoints. All tests use the visible raster; PDF text bboxes alone are too
    large for LilyPond's music font.
    """
    import numpy as np

    if abs(target_width - clip.width) < .1 or clip.height < 6 or clip.width < 12:
        return None
    drawings = page.get_drawings()
    curves = [d for d in drawings if d['rect'].intersects(clip)
              and d['items'] and all(item[0] == 'c' for item in d['items'])
              and d['rect'].width >= clip.width * .35
              and d['rect'].y1 < clip.y1 - 2
              and d['type'] in ('s', 'f')]
    if not curves:
        return None
    bottom = max(d['rect'].y1 for d in curves)
    scale = 4
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=clip,
                          alpha=False, colorspace=fitz.csGRAY)
    pixels = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.stride)[:, :pix.width]
    # The first and last two points can contain the measure's vertical
    # barlines, which may safely be divided at the horizontal seam.
    margin = 2 * scale
    if pix.width <= 2 * margin:
        return None
    start = max(0, int((bottom + .25) * scale) - pix.y)
    end = min(pix.height - 2, int((bottom + 3) * scale) - pix.y)
    choices = [i for i in range(start, end) if np.all(pixels[i:i + 2, margin:-margin] >= 245)]
    if not choices:
        return None
    seam = (pix.y + choices[0] + 1) / scale
    if not clip.y0 + 1 < seam < clip.y1 - 2:
        return None
    # Never split a glyph close to the barline edge that was excluded above.
    for block in page.get_text('rawdict')['blocks']:
        for line in block.get('lines', []):
            for span in line.get('spans', []):
                for char in span['chars']:
                    box = fitz.Rect(char['bbox'])
                    if box.y0 < seam < box.y1 and ((box.x1 > clip.x0 and box.x0 < clip.x0 + 2)
                                                    or (box.x0 < clip.x1 and box.x1 > clip.x1 - 2)):
                        return None
    lower = fitz.Rect(clip.x0, seam, clip.x1, clip.y1)
    gaps = _blank_column_runs(page, lower)
    capacity = [max(0.0, right - left - 1.5) for left, right in gaps]
    total = sum(capacity)
    delta = target_width - clip.width
    remove = max(0.0, -delta)
    if (delta < 0 and total < remove + .01) or (delta > 0 and not gaps):
        return None
    # Keep every visible slur endpoint near its associated digit after the
    # two bands undergo different horizontal mappings.
    for drawing in curves:
        for item in drawing['items']:
            for point in (item[1], item[4]):
                if clip.x0 < point.x < clip.x1:
                    elastic = point.x - clip.x0
                    if delta < 0:
                        for (left, right), cap in zip(gaps, capacity):
                            amount = remove * cap / total
                            mid = (left + right) / 2
                            elastic -= min(amount, max(0.0, point.x - (mid - amount / 2)))
                    else:
                        elastic += delta * sum(point.x > (left + right) / 2
                                               for left, right in gaps) / len(gaps)
                    uniform = (point.x - clip.x0) * target_width / clip.width
                    if abs(elastic - uniform) > 1.0:
                        return None
    # Construct a generous mask of the known curves. Any other visible ink in
    # the upper band makes the separation unsafe, including text and symbols.
    mask_doc = fitz.open()
    try:
        mask_page = mask_doc.new_page(width=page.rect.width, height=page.rect.height)
        for drawing in curves:
            shape = mask_page.new_shape()
            for _, a, b, c, d in drawing['items']:
                shape.draw_bezier(a, b, c, d)
            shape.finish(color=(0, 0, 0), fill=(0, 0, 0) if drawing['type'] == 'f' else None,
                         width=max(1.0, drawing.get('width') or 0), closePath=False)
            shape.commit()
        # Include only vertical measure boundaries in the edge mask; other
        # ink near the clipping edge must still be accounted for.
        for drawing in drawings:
            rect = drawing['rect']
            if (drawing['type'] == 's' and rect.width < 2.5 and rect.height > 12
                    and min(abs(rect.x0 - clip.x0), abs(rect.x0 - clip.x1)) < 1.5
                    and drawing['items'] and all(item[0] == 'l' for item in drawing['items'])):
                shape = mask_page.new_shape()
                for _, a, b in drawing['items']:
                    shape.draw_line(a, b)
                shape.finish(color=(0, 0, 0), width=max(1.0, drawing.get('width') or 0), closePath=False)
                shape.commit()
        mask_pix = mask_page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=clip,
                                         alpha=False, colorspace=fitz.csGRAY)
        mask = np.frombuffer(mask_pix.samples, dtype=np.uint8).reshape(mask_pix.height, mask_pix.stride)[:, :mask_pix.width] < 245
        if mask.shape != pixels.shape:
            return None
        # Tolerate small rasterization differences without accepting other ink.
        expanded = mask.copy()
        expanded[1:] |= mask[:-1]
        expanded[:-1] |= mask[1:]
        expanded[:, 1:] |= mask[:, :-1]
        expanded[:, :-1] |= mask[:, 1:]
        seam_row = max(0, min(pix.height, int(seam * scale) - pix.y))
        if np.any((pixels[:seam_row] < 245) & ~expanded[:seam_row]):
            return None
    finally:
        mask_doc.close()
    return seam


def _place_cell(target_page, source_doc, clip: fitz.Rect, target: fitz.Rect) -> bool:
    """Fit a cell by changing blank gaps, preserving glyph proportions when possible."""
    delta = target.width - clip.width
    gaps = _blank_column_runs(source_doc[0], clip)
    if abs(delta) < 0.1:
        target_page.show_pdf_page(target, source_doc, 0, clip=clip, keep_proportion=False)
        return True
    if ((delta < 0 and sum(max(0.0, right - left - 1.5) for left, right in gaps) < -delta - .01)
            or (delta > 0 and not gaps)):
        if _place_curve_gaps(target_page, source_doc, clip, target):
            return True
        seam = _slur_seam(source_doc[0], clip, target.width)
        if seam is not None:
            dest_y = target.y0 + (seam - clip.y0) * target.height / clip.height
            target_page.show_pdf_page(fitz.Rect(target.x0, target.y0, target.x1, dest_y), source_doc, 0,
                                     clip=fitz.Rect(clip.x0, clip.y0, clip.x1, seam), keep_proportion=False)
            return _place_cell(target_page, source_doc, fitz.Rect(clip.x0, seam, clip.x1, clip.y1),
                               fitz.Rect(target.x0, dest_y, target.x1, target.y1))
    if delta > 0 and gaps:
        cuts = [clip.x0, *[(left + right) / 2 for left, right in gaps], clip.x1]
        shift = 0.0
        extra = delta / len(gaps)
        for left, right in zip(cuts, cuts[1:]):
            dest = fitz.Rect(target.x0 + left - clip.x0 + shift, target.y0,
                             target.x0 + right - clip.x0 + shift, target.y1)
            target_page.show_pdf_page(dest, source_doc, 0,
                                     clip=fitz.Rect(left, clip.y0, right, clip.y1), keep_proportion=False)
            shift += extra
        return True
    if delta < 0 and gaps:
        capacity = [max(0.0, right - left - 1.5) for left, right in gaps]
        total = sum(capacity)
        if total > 0:
            # Remove safe whitespace first. If that is insufficient, the
            # remaining scale is smaller than scaling the entire cell directly.
            remove = min(-delta, total)
            scale = target.width / (clip.width - remove)
            removed = 0.0
            previous = clip.x0
            for (left, right), available in zip(gaps, capacity):
                amount = remove * available / total
                mid = (left + right) / 2
                cut_start, cut_end = mid - amount / 2, mid + amount / 2
                if cut_start > previous:
                    dest = fitz.Rect(target.x0 + (previous - clip.x0 - removed) * scale, target.y0,
                                     target.x0 + (cut_start - clip.x0 - removed) * scale, target.y1)
                    target_page.show_pdf_page(dest, source_doc, 0,
                                             clip=fitz.Rect(previous, clip.y0, cut_start, clip.y1),
                                             keep_proportion=False)
                removed += amount
                previous = cut_end
            if clip.x1 > previous:
                target_page.show_pdf_page(fitz.Rect(target.x0 + (previous - clip.x0 - removed) * scale, target.y0,
                                                    target.x1, target.y1), source_doc, 0,
                                          clip=fitz.Rect(previous, clip.y0, clip.x1, clip.y1),
                                          keep_proportion=False)
            return scale >= 0.999
    # No safe blank columns: scale only this measure, preserving all ink.
    target_page.show_pdf_page(target, source_doc, 0, clip=clip, keep_proportion=False)
    return False


def _place_tile(target_page, tile: Path, ink: fitz.Rect, layout, top: float, count: int,
                *, measure_report=None) -> str:
    """Place each Jianpu measure below the corresponding engraved measure."""
    with fitz.open(str(tile)) as src:
        bars = _tile_barlines(src[0], ink, count)
        targets = [layout.staff.x0, *layout.barlines]
        if bars is None or len(targets) != count + 1:
            target_page.show_pdf_page(fitz.Rect(layout.staff.x0, top, layout.staff.x0 + ink.width,
                                                top + ink.height), src, 0, clip=ink, overlay=True)
            return 'line'
        # Keep the first line's key label and a final double bar outside cells.
        if bars[0] - ink.x0 > 0.2:
            clip = fitz.Rect(ink.x0, ink.y0, bars[0], ink.y1)
            target_page.show_pdf_page(fitz.Rect(targets[0] - clip.width, top, targets[0], top + ink.height),
                                      src, 0, clip=clip, keep_proportion=False)
        unscaled = True
        for offset, (left, right, x0, x1) in enumerate(zip(bars, bars[1:], targets, targets[1:])):
            clip = fitz.Rect(left, ink.y0, right, ink.y1)
            target = fitz.Rect(x0, top, x1, top + ink.height)
            cell_unscaled = _place_cell(target_page, src, clip, target)
            if measure_report is not None:
                measure_report.append({'measure_offset': offset,
                                       'placement': 'unscaled' if cell_unscaled else 'scaled',
                                       'source_width_points': round(clip.width, 3),
                                       'target_width_points': round(target.width, 3)})
            unscaled = cell_unscaled and unscaled
        if ink.x1 - bars[-1] > 0.2:
            clip = fitz.Rect(bars[-1], ink.y0, ink.x1, ink.y1)
            target_page.show_pdf_page(fitz.Rect(targets[-1], top, targets[-1] + clip.width, top + ink.height),
                                      src, 0, clip=clip, keep_proportion=False)
        return 'aligned' if unscaled else 'scaled'


def _write_json_report(path: Path, report: dict) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix='.insitu-report-', suffix='.json', delete=False) as f:
            temporary = Path(f.name)
            json.dump(report, f, ensure_ascii=False, indent=2)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _render_source(chunk, line_key, time_sig, title, workdir, local_repeats):
    text = _run_jianpu_ly(_jianpu_ly_source([chunk], line_key, time_sig, title), workdir)
    return _insert_repeat_bar_commands(text, local_repeats)


def build(pdf_path: Path, mxl_path: Path, out_path: Path,
          title: Optional[str] = None, expand: bool = True,
          pitch_mode: str = 'movable', layout: str = 'single') -> Path:
    from insitu.line_layout import render_lines, place_line
    from insitu.reflow import SourceObjects, paginate

    pdf_path, mxl_path, out_path = map(Path, (pdf_path, mxl_path, out_path))
    if out_path.resolve() in {pdf_path.resolve(), mxl_path.resolve()}:
        raise ValueError('output must not replace a source PDF or MusicXML')
    if layout not in ('single', 'wrap'):
        raise ValueError(f'unsupported insitu layout: {layout}')
    if pitch_mode not in ('movable', 'fixed'):
        raise ValueError(f'unsupported pitch mode: {pitch_mode}')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_path = out_path.with_suffix('.report.json')
    protected = {pdf_path.resolve(), mxl_path.resolve(), out_path.resolve()}
    while report_path.resolve() in protected:
        report_path = report_path.with_name(report_path.stem + '.report.json')
    # Never leave a previous success report beside an interrupted regeneration.
    _write_json_report(report_path, {'status': 'in_progress',
                                     'scope': 'in_situ_rendering_not_omr_accuracy',
                                     'source_pdf': str(pdf_path), 'source_musicxml': str(mxl_path),
                                     'output_pdf': str(out_path), 'layout': layout})
    staves, slices = align_from_files(pdf_path, mxl_path)
    score = parse_musicxml_compatible(mxl_path)
    tonic, key_header = _get_score_key_tonic(score)
    if pitch_mode == 'fixed':
        tonic, key_header = 0, '1=C'
    part = score.parts[0] if score.parts else None
    # Source scores are monophonic: one staff maps to exactly one jianpu line.
    # Any OMR-tagged "second voice" is recognition noise, never a second track.
    tracks = [extract_jianpu_measures(score, tonic, follow_key_changes=True,
                                      pitch_mode=pitch_mode)]
    if len(score.parts) > 1:
        print('[insitu] 检测到多个乐器分部；当前原位对齐仅使用第一个分部，请校对。')
    repeat_bars = _extract_part_repeat_barlines(part) if part is not None else {}
    title = title or pdf_path.stem
    slice_map = {sl.staff_index: sl for sl in slices}
    row_reports = []
    for index, staff_layout in enumerate(staves):
        sl = slice_map.get(index)
        row_reports.append({'staff_index': index + 1, 'page': staff_layout.staff.page,
                            'musicxml_start_index': sl.omr_start if sl else None,
                            'musicxml_end_index_exclusive': sl.omr_end if sl else None,
                            'pdf_measures': sl.pdf_measures if sl else None,
                            'musicxml_measures': sl.omr_measures if sl else None,
                            'voices': [], 'rendered': False})

    with fitz.open(str(pdf_path)) as doc, tempfile.TemporaryDirectory(prefix='insitu_ly_') as temporary:
        tmpdir, tiles = Path(temporary), {}
        first_key = True
        for index, staff_layout in enumerate(staves):
            sl = slice_map.get(index)
            if sl is None:
                continue
            staff = staff_layout.staff
            page_width = doc[staff.page - 1].rect.width
            available = min(staff.x1 + 1, page_width - 18) - staff.x0
            row_report = row_reports[index]
            line_tiles = []
            for voice_index, (measures, time_sig) in enumerate(tracks):
                voice_report = {'voice_index': voice_index + 1, 'rendered': False,
                                'placement': None, 'compressed_measure_count': 0,
                                'error': None, 'measures': [], 'lines': []}
                row_report['voices'].append(voice_report)
                voice_report['tie_review_measures'] = [
                    {'musicxml_index': sl.omr_start + offset + 1,
                     'reasons': sorted({event.tie_review for event in measure if event.tie_review})}
                    for offset, measure in enumerate(measures[sl.omr_start:sl.omr_end])
                    if any(event.tie_review for event in measure)
                ]

                def source_chunk(start, end):
                    absolute = sl.omr_start + start
                    line = _line_slur_continuations(measures, absolute, sl.omr_start + end)
                    texts = _with_time_signatures(_measure_texts(line, time_sig), 0,
                                                  layout_time_tokens(line, 64))
                    changes = getattr(measures, 'key_changes', {})
                    return [f'{changes[absolute + offset][1]} {text}'
                            if absolute + offset in changes else text
                            for offset, text in enumerate(texts)]

                chunk = source_chunk(0, sl.omr_end - sl.omr_start)
                if not chunk:
                    voice_report['error'] = 'no_measures_assigned'
                    continue
                try:
                    workdir = tmpdir / f'sys_{index}_voice_{voice_index}'
                    workdir.mkdir(parents=True, exist_ok=True)
                    local_repeats = {i - sl.omr_start: info for i, info in repeat_bars.items()
                                     if sl.omr_start <= i < sl.omr_end}
                    lines = render_lines(chunk, key_header if first_key else '', time_sig,
                                         title, workdir, local_repeats, available, layout,
                                         render_source=_render_source, render_tile=_render_ly_tile,
                                         source_chunk=source_chunk,
                                         tile_barlines=_tile_barlines)
                    line_tiles.extend((*line, voice_report) for line in lines)
                    voice_report['rendered'] = True
                    first_key = False
                except Exception as exc:
                    voice_report['error'] = str(exc)
                    print(f'  ! staff {index + 1} voice {voice_index + 1}: {exc}')
            tiles[index] = line_tiles

        def place_tiles(page, index, row_tiles, top):
            staff_layout = staves[index]
            available = min(staff_layout.staff.x1 + 1, page.rect.width - 18) - staff_layout.staff.x0
            for tile, ink, start, count, voice in row_tiles:
                before = len(voice['measures'])
                placement = place_line(page, (tile, ink, start, count), staff_layout.staff.x0,
                                       top, available, layout,
                                       target_barlines=[staff_layout.staff.x0, *staff_layout.barlines],
                                       tile_barlines=_tile_barlines, place_cell=_place_cell,
                                       measure_report=voice['measures'])
                compressed = sum(1 for item in voice['measures'][before:]
                                 if item['placement'] == 'scaled')
                voice['compressed_measure_count'] = (
                    voice.get('compressed_measure_count', 0) + compressed)
                previous = voice['placement']
                rank = {'natural': 0, 'aligned': 0, 'line': 1, 'scaled': 2}
                if previous is None or rank[placement] > rank[previous]:
                    voice['placement'] = placement
                voice['lines'].append({'output_page': page.number + 1,
                                       'measure_start': start, 'measure_count': count,
                                       'top': top, 'bottom': top + ink.height,
                                       'natural_width_points': ink.width,
                                       'available_width_points': available,
                                       'placement': placement,
                                       'compressed_measure_count': compressed})
                top += ink.height + 9
            return top - 9

        with fitz.open() as output:
            if expand:
                page_map = paginate(doc, staves, tiles, row_reports, output, place_tiles)
            else:
                # Explicit legacy overlay: retain source positions, but never
                # clip a tile outside the page or overwrite the next staff.
                output.insert_pdf(doc)
                page_map = [{'output_page': i + 1, 'source_page': i + 1} for i in range(len(doc))]
                for page_index in range(len(doc)):
                    page_layouts = [(i, s) for i, s in enumerate(staves) if s.staff.page == page_index + 1]
                    objects = SourceObjects(doc, page_index, page_layouts)
                    try:
                        for pos, row in enumerate(objects.rows):
                            row_report = row_reports[row.index]
                            top = max(row.bottom + 9, staves[row.index].staff.bottom + Y_OFFSET)
                            bottom = (objects.rows[pos + 1].top - 9 if pos + 1 < len(objects.rows)
                                      else min([doc[page_index].rect.height - 18] +
                                               [g.bounds.y0 - 9 for g in objects.footer if not g.bounds.is_empty]))
                            row_tiles = tiles.get(row.index, [])
                            needed = sum(t[1].height + 9 for t in row_tiles) - 9
                            if row_tiles and top + needed > bottom:
                                for *_, voice in row_tiles:
                                    voice.update(rendered=False, error='insufficient_overlay_space: use --engine spaced')
                            elif row_tiles:
                                place_tiles(output[page_index], row.index, row_tiles, top)
                            row_report.update(output_page=page_index + 1, source_placement='unchanged',
                                              source_bounds=[row.top, row.bottom],
                                              output_source_bounds=[row.top, row.bottom])
                    finally:
                        objects.close()
            # Save atomically so layout failures cannot destroy a previous PDF.
            with tempfile.NamedTemporaryFile(dir=out_path.parent, prefix='.insitu-', suffix='.pdf',
                                             delete=False) as saved:
                saved_path = Path(saved.name)
            try:
                output.save(str(saved_path), garbage=3, deflate=True)
                saved_path.replace(out_path)
            finally:
                saved_path.unlink(missing_ok=True)

    for row in row_reports:
        for voice in row['voices']:
            voice['line_count'] = len(voice['lines'])
        row['rendered'] = (len(row['voices']) == len(tracks)
                           and all(v['rendered'] and v['lines'] for v in row['voices']))
    placed = sum(row['rendered'] for row in row_reports)
    missing = len(staves) - placed
    fallback_rows = sum(any(v['placement'] in ('scaled', 'line') for v in row['voices'])
                        for row in row_reports)
    mismatches = sum(row['pdf_measures'] != row['musicxml_measures'] for row in row_reports)
    tie_review_count = sum(len(v['tie_review_measures']) for row in row_reports for v in row['voices'])
    outlined = any(p.get('source_text_outlined') for p in page_map)
    status = ('incomplete' if not staves or missing else
              'needs_review' if fallback_rows or mismatches or tie_review_count or outlined
              or len(score.parts) > 1 else 'rendered')

    def digest(path):
        hasher = hashlib.sha256()
        with path.open('rb') as source:
            for block in iter(lambda: source.read(1024 * 1024), b''):
                hasher.update(block)
        return hasher.hexdigest()

    report = {'status': status, 'scope': 'in_situ_rendering_not_omr_accuracy',
              'source_pdf': str(pdf_path), 'source_pdf_sha256': digest(pdf_path),
              'source_musicxml': str(mxl_path), 'source_musicxml_sha256': digest(mxl_path),
              'output_pdf': str(out_path), 'output_pdf_sha256': digest(out_path),
              'engine': 'spaced' if expand else 'stamp', 'layout': layout, 'pitch_mode': pitch_mode,
              'source_page_count': len({p['source_page'] for p in page_map}),
              'output_page_count': len(page_map), 'pages': page_map,
              'staff_count': len(staves), 'rendered_staff_count': placed,
              'incomplete_staff_count': missing, 'alignment_mismatch_count': mismatches,
              'placement_fallback_count': fallback_rows, 'tie_review_measure_count': tie_review_count,
              'musicxml_part_count': len(score.parts), 'rows': row_reports}
    _write_json_report(report_path, report)
    print(f'placed {placed}/{len(staves)} lines')
    print(f'[insitu] 逐行报告: {report_path}（{status}）')
    if missing:
        print('  ! 原位输出不完整：至少有谱表或声部未生成，请查看逐行报告。')
    return out_path


def main(argv: list[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(
        description='把简谱逐行贴到刻印五线谱 PDF 每行谱表正下方（矢量贴字，不重绘原谱）')
    ap.add_argument('score', help='源五线谱 PDF')
    ap.add_argument('musicxml', help='OMR 得到的 MusicXML')
    ap.add_argument('output', help='输出 PDF')
    ap.add_argument('--title', default=None, help='标题（默认取 PDF 文件名）')
    args = ap.parse_args(argv)
    out = build(Path(args.score), Path(args.musicxml), Path(args.output), title=args.title)
    print(f'INSITU {out} ({out.stat().st_size} bytes)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
