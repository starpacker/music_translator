"""In-place jianpu: typeset each engraved staff's jianpu directly beneath it.

Every jianpu line is produced by the **same** pipeline that makes the standalone
`梆笛.jianpu.pdf`: the project's jianpu-ly preprocessor feeds LilyPond, and the
resulting vector PDF is stamped into the whitespace between a staff and the next
one down.  That is what makes the two documents identical glyph for glyph -
same font, same em-dash rhythm marks, same octave dots - rather than merely
similar.

How the jianpu-ly source is assembled
-------------------------------------
`core.notation.jianpu` renders each measure with `jianpu_note_token` and
per-measure layout time signatures, using jianpu-ly grammar where a measure is
a token list and `-` means "hold", `'` raises an octave, `q`/`s`/`d` shorten the
note.  We reuse those two functions verbatim so the tokens are the reference
renderer's, and only differ in **how many measures go on one line**: instead of
the library's fixed 4, we emit one line per engraved staff, with the measures
distributed exactly as `align.py` sliced them.

Placement
---------
* Each slice is written as a one-line jianpu-ly source and rendered by LilyPond
  to a **single unbroken system** at the reference staff size (20 -> 10.96pt
  note digits, identical to `梆笛.jianpu.pdf`).  The rendered vector PDF is
  cropped to its ink and stamped at its **natural** size, left-aligned at the
  staff's own x0, so glyphs are the same size on every line - no per-line
  rescaling, which is what previously made the font grow/shrink from staff to
  staff.
* A wide ``line-width`` plus ``ragged-last = ##f`` guarantees the measures stay
  on one system; ``| |`` artefacts produced by empty bars are collapsed so
  LilyPond does not insert a spurious system break.
* `Y_OFFSET` is where the top of the jianpu ink sits below `staff.bottom`.
* `STAFF_SIZE` matches jianpu-ly's default of 20, which is what `梆笛.jianpu.pdf`
  is rendered at, so glyph sizes agree.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import shutil
import copy
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
    _get_voice_ids_in_part,
    _extract_part_repeat_barlines,
)
from core.render.jianpu_runner import _insert_repeat_bar_commands  # noqa: E402
from music21 import converter  # noqa: E402

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
    head.append('OctavesAfter')
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


def _render_ly_tile(ly_text: str, workdir: Path) -> tuple[Path, fitz.Rect]:
    """Render `ly_text` with LilyPond; return (pdf path, ink bbox) of the system."""
    ly_text = re.sub(r'#\(set-global-staff-size \d+\)',
                     f'#(set-global-staff-size {STAFF_SIZE})', ly_text)
    ly_text = _strip_title_and_tagline(ly_text)
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
    best = int(counts.min())
    choices = np.flatnonzero(counts == best)
    middle = (len(counts) - 1) / 2
    row = min(choices, key=lambda i: abs(i - middle))
    if best:
        print(f'  ! 无完全空白的插入位置，使用墨迹最少的横线（{best} 像素）')
    return (start + int(row)) / 2


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


def _place_cell(target_page, source_doc, clip: fitz.Rect, target: fitz.Rect) -> bool:
    """Fit a cell by changing blank gaps, preserving glyph proportions when possible."""
    delta = target.width - clip.width
    gaps = _blank_column_runs(source_doc[0], clip)
    if abs(delta) < 0.1:
        target_page.show_pdf_page(target, source_doc, 0, clip=clip, keep_proportion=False)
        return True
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


def _place_tile(target_page, tile: Path, ink: fitz.Rect, layout, top: float, count: int) -> str:
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
        for left, right, x0, x1 in zip(bars, bars[1:], targets, targets[1:]):
            clip = fitz.Rect(left, ink.y0, right, ink.y1)
            target = fitz.Rect(x0, top, x1, top + ink.height)
            unscaled = _place_cell(target_page, src, clip, target) and unscaled
        if ink.x1 - bars[-1] > 0.2:
            clip = fitz.Rect(bars[-1], ink.y0, ink.x1, ink.y1)
            target_page.show_pdf_page(fitz.Rect(targets[-1], top, targets[-1] + clip.width, top + ink.height),
                                      src, 0, clip=clip, keep_proportion=False)
        return 'aligned' if unscaled else 'scaled'


def build(pdf_path: Path, mxl_path: Path, out_path: Path,
          title: Optional[str] = None, expand: bool = False) -> Path:
    staves, slices = align_from_files(pdf_path, mxl_path)

    score = converter.parse(str(mxl_path))
    tonic, key_header = _get_score_key_tonic(score)
    part = score.parts[0] if score.parts else None
    voice_ids = _get_voice_ids_in_part(part) if part is not None else []
    if len(voice_ids) > 1:
        tracks = [extract_jianpu_measures(score, tonic, _part=part, _voice_id=vid,
                                          _multi_voice_mode=True, _is_primary_voice=(index == 0))
                  for index, vid in enumerate(voice_ids)]
        print(f'[insitu] 检测到 {len(tracks)} 个声部，逐行分别排版。')
        if not expand:
            print('[insitu] 多声部需要更多行间空间；建议使用 spaced 模式并校对。')
    else:
        tracks = [extract_jianpu_measures(score, tonic)]
    if len(score.parts) > 1:
        print('[insitu] 检测到多个乐器分部；当前原位对齐仅使用第一个分部，请校对。')
    repeat_bars = _extract_part_repeat_barlines(score.parts[0]) if score.parts else {}
    if title is None:
        title = pdf_path.stem

    doc = fitz.open(str(pdf_path))
    tmpdir = Path(tempfile.mkdtemp(prefix='insitu_ly_'))
    placed = 0
    fallback_rows = 0
    incomplete_voice_rows = 0
    tiles = {}

    for sl, layout in zip(slices, staves):
        staff = layout.staff
        line_tiles = []
        for voice_index, (measures, time_sig) in enumerate(tracks):
            line_measures = _line_slur_continuations(measures, sl.omr_start, sl.omr_end)
            chunk = _with_time_signatures(
                _measure_texts(line_measures, time_sig), 0,
                layout_time_tokens(line_measures, 64),
            )
            if not chunk:
                continue
            line_key = key_header if placed == 0 and voice_index == 0 else ''
            try:
                workdir = tmpdir / f'sys_{sl.staff_index}_voice_{voice_index}'
                workdir.mkdir(parents=True, exist_ok=True)
                ly_text = _run_jianpu_ly(
                    _jianpu_ly_source([chunk], line_key, time_sig, title), workdir)
                local_repeats = {index - sl.omr_start: info for index, info in repeat_bars.items()
                                 if sl.omr_start <= index < sl.omr_end}
                ly_text = _insert_repeat_bar_commands(ly_text, local_repeats)
                tile_pdf, ink = _render_ly_tile(ly_text, workdir)
                line_tiles.append((tile_pdf, ink, len(chunk)))
            except Exception as exc:                   # keep the rest of the run
                print(f'  ! staff {sl.staff_index} voice {voice_index + 1}: {exc}')
        if not line_tiles:
            continue
        if len(line_tiles) != len(tracks):
            incomplete_voice_rows += 1

        if expand:
            tiles[sl.staff_index] = line_tiles
            placed += 1
            continue

        page = doc[staff.page - 1]
        top = staff.bottom + Y_OFFSET
        for tile_pdf, ink, count in line_tiles:
            placement = _place_tile(page, tile_pdf, ink, layout, top, count)
            if placement != 'aligned':
                fallback_rows += 1
                print(f'  ! 第 {staff.page} 页谱表 {sl.staff_index + 1}: 对齐回退 ({placement})')
            top += ink.height + 9
        placed += 1

    if expand:
        output = fitz.open()
        for page_index, page in enumerate(doc):
            page_rows = [(i, layout) for i, layout in enumerate(staves)
                         if layout.staff.page == page_index + 1 and i in tiles]
            insertions = []
            for row, (index, layout) in enumerate(page_rows):
                staff = layout.staff
                next_top = page_rows[row + 1][1].staff.top if row + 1 < len(page_rows) else page.rect.height
                low = min(staff.bottom + 6, next_top - 1)
                high = min(next_top - 5, staff.bottom + 55)
                cut = _blank_cut(page, low, max(low + 1, high))
                overrides = (_tempo_cut_overrides(page, low, cut, next_top)
                             if row + 1 < len(page_rows) else [])
                line_tiles = tiles[index]
                total_ink = sum(ink.height for _, ink, _ in line_tiles) + 9 * (len(line_tiles) - 1)
                spread = max([0.0] + [cut - y for _, _, y in overrides])
                insertions.append((cut, max(34.0, total_ink + 18.0) + spread,
                                   index, layout, line_tiles, None, None, overrides))
            added = sum(item[1] for item in insertions)
            target_page = output.new_page(width=page.rect.width, height=page.rect.height + added)
            _copy_with_column_cuts(target_page, doc, page_index, insertions)
            shift = 0.0
            for cut, height, index, layout, line_tiles, _, _, _ in insertions:
                top = cut + shift + 9
                for tile, ink, count in line_tiles:
                    placement = _place_tile(target_page, tile, ink, layout, top, count)
                    if placement != 'aligned':
                        fallback_rows += 1
                        print(f'  ! 第 {page_index + 1} 页谱表 {index + 1}: 对齐回退 ({placement})')
                    top += ink.height + 9
                shift += height
        output.save(str(out_path), garbage=3, deflate=True)
        output.close()
    else:
        doc.save(str(out_path), garbage=3, deflate=True)
    doc.close()
    shutil.rmtree(tmpdir, ignore_errors=True)
    print(f'placed {placed}/{len(staves)} lines')
    if fallback_rows:
        print(f'  ! {fallback_rows} 行因分隔线不明确或空间不足，使用整体放置或压缩；请校对。')
    if incomplete_voice_rows:
        print(f'  ! {incomplete_voice_rows} 个谱表存在未生成的声部，结果不完整；请校对。')
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
