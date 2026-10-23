"""In-place jianpu: typeset each engraved staff's jianpu directly beneath it.

Every jianpu line is produced by the **same** pipeline that makes the standalone
`梆笛.jianpu.pdf`: the project's jianpu-ly preprocessor feeds LilyPond, and the
resulting vector PDF is stamped into the whitespace between a staff and the next
one down.  That is what makes the two documents identical glyph for glyph -
same font, same em-dash rhythm marks, same octave dots - rather than merely
similar.

How the jianpu-ly source is assembled
-------------------------------------
`core.notation.jianpu` renders each measure with `jianpu_note_token` +
`pad_measure_to_bar`, and emits the ordinary jianpu-ly grammar where a measure is
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
from pathlib import Path
from typing import Optional

import pymupdf as fitz

sys.path.insert(0, str(Path(__file__).resolve().parent))        # align.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # _env.py
from _env import ensure_sumisora, jianpu_ly_script, lilypond_exe, python_exe  # noqa: E402

ensure_sumisora()

from core.notation.jianpu import (  # noqa: E402
    jianpu_note_token,
    pad_measure_to_bar,
    _parse_bar_units,
    _parse_anacrusis_units,
    extract_jianpu_measures,
    _get_score_key_tonic,
)
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
    bar_units = _parse_bar_units(time_sig)
    pickup_units = _parse_anacrusis_units(time_sig)
    out = []
    for i, m in enumerate(measures):
        units = pickup_units if (i == 0 and pickup_units is not None) else bar_units
        out.append(' '.join(jianpu_note_token(n) for n in pad_measure_to_bar(m, units)))
    return out


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
    if time_sig:
        pass  # dropped: jianpu-ly prints \time <sig> (an Emmentaler "4/4"
              # glyph) on the staff, which is unrelated to jianpu and shows up
              # below every engraved staff.  Bars stay correct because we pad
              # every measure to the full signature ourselves.
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

def build(pdf_path: Path, mxl_path: Path, out_path: Path,
          title: Optional[str] = None) -> Path:
    staves, slices = align_from_files(pdf_path, mxl_path)

    score = converter.parse(str(mxl_path))
    tonic, key_header = _get_score_key_tonic(score)
    measures, time_sig = extract_jianpu_measures(score, tonic)
    per_measure = _measure_texts(measures, time_sig)
    if title is None:
        title = pdf_path.stem

    doc = fitz.open(str(pdf_path))
    tmpdir = Path(tempfile.mkdtemp(prefix='insitu_ly_'))
    placed = 0

    for sl, layout in zip(slices, staves):
        staff = layout.staff
        chunk = per_measure[sl.omr_start:sl.omr_end]
        if not chunk:
            continue
        # `1=C` appears once, on the first rendered line only - exactly as in
        # 梆笛.jianpu.pdf, instead of above every system.
        line_key = key_header if placed == 0 else ''
        try:
            workdir = tmpdir / f'sys_{sl.staff_index}'
            workdir.mkdir(parents=True, exist_ok=True)
            ly_text = _run_jianpu_ly(
                _jianpu_ly_source([chunk], line_key, time_sig, title), workdir)
            tile_pdf, ink = _render_ly_tile(ly_text, workdir)
        except Exception as exc:                       # keep the rest of the run
            print(f'  ! staff {sl.staff_index}: {exc}')
            continue

        page = doc[staff.page - 1]
        # Natural size, left-aligned at the staff's own left edge.  No rescaling,
        # so every line keeps the reference 10.96pt glyph size.
        top = staff.bottom + Y_OFFSET
        target = fitz.Rect(staff.x0, top, staff.x0 + ink.width, top + ink.height)

        src = fitz.open(str(tile_pdf))
        page.show_pdf_page(target, src, 0, clip=ink, overlay=True)
        src.close()
        placed += 1

    doc.save(str(out_path), garbage=3, deflate=True)
    doc.close()
    print(f'placed {placed}/{len(staves)} lines')
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
