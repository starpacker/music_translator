"""Generate the project logo (SVG + PNG) and the GitHub social preview."""
from __future__ import annotations

from pathlib import Path

import matplotlib
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.ttLib import TTFont

LATIN = Path(matplotlib.get_data_path()) / 'fonts' / 'ttf' / 'DejaVuSans-Bold.ttf'
CJK = Path('/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf')
CACHE: dict[Path, TTFont] = {}


def _font(path: Path) -> TTFont:
    if path not in CACHE:
        CACHE[path] = TTFont(str(path))
    return CACHE[path]


def _face(font: Path):
    tt = _font(font)
    return tt['head'].unitsPerEm, tt.getBestCmap(), tt.getGlyphSet()


def text_paths(text: str, size: float, tracking: float = 0.0,
               font: Path = LATIN):
    """Lay out `text` and return ([(svg_d, x_offset, advance)], total_width).

    Coordinates are in output units with the baseline at y=0 and y pointing up.
    Glyphs missing from `font` fall back to the CJK (or Latin) face.
    """
    faces = [_face(font)]
    for alt in (CJK, LATIN):
        if alt != font and alt.is_file():
            faces.append(_face(alt))
    out, pen_x = [], 0.0
    for ch in text:
        if ch == ' ':
            pen_x += size * 0.28 + tracking * size
            continue
        hit = None
        for upm, cmap, gs in faces:
            name = cmap.get(ord(ch))
            if name is not None:
                hit = (upm, cmap, gs, name)
                break
        if hit is None:
            print(f'warning: no glyph for {ch!r} (U+{ord(ch):04X})')
            pen_x += size * 0.5
            continue
        upm, cmap, gs, name = hit
        scale = size / upm
        sp = SVGPathPen(gs)
        gs[name].draw(sp)
        adv = gs[name].width * scale
        out.append((sp.getCommands(), pen_x, adv, upm))
        pen_x += adv + tracking * size
    total = pen_x - tracking * size if text else 0.0
    return out, total


def placed(chars, x0, baseline, size, fill, font=LATIN, tracking=0.0):
    paths, width = text_paths(chars, size, tracking, font)
    out = []
    for d, x, _, upm in paths:
        k = size / upm
        out.append(f'<path d="{d}" fill="{fill}" '
                   f'transform="translate({x0 + x:.2f},{baseline:.2f}) '
                   f'scale({k:.6f},{-k:.6f})"/>')
    return out, width


# ── palette ─────────────────────────────────────────────────────────────────
BG = '#0E1626'
PANEL = '#16203A'
RULE = '#26344E'
AMBER = '#F7B955'
CORAL = '#FF7A5C'
WHITE = '#F2F5FA'
MUTED = '#8FA1BC'


def staff(x, y, width, gap=22.0, thick=4.5, colour=RULE, radius=2.2):
    return [f'<rect x="{x}" y="{y + i * gap}" width="{width}" height="{thick}" '
            f'rx="{radius}" fill="{colour}"/>' for i in range(5)]


def digits_group(centre_x, baseline, size, dot=True, dash=True):
    """The "123" motif: three jianpu digits with an octave dot and a hold dash."""
    paths, width = text_paths('123', size, tracking=0.10)
    x0 = centre_x - width / 2
    parts = [f'<path d="{d}" fill="{AMBER}" '
             f'transform="translate({x0 + x:.2f},{baseline:.2f}) '
             f'scale({size / upm:.6f},{-size / upm:.6f})"/>'
             for d, x, _, upm in paths]
    cap = size * 0.73          # DejaVu digits top out around 0.73 em
    if dot:
        parts.append(f'<circle cx="{x0 + paths[0][2] / 2:.2f}" '
                     f'cy="{baseline - cap - size * 0.13:.2f}" '
                     f'r="{size * 0.065:.2f}" fill="{CORAL}"/>')
    if dash:
        d, dx, adv, _ = paths[2]
        w = adv * 0.78
        parts.append(f'<rect x="{x0 + dx + (adv - w) / 2:.2f}" '
                     f'y="{baseline + size * 0.10:.2f}" width="{w:.2f}" '
                     f'height="{size * 0.05:.2f}" rx="{size * 0.025:.2f}" fill="{CORAL}"/>')
    return parts


def logo_svg(size: int = 512) -> str:
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
             f'viewBox="0 0 512 512">',
             f'<rect width="512" height="512" rx="116" fill="{BG}"/>']
    parts += staff(62, 96, 388)
    parts.append(f'<rect x="62" y="88" width="4.5" height="112" rx="2.2" fill="{RULE}"/>')
    parts += digits_group(256, 400, 176)
    parts.append('</svg>')
    return '\n'.join(parts)


def social_svg(w: int = 1280, h: int = 640) -> str:
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
             f'viewBox="0 0 {w} {h}">',
             f'<rect width="{w}" height="{h}" fill="{BG}"/>']

    # ── logo tile ──
    s, ox, oy = 0.72, 96, (h - 512 * 0.72) / 2
    parts.append(f'<g transform="translate({ox},{oy}) scale({s})">')
    parts.append(f'<rect width="512" height="512" rx="116" fill="{PANEL}"/>')
    parts += staff(62, 96, 388)
    parts.append('<rect x="62" y="88" width="4.5" height="112" rx="2.2" fill="#2F405E"/>')
    parts += digits_group(256, 400, 176)
    parts.append('</g>')

    # ── wordmark ──
    tx = 96 + 512 * 0.72 + 76
    word, ww = placed('jianpu', tx, 300, 148, WHITE)
    parts += word
    sub, sw = placed('五线谱・原位简谱', tx, 386, 52, AMBER, font=CJK)
    parts += sub
    parts.append(f'<rect x="{tx}" y="424" width="{max(sw, 260):.0f}" height="5" rx="2.5" fill="{CORAL}"/>')
    tag, tw = placed('PDF  →  OMR  →  LILYPOND', tx, 492, 30, MUTED, tracking=0.16)
    parts += tag
    parts.append('</svg>')
    return '\n'.join(parts)


def rasterise(svg: Path, png: Path, dpi: int = 72) -> tuple[int, int]:
    import pymupdf
    doc = pymupdf.open(str(svg))
    pix = doc[0].get_pixmap(dpi=dpi, alpha=False)
    pix.save(str(png))
    return pix.width, pix.height


if __name__ == '__main__':
    out = Path(__file__).resolve().parent.parent / 'assets'
    out.mkdir(parents=True, exist_ok=True)
    jobs = [('logo.svg', logo_svg(512)), ('social-preview.svg', social_svg())]
    for name, svg in jobs:
        (out / name).write_text(svg, encoding='utf-8')
        size = rasterise(out / name, out / name.replace('.svg', '.png'))
        print(f'{name} -> {size[0]}x{size[1]}')
    print('ok')
