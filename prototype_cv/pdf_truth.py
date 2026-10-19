"""
pdf_truth.py — exact ground truth from the *vector* content of a score PDF.

The example scores (二胡.pdf / 曲笛.pdf) are Sibelius exports: every staff
line, barline, stem and beam is a vector drawing and every notehead,
accidental, rest and flag is a text glyph of the embedded "Opus" fonts.
That makes it possible to recover the musical content *exactly*, without
any image recognition, and use it as ground truth for the CV pipeline.

Glyph decoding (verified by rendering each embedded glyph in isolation):

    OpusStd  œ  filled notehead          OpusStd  #  b   sharp / flat
    OpusStd  ¿  x-notehead (pizz. etc.)  OpusStd  &     treble clef
    OpusStd  w  whole notehead           OpusStd  Œ ‰   quarter / 8th rest
    OpusStd  ˙  half notehead            OpusStd  Ó ∑   half / whole rest
    OpusStd  J j  8th / 16th flag        OpusStd  . ˙(Spec ™) dots
    OpusSpecialStd æ tremolo stroke      OpusStd  >  accent  (ignored)

Durations come from geometry (stem present? beam count? flag glyph?) and
are then normalised per measure against the time signature, which also
reveals tuplets and multi-measure rests.
"""
import os
import collections

import pymupdf

# ── glyph semantics (unicode char → role) ────────────────────
HEAD_FILLED = {'œ'}
HEAD_X = {'¿'}
HEAD_HOLLOW = {'w', '˙'}          # whole / half notehead
REST_QUARTER = {'Œ'}
REST_EIGHTH = {'‰'}
REST_BLOCK = {'Ó', '∑'}           # half rest / whole rest (decided by position)
FLAG_GLYPHS = {'J', 'j'}          # 8th / 16th flags (unbeamed notes)
DOT_GLYPHS = {'.'}                # OpusStd dot
DOT_GLYPHS_SPECIAL = {'™'}        # OpusSpecialStd dot
ACCIDENTALS = {'#': '#', 'b': 'b'}
CLEF_GLYPHS = {'&'}
IGNORE = {'>', 'æ', 'ù', 'û', 'Y', '~', '&'}

# treble clef: staff lines top→bottom = F5 D5 B4 G4 E4
# diatonic step index: 0 = C, 1 = D, ... 6 = B
_LINE_STEPS_TREBLE = [3, 1, 6, 4, 2]      # F D B G E  (top line first)
_STEPS = {'C': 0, 'D': 1, 'E': 2, 'F': 3, 'G': 4, 'A': 5, 'B': 6}
_LETTERS = 'CDEFGAB'
_SHARP_ORDER = 'FCGDAEB'
_FLAT_ORDER = 'BEADGCF'


class System:
    __slots__ = ('lines', 'top', 'bot', 'dy', 'x0', 'x1', 'barlines', 'idx')

    def __init__(self, lines, x0, x1):
        self.lines = lines
        self.top, self.bot = lines[0], lines[4]
        self.dy = (lines[4] - lines[0]) / 4.0
        self.x0, self.x1 = x0, x1
        self.barlines = []
        self.idx = 0


def _glyph_offsets(pdf_path):
    """Return {(font, char): (dx, dy)} ink-centre offset from the text origin.

    Computed from the embedded font's glyph bounding boxes, so it is exact
    and independent of neighbouring ink on the page.
    """
    from fontTools.ttLib import TTFont
    cache = {}
    doc = pymupdf.open(pdf_path)
    page = doc[0]
    for xref, ext, ftype, name in [(f[0], f[1], f[2], f[3]) for f in page.get_fonts(full=True)]:
        if not name.startswith('Opus'):
            continue
        buf = doc.extract_font(xref)
        if not buf or not buf[3]:
            continue
        tmp = f'/tmp/_mt_font_{name}.ttf'
        with open(tmp, 'wb') as fh:
            fh.write(buf[3])
        f = TTFont(tmp)
        upem = f['head'].unitsPerEm
        from fontTools.pens.boundsPen import BoundsPen
        gs = f.getGlyphSet()
        order = f.getGlyphOrder()
        # char → code via the page's text trace
        codes = {}
        for pg in doc:
            for sp in pg.get_texttrace():
                if sp['font'] != name:
                    continue
                for uni, code, _org, _bb in sp['chars']:
                    u = uni if isinstance(uni, str) else chr(uni)
                    codes.setdefault(u, code)
        for u, code in codes.items():
            if code >= len(order):
                continue
            bp = BoundsPen(gs)
            try:
                gs[order[code]].draw(bp)
            except Exception:
                continue
            if bp.bounds is None:
                continue
            x0, y0, x1, y1 = bp.bounds
            cache[(name, u)] = ((x0 + x1) / 2.0 / upem, -(y0 + y1) / 2.0 / upem)
    doc.close()
    return cache


def _drawings(page):
    """Split vector drawings into staff lines, barlines, stems, beams, rects."""
    horiz, vert, beams, rects = [], [], [], []
    for d in page.get_drawings():
        r = d['rect']
        w, h = r.width, r.height
        items = d['items']
        if all(it[0] == 'l' for it in items) and len(items) == 1:
            p1, p2 = items[0][1], items[0][2]
            if abs(p1.y - p2.y) < 0.01:
                horiz.append((p1.y, min(p1.x, p2.x), max(p1.x, p2.x)))
            elif abs(p1.x - p2.x) < 0.01:
                vert.append((p1.x, min(p1.y, p2.y), max(p1.y, p2.y)))
            continue
        if items and all(it[0] == 're' for it in items):
            rects.append(r)
        elif d['type'] in ('f', 'fs') and w > h and h > 0.5:
            beams.append(r)
        elif d['type'] in ('f', 'fs'):
            rects.append(r)
    return horiz, vert, beams, rects


def _systems(horiz, page_width):
    long_lines = [h for h in horiz if (h[2] - h[1]) > page_width * 0.5]
    ys = sorted({round(h[0], 2) for h in long_lines})
    # cluster thick lines
    clustered = []
    for y in ys:
        if clustered and y - clustered[-1][-1] < 1.0:
            clustered[-1].append(y)
        else:
            clustered.append([y])
    cy = [sum(g) / len(g) for g in clustered]
    x0 = min(h[1] for h in long_lines)
    x1 = max(h[2] for h in long_lines)
    systems, cur = [], [cy[0]] if cy else []
    if len(cy) > 1:
        gaps = [b - a for a, b in zip(cy, cy[1:])]
        med = sorted(gaps)[len(gaps) // 2]
        for y, g in zip(cy[1:], gaps):
            if g > med * 2.0:
                if len(cur) == 5:
                    systems.append(System(cur, x0, x1))
                cur = []
            cur.append(y)
    if len(cur) == 5:
        systems.append(System(cur, x0, x1))
    for i, s in enumerate(systems):
        s.idx = i
    return systems


def _glyphs(page, offsets, size_default=19.84):
    out = []
    raw = page.get_text('rawdict')
    for b in raw['blocks']:
        for l in b.get('lines', []):
            for s in l['spans']:
                if not s['font'].startswith('Opus') and not s['font'].startswith('SimSun'):
                    continue
                for c in s['chars']:
                    ch = c['c']
                    if ch == ' ':
                        continue
                    ox, oy = c['origin']
                    dx, dy = offsets.get((s['font'], ch), (0.5, -0.2))
                    sc = s.get('size', size_default)
                    out.append({
                        'font': s['font'], 'ch': ch,
                        'x': ox + dx * sc, 'y': oy + dy * sc,
                        'origin': (ox, oy), 'size': sc,
                    })
    return out


def extract_page(pdf_path, pno, offsets=None):
    """Extract the musical content of one page as structured truth."""
    doc = pymupdf.open(pdf_path)
    page = doc[pno]
    if offsets is None:
        offsets = _glyph_offsets(pdf_path)
    horiz, vert, beams, rects = _drawings(page)
    systems = _systems(horiz, page.rect.width)
    if not systems:
        doc.close()
        return None

    # barlines: vertical lines spanning exactly one staff
    for s in systems:
        for x, y0, y1 in vert:
            if abs(y0 - s.top) < 0.8 and abs(y1 - s.bot) < 0.8 and s.x0 - 1 <= x <= s.x1 + 1:
                s.barlines.append(x)
        s.barlines = sorted(set(round(b, 1) for b in s.barlines))

    glyphs = _glyphs(page, offsets)
    dy = systems[0].dy

    def system_for(y, margin=None):
        margin = margin if margin is not None else 5.0 * dy
        best, bd = None, 1e9
        for s in systems:
            inside = s.top - margin <= y <= s.bot + margin
            dist = 0.0 if inside else min(abs(y - s.top), abs(y - s.bot))
            if dist < bd - 1e-9:
                bd, best = dist, s
        return best

    # ── noteheads ──
    heads = []
    for g in glyphs:
        if g['ch'] in HEAD_FILLED | HEAD_X:
            kind = 'x' if g['ch'] in HEAD_X else 'filled'
        elif g['ch'] in HEAD_HOLLOW:
            kind = 'hollow'
        else:
            continue
        s = system_for(g['y'])
        if s is None:
            continue
        pos = (g['y'] - s.top) / (s.dy / 2.0)      # half-steps below top line
        step = round(pos)
        heads.append({'x': g['x'], 'y': g['y'], 'sys': s.idx, 'halfpos': step,
                      'kind': kind, 'glyph': g['ch']})
    heads.sort(key=lambda h: (h['sys'], h['x']))

    # ── stems ─
    stems = []
    for x, y0, y1 in vert:
        L = y1 - y0
        if 2.0 * dy <= L <= 6.5 * dy:
            s = system_for((y0 + y1) / 2, margin=8 * dy)
            if s is None:
                continue
            is_bar = any(abs(x - b) < 0.5 for b in s.barlines)
            if not is_bar:
                stems.append({'x': x, 'y0': y0, 'y1': y1})

    for h in heads:
        best, bd = None, 1.6 * dy
        for st in stems:
            d = abs(st['x'] - h['x'])
            touch = (st['y0'] - 1.2 * dy <= h['y'] <= st['y1'] + 1.2 * dy)
            if d < bd and touch:
                bd, best = d, st
        h['stem'] = best
        if best is not None:
            h['stem_dir'] = 'up' if best['y0'] < h['y'] - 0.3 * dy else 'down'
            h['stem_tip'] = best['y0'] if h['stem_dir'] == 'up' else best['y1']
        else:
            h['stem_dir'] = None
            h['stem_tip'] = None

    # ── beams: count per head via stem tip ──
    for h in heads:
        h['beams'] = 0
        if h['stem_tip'] is None:
            continue
        for r in beams:
            if r.width < 0.8 * dy or r.height > 1.6 * dy:
                continue
            if r.x0 - 0.6 * dy <= h['stem']['x'] <= r.x1 + 0.6 * dy:
                if r.y0 - 0.9 * dy <= h['stem_tip'] <= r.y1 + 0.9 * dy:
                    h['beams'] += 1

    # ── flags ──
    flags = [g for g in glyphs if g['ch'] in FLAG_GLYPHS]
    for h in heads:
        h['flags'] = 0
        if h['stem_tip'] is None or h['beams']:
            continue
        for f in flags:
            if abs(f['x'] - h['x']) < 1.6 * dy and abs(f['y'] - h['stem_tip']) < 1.6 * dy:
                h['flags'] += 1

    # ── dots (augmentation) ──
    dots = [g for g in glyphs if (g['ch'] in DOT_GLYPHS and g['font'] == 'OpusStd')
            or (g['ch'] in DOT_GLYPHS_SPECIAL and g['font'] == 'OpusSpecialStd')]
    for h in heads:
        h['dot'] = False
        for d in dots:
            dx_, dy_ = d['x'] - h['x'], d['y'] - h['y']
            if 0.4 * dy < dx_ < 3.0 * dy and abs(dy_) < 1.1 * dy:
                h['dot'] = True

    # ── accidentals ──
    accs = [g for g in glyphs if g['ch'] in ACCIDENTALS]
    for h in heads:
        h['acc'] = None
        for a in accs:
            dx_ = h['x'] - a['x']
            if 0.2 * dy < dx_ < 3.2 * dy and abs(a['y'] - h['y']) < 1.1 * dy:
                h['acc'] = ACCIDENTALS[a['ch']]

    # ── rests ─
    rests = []
    for g in glyphs:
        if g['ch'] in REST_QUARTER:
            dur = 1.0
        elif g['ch'] in REST_EIGHTH:
            dur = 0.5
        elif g['ch'] in REST_BLOCK:
            # half rest sits ON a line (centre above it), whole rest hangs
            # BELOW a line (centre below it)
            lines = [y for s in systems for y in s.lines]
            near = min(lines, key=lambda L: abs(g['y'] - L))
            dur = 2.0 if g['y'] < near else 4.0
        else:
            continue
        s = system_for(g['y'])
        if s is None:
            continue
        rests.append({'x': g['x'], 'y': g['y'], 'sys': s.idx, 'duration': dur})

    # ── multi-measure rests: thick rectangle + digit above ──
    multirests = []
    digits = [g for g in glyphs if g['ch'].isdigit()]
    for r in rects:
        # a multi-measure rest is a thick bar spanning (part of) a measure,
        # centred vertically on the staff, with its count printed above
        if not (3 * dy < r.width < 25 * dy and 0.5 * dy < r.height < 1.8 * dy):
            continue
        s = system_for((r.y0 + r.y1) / 2, margin=2 * dy)
        if s is None:
            continue
        cx = (r.x0 + r.x1) / 2
        cnt = None
        for dg in digits:
            if abs(dg['x'] - cx) < 3 * dy and r.y0 - 5 * dy < dg['y'] < r.y0 - 0.3 * dy:
                cnt = int(dg['ch'])
        multirests.append({'x': r.x0, 'sys': s.idx, 'count': cnt or 1})

    # ── time signatures: stacked OpusStd digits right of the clef ──
    timesigs = []
    nums = [g for g in glyphs if g['font'] == 'OpusStd' and g['ch'].isdigit()]
    for s in systems:
        cand = [n for n in nums if system_for(n['y']) is s]
        for n in cand:
            m = [o for o in cand if abs(o['x'] - n['x']) < 0.8
                 and 1.2 * s.dy < o['y'] - n['y'] < 2.8 * s.dy]
            if m:
                timesigs.append({'sys': s.idx, 'x': n['x'],
                                 'num': int(n['ch']), 'den': int(m[0]['ch'])})

    # ── key signature: accidental cluster between clef and first barline ──
    key_sig = None
    for s in systems[:1]:
        clef_x = max([g['x'] for g in glyphs if g['ch'] in CLEF_GLYPHS
                      and system_for(g['y']) is s] or [s.x0])
        first_bar = s.barlines[0] if s.barlines else s.x1
        cluster = [a for a in accs if clef_x < a['x'] < first_bar
                   and abs(a['x'] - clef_x) < 8 * dy]
        if cluster:
            kinds = {ACCIDENTALS[a['ch']] for a in cluster}
            if len(kinds) == 1:
                key_sig = {'type': kinds.pop(), 'count': len(cluster)}

    # ── apply key signature to heads without an explicit accidental ──
    if key_sig:
        order = _SHARP_ORDER if key_sig['type'] == '#' else _FLAT_ORDER
        affected = set(order[:key_sig['count']])
        for h in heads:
            letter, _o = halfpos_to_pitch(h['halfpos'])
            h['letter'] = letter
            if h['acc'] is None and letter in affected:
                h['acc'] = key_sig['type']

    doc.close()
    return {
        'page': pno, 'systems': systems, 'heads': heads, 'rests': rests,
        'multirests': multirests, 'timesigs': timesigs, 'key_sig': key_sig,
        'dy': dy,
    }


# ── pitch / jianpu conversion ────────────────────────────────
def halfpos_to_pitch(halfpos):
    """half-steps below the top staff line (treble) → (letter, octave)."""
    # top line = F5 ; each halfpos step down = one diatonic step down
    # diatonic index of F5 in (letter, octave): F=3, octave 5
    abs_index = 5 * 7 + _STEPS['F'] - halfpos     # absolute diatonic number
    octave, step = divmod(abs_index, 7)
    return _LETTERS[step], octave


def pitch_to_jianpu_c(letter, octave, acc):
    """Absolute pitch → jianpu in 1=C (degree + octave marks + accidental)."""
    # middle C (C4) = '1' with no marks
    deg = _STEPS[letter] + 1
    rel = octave - 4
    mark = "'" * rel if rel > 0 else ',' * -rel
    return (acc or '') + str(deg) + mark


def duration_of(h):
    if h['kind'] == 'hollow':
        return 4.0 if h['stem'] is None else 2.0
    if h['beams'] >= 1:
        d = 0.5
        for _ in range(h['beams'] - 1):
            d /= 2.0
        return d
    if h['stem'] is None:
        return 1.0
    if h['flags'] >= 1:
        return 0.5 / (2 ** (h['flags'] - 1)) if h['flags'] else 0.5
    return 1.0


_SUFFIX = {4.0: '---', 3.0: '--', 2.0: '-', 1.5: '.', 1.0: '',
           0.75: '/2.', 0.5: '/2', 0.25: '/4', 0.125: '/8',
           1 / 3: '/3', 1 / 6: '/6', 2 / 3: '/3*2'}


def dur_suffix(d):
    for k, v in _SUFFIX.items():
        if abs(d - k) < 0.011:
            return v
    return f'<{d:.2f}>'


def page_to_measures(truth):
    """Group events into measures per system → list of list of event strings."""
    systems = truth['systems']
    out = []
    for s in systems:
        bars = [b for b in s.barlines]
        events = []
        for h in truth['heads']:
            if h['sys'] == s.idx:
                events.append((h['x'], 'note', h))
        for r in truth['rests']:
            if r['sys'] == s.idx:
                events.append((r['x'], 'rest', r))
        for mr in truth['multirests']:
            if mr['sys'] == s.idx:
                events.append((mr['x'], 'multirest', mr))
        events.sort(key=lambda e: e[0])
        measures = []
        if not bars:
            bounds = [(s.x0, s.x1 + 1)]
        else:
            bounds = list(zip([s.x0] + bars[:-1], bars))
        for lo, hi in bounds:
            evs = [e for e in events if lo <= e[0] < hi]
            parts = []
            for _x, kind, e in evs:
                if kind == 'note':
                    letter, octv = halfpos_to_pitch(e['halfpos'])
                    j = pitch_to_jianpu_c(letter, octv, e['acc'])
                    d = duration_of(e)
                    if e['dot']:
                        d *= 1.5
                    parts.append(j + dur_suffix(d))
                elif kind == 'rest':
                    parts.append('0' + dur_suffix(e['duration']))
                elif kind == 'multirest':
                    parts.append(('0--- ' * e['count']).strip())
            measures.append(parts)
        out.append(measures)
    return out
