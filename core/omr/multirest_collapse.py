# core/omr/multirest_collapse.py — 方案D：多小节休止（粗实线）折叠为单小节
"""把五线谱里的「多小节休止」符号（粗实线 + 上方数字，如 ``|—41—|``）
在 MusicXML 中折叠成 **一个小节** 的全小节休止符。

问题
----
homr 的 Transformer 词表只有 ``rest_{c}m``（c ∈ 2..10），根本输不出 41 / 18 / 26
这类大数：它把一个 41 小节的休止识别成一串互相独立的休止小节。于是简谱的小节数
（145+）永远对不上五线谱的小节数。

方案D
-----
不展开、也不去数清 N。只要在原 PDF 里识别到「粗实线」这个排版符号，就让该符号
在 MusicXML 里只占 **一个** 小节（输出 ``|0---|``）。五线谱与简谱于是天然在
小节「数量」上一一对应。

识别依据（全部来自 PDF 矢量层，不依赖位图 / OCR）
-------------------------------------------------
1. 谱表系统：长细横线（宽 > 300pt，高 < 1pt），按 y 间距 ≤ 7pt 聚成 5 线一组。
2. 小节线：竖线（宽 < 0.6pt，高 ≈ 谱表高，顶端对齐首线）。
3. **多小节休止粗实线**：压在第三线（中间线）上、线宽 ≈ 2.64pt（谱线 0.48pt 的
   5.5 倍）、长度 ≥ 20pt 的横线 —— 全谱唯一「压在中间线上」的粗线。
4. 单小节休止记号：第 2 线上的 5~9pt 短横线（whole-rest 的方块笔画）。

把每个系统的「排版小节槽位类型序列」（M/W/N）与 MusicXML 的
「小节类型序列」（R/N）做分段对齐，就能确定每个排版小节对应哪些 MusicXML 小节。
"""
from __future__ import annotations

import os
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterator, Optional

# ── 几何阈值（PDF 用户空间 pt） ──────────────────────────────────────────────
_STAFF_MIN_WIDTH = 300.0      # 谱线最小长度
_STAFF_MAX_HEIGHT = 1.0       # 谱线最大厚度
_SYSTEM_GAP = 7.0             # 同一系统内相邻谱线的最大 y 间距
_BARLINE_MAX_WIDTH = 0.6      # 小节线最大厚度
_BARLINE_HEIGHT_TOL = 0.7     # 小节线高度与谱表高的容差
_THICK_MIN_WIDTH = 20.0       # 粗实线最小长度
_THICK_MIN_LW = 1.4           # 粗实线最小线宽（谱线为 0.48）
_THICK_Y_TOL = 1.5            # 粗实线 y 与中间线的容差
_WHOLE_REST_MIN_W = 5.0       # 单小节休止记号横线宽度的下限
_WHOLE_REST_MAX_W = 9.0       # ……上限
_MIN_SLOT = 25.0              # 小于此宽度的小节槽位视为重复 / 桩小节线
_SLOT_MEDIAN_RATIO = 0.25     # 或中位宽度的 25%

_MAX_MEASURES = 20000         # 防御性上限


def _load_pymupdf():
    try:
        import pymupdf  # type: ignore
        return pymupdf
    except Exception:  # pragma: no cover - 兼容旧版 PyMuPDF
        try:
            import fitz as pymupdf  # type: ignore
            return pymupdf
        except Exception:
            return None


# ─────────────────────────────────────────────────────────────────────────────
# PDF 矢量层 → 每系统的排版小节槽位类型
# ─────────────────────────────────────────────────────────────────────────────

def page_system_slot_kinds(page, *, _geometry=None) -> list[list[str]]:
    """返回该 PDF 页每个谱表系统的小节槽位类型序列。

    元素取值：``'M'`` = 多小节休止粗实线，``'W'`` = 单小节休止记号，
    ``'N'`` = 有音符的小节。
    """
    drawings = page.get_drawings()
    staff = [
        d for d in drawings
        if d['type'] == 's'
        and d['rect'].width > _STAFF_MIN_WIDTH
        and d['rect'].height < _STAFF_MAX_HEIGHT
    ]
    if not staff:
        return []

    ys = sorted({round(d['rect'].y0, 2) for d in staff})
    groups: list[list[float]] = []
    cur = [ys[0]]
    for y in ys[1:]:
        if y - cur[-1] <= _SYSTEM_GAP:
            cur.append(y)
        else:
            groups.append(cur)
            cur = [y]
    groups.append(cur)

    systems: list[list[str]] = []
    for lines in groups:
        if len(lines) < 5:
            continue
        top, bottom, middle = lines[0], lines[-1], lines[2]
        height = bottom - top
        top_line = next((d for d in staff if abs(d['rect'].y0 - top) < 0.6), None)
        if top_line is None:
            continue
        left, right = top_line['rect'].x0, top_line['rect'].x1

        # 小节线：顶端与首线对齐、高度≈谱表高、线宽 < 1pt
        raw = sorted({
            round(d['rect'].x0, 2) for d in drawings
            if d['type'] == 's'
            and d['rect'].width < _BARLINE_MAX_WIDTH
            and abs(d['rect'].y0 - top) <= 0.6
            and abs(d['rect'].height - height) <= _BARLINE_HEIGHT_TOL
            and (d.get('width') or 1.0) < 1.0
        })
        # 同一位置常有双 / 三重描边（终止线、装饰重线），8pt 内合并
        merged: list[float] = []
        for v in raw:
            if not merged or v - merged[-1] > 8.0:
                merged.append(v)

        # 系统开头的 ``|:`` 反复记号：粗线（线宽 ≈2.64）画在左边距内 26pt 处，
        # 细线紧随其后。它不是小节线（那时谱号缩进还没走完），若当成一个
        # 小节会在系统最前面切出 26pt 的假槽位，简谱格就整体错位一格。
        # 判定条件与 score_align.detect_staff_systems 保持一致：贴着左边距
        # 且旁边确实有一条谱表高的粗线。
        start_repeat = [
            d['rect'].x0 for d in drawings
            if d['type'] == 's'
            and (d.get('width') or 0) >= 1.5
            and d['rect'].width < 1.2
            and d['rect'].height > 8.0
            and abs(d['rect'].y0 - top) < 1.0
        ]
        if (len(merged) > 1
                and merged[0] - left < 45.0
                and any(abs(tx - merged[0]) < 8.0 for tx in start_repeat)):
            merged = merged[1:]

        edges = [left] + [v for v in merged if left + 2 < v < right - 2] + [right]
        slots = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
        widths = [b - a for a, b in slots]
        median = statistics.median(widths) if widths else 0.0
        min_width = max(_MIN_SLOT, _SLOT_MEDIAN_RATIO * median)
        slots = [(a, b) for (a, b), w in zip(slots, widths) if w >= min_width]

        # 多小节休止：压在第三线（中间线）上的粗实线
        thick = [
            d['rect'].x0 for d in drawings
            if d['type'] == 's'
            and d['rect'].height <= 1.0
            and d['rect'].width >= _THICK_MIN_WIDTH
            and (d.get('width') or 0) >= _THICK_MIN_LW
            and abs(d['rect'].y0 - middle) < _THICK_Y_TOL
        ]
        # 单小节休止记号：第 2 线上的短横线
        whole_rests = [
            d for d in drawings
            if d['type'] == 's'
            and _WHOLE_REST_MIN_W <= d['rect'].width <= _WHOLE_REST_MAX_W
            and (d.get('width') or 1.0) < 1.0
            and abs(d['rect'].y0 - lines[1]) < 1.5
        ]

        kinds: list[str] = []
        for a, b in slots:
            if any(a - 1 <= t <= b + 1 for t in thick):
                kinds.append('M')
            elif any(a <= w['rect'].x0 <= b for w in whole_rests):
                kinds.append('W')
            else:
                kinds.append('N')
        systems.append(kinds)
        if _geometry is not None:
            _geometry.append((lines, slots))

    return systems


# ─────────────────────────────────────────────────────────────────────────────
# MusicXML 侧工具
# ─────────────────────────────────────────────────────────────────────────────

def _xml_system_blocks(part: ET.Element) -> list[list[ET.Element]]:
    blocks: list[list[ET.Element]] = []
    cur: list[ET.Element] = []
    for measure in part.findall('measure'):
        pr = measure.find('print')
        starts = pr is not None and (
            pr.get('new-system') == 'yes' or pr.get('new-page') == 'yes'
        )
        if starts and cur:
            blocks.append(cur)
            cur = []
        cur.append(measure)
    if cur:
        blocks.append(cur)
    return blocks


def _measure_kind(measure: ET.Element) -> str:
    notes = measure.findall('note')
    if not notes:
        return 'R'
    return 'N' if any(n.find('rest') is None for n in notes) else 'R'


def _read_state(measure: ET.Element):
    """读取小节内声明的 (divisions, beats, beat_type)，未声明的位置返回 None。"""
    div = beats = beat_type = None
    for attrs in measure.findall('attributes'):
        d = attrs.find('divisions')
        if d is not None and d.text:
            try:
                div = int(d.text)
            except ValueError:
                pass
        t = attrs.find('time')
        if t is not None:
            try:
                b = t.find('beats')
                bt = t.find('beat-type')
                if b is not None and b.text:
                    beats = int(b.text)
                if bt is not None and bt.text:
                    beat_type = int(bt.text)
            except ValueError:
                pass
    return div, beats, beat_type


def _strip_measure_style(measure: ET.Element) -> None:
    """删除 homr 写入的 <measure-style><multiple-rest>：折叠后它已无意义。"""
    for attrs in measure.findall('attributes'):
        for style in list(attrs.findall('measure-style')):
            attrs.remove(style)


def _whole_rest_duration(divisions: int, beats: int, beat_type: int) -> int:
    return int(round(divisions * beats * 4 / max(beat_type, 1)))


def _make_whole_rest_measure(divisions: int, beats: int, beat_type: int) -> ET.Element:
    measure = ET.Element('measure', {'number': '?'})
    note = ET.SubElement(measure, 'note')
    ET.SubElement(note, 'rest', {'measure': 'yes'})
    ET.SubElement(note, 'duration').text = str(_whole_rest_duration(divisions, beats, beat_type))
    ET.SubElement(note, 'type').text = 'whole'
    return measure


def _is_whole_measure_rest(measure: ET.Element) -> bool:
    """该小节是否已经是「一个全小节休止符」（homr 的标准输出形态）。"""
    notes = measure.findall('note')
    if len(notes) != 1:
        return False
    rest = notes[0].find('rest')
    return rest is not None and rest.get('measure') == 'yes'


def _absorb_attributes(head: ET.Element, victim: ET.Element) -> None:
    """删除 victim 之前，把它携带的 <attributes>（拍号 / 调号 / divisions）转移给 head。

    MusicXML 中同一小节内可以有多个 <attributes>，按出现顺序生效，
    等价于它们原先分别出现在连续的几个小节里。
    """
    for attrs in list(victim.findall('attributes')):
        for style in list(attrs.findall('measure-style')):
            attrs.remove(style)
        if len(attrs) == 0:
            continue
        victim.remove(attrs)
        head.append(attrs)


# ─────────────────────────────────────────────────────────────────────────────
# 对齐：排版小节槽位 ↔ MusicXML 小节
# ─────────────────────────────────────────────────────────────────────────────

def _anchor_pairs(slots: list[str], kinds: list[str]) -> list[tuple[int, int]]:
    """贪心对齐「有音符的槽位」与「有音符的小节」，返回 (槽位下标, 小节下标)。"""
    pairs: list[tuple[int, int]] = []
    si = xi = 0
    while True:
        while si < len(slots) and slots[si] != 'N':
            si += 1
        while xi < len(kinds) and kinds[xi] != 'N':
            xi += 1
        if si >= len(slots) or xi >= len(kinds):
            break
        pairs.append((si, xi))
        si += 1
        xi += 1
    return pairs


def collapse_page_multirests(part: ET.Element, page_systems: list[list[str]]) -> dict:
    """按「一个排版小节 ⇒ 一个 MusicXML 小节」重排该页（原地修改 part）。

    对每个系统：
      * ``M``（多小节休止粗实线）⇒ **一个**小节；
      * ``W``（单小节休止记号）⇒ 一个小节；
      * ``N``（有音符）⇒ 一个小节，内容不动。
    多出来的休止小节被合并删除（属性迁移到保留的小节），缺少的补全小节休止符。
    """
    stats = {'systems': 0, 'skipped': 0, 'removed': 0, 'inserted': 0,
             'multirest': 0, 'normalized': 0, 'reordered': 0, 'dropped': 0}

    blocks = _xml_system_blocks(part)
    if len(blocks) != len(page_systems):
        stats['skipped'] = max(len(blocks), len(page_systems)) or 1
        return stats

    ordered: list[ET.Element] = []
    state = (1, 4, 4)
    changed = False

    for block, slots in zip(blocks, page_systems):
        stats['multirest'] += slots.count('M')
        if len(block) == len(slots) and 'M' not in slots:
            # 系统内没有多小节休止、数量也已对上：只做时值规范化
            for measure in block:
                _strip_measure_style(measure)
                div, b, bt = _read_state(measure)
                merged = tuple(x if x is not None else y
                               for x, y in zip((div, b, bt), state))
                state = merged
            ordered.extend(block)
            continue

        # Rest-count alignment cannot infer missing note barlines. Refuse a
        # destructive guess: it would drop pitched measures or invent rests.
        if sum(_measure_kind(m) == 'N' for m in block) != slots.count('N'):
            raise ValueError('Ambiguous note/bar alignment; preserve original MusicXML '
                             'and review source barlines before collapsing rests')

        stats['systems'] += 1
        changed = True
        # The caller only writes the file back when a measure was removed or
        # inserted.  A system can also need a rewrite with *neither*: a trailing
        # rest measure that has no slot is simply left out of ``plan`` (it is
        # not an absorbed "victim", so ``removed`` stays 0), yet dropping it is
        # exactly what makes the measure count match the engraving.  Without
        # this flag the corrected plan was computed and then thrown away, and
        # the page kept one spurious whole-rest measure — the extra jianpu cell.
        stats['reordered'] += 1

        pairs = _anchor_pairs(slots, [_measure_kind(m) for m in block])
        s_bounds = [0] + [a for a, _ in pairs] + [len(slots)]
        x_bounds = [0] + [b for _, b in pairs] + [len(block)]

        plan: list[ET.Element] = []
        prev = state

        for seg in range(len(s_bounds) - 1):
            s0, s1 = s_bounds[seg], s_bounds[seg + 1]
            x0, x1 = x_bounds[seg], x_bounds[seg + 1]
            seg_slots = slots[s0:s1]
            if not seg_slots:
                continue
            seg_measures = block[x0:x1]
            note_measures = [m for m in seg_measures if _measure_kind(m) == 'N']
            rest_measures = [m for m in seg_measures if _measure_kind(m) == 'R']

            rest_slot_kinds = [t for t in seg_slots if t in 'MW']
            # 把本段实际的休止小节按顺序分配给各休止槽位：
            #   多余的小节**全部并入第一个 M（多小节休止）槽位**——这正是方案D；
            #   若段内没有 M，就并入第一个休止槽位。缺的槽位标记为 None（稍后补空小节）。
            n_slots = len(rest_slot_kinds)
            alloc = [0] * n_slots
            if n_slots:
                take = min(len(rest_measures), n_slots)
                for k in range(take):
                    alloc[k] = 1
                surplus = len(rest_measures) - take
                if surplus > 0:
                    targets = [k for k, t in enumerate(rest_slot_kinds) if t == 'M']
                    alloc[(targets or [0])[0]] += surplus

            rest_out: list[ET.Element | None] = []
            pos = 0
            for k, count in enumerate(alloc):
                if count == 0:
                    rest_out.append(None)
                    continue
                group = rest_measures[pos:pos + count]
                pos += count
                head = group[0]
                # 后续小节被并入 head：先把它们携带的 <attributes> 迁移过来
                for victim in group[1:]:
                    _absorb_attributes(head, victim)
                    stats['removed'] += 1
                rest_out.append(head)

            ni = ri = 0
            for slot_kind in seg_slots:
                measure: ET.Element | None = None
                if slot_kind == 'N' and ni < len(note_measures):
                    measure = note_measures[ni]
                    ni += 1
                elif ri < len(rest_out):
                    measure = rest_out[ri]
                    ri += 1
                if measure is None:
                    measure = _make_whole_rest_measure(*prev)
                    stats['inserted'] += 1
                div, b, bt = _read_state(measure)
                prev = tuple(x if x is not None else y for x, y in zip((div, b, bt), prev))
                plan.append(measure)

            # 该段没有对应槽位的多余额外小节一律丢弃：本模块的契约就是
            # 「一个排版小节 ⇔ 一个 MusicXML 小节」。这里以前会把多余的
            # **音符**小节追加到计划末尾（怕丢内容），结果该系统仍比排版多
            # 一小节 —— homr 在系统末尾常多切出一个只含零散音符的残小节
            # （有时甚至是 duration=0 的音符），留着会让简谱小节数多于
            # 五线谱，后面所有简谱格整体错位。休止小节本来就走不到这里
            # （它们已被并入 M 槽位或被 alloc 丢弃）。

        # Surplus measures that had no slot are dropped; record them so the
        # caller knows the page really changed (see 'reordered').
        if len(plan) < len(block):
            stats['dropped'] += len(block) - len(plan)

        state = prev

        # 规范化：M/W 槽位必须是「一个全小节休止符」；每小节时值补齐/截短到
        # 一小节，music21 重写时才不会把小节拆开，破坏小节数对应关系。
        for measure, slot_kind in zip(plan, slots):
            div, b, bt = _read_state(measure)
            div, b, bt = tuple(x if x is not None else y for x, y in zip((div, b, bt), state))
            if slot_kind in 'MW' and not _is_whole_measure_rest(measure):
                for note in measure.findall('note'):
                    measure.remove(note)
                note = ET.SubElement(measure, 'note')
                ET.SubElement(note, 'rest', {'measure': 'yes'})
                ET.SubElement(note, 'duration').text = str(_whole_rest_duration(div, b, bt))
                ET.SubElement(note, 'type').text = 'whole'

        ordered.extend(plan)

    if not changed:
        return stats

    for measure in part.findall('measure'):
        part.remove(measure)
    for idx, measure in enumerate(ordered, 1):
        measure.set('number', str(idx))
        part.append(measure)
    for measure in part.findall('measure'):
        _strip_measure_style(measure)
    return stats



# ─────────────────────────────────────────────────────────────────────────────
# 对外接口
# ─────────────────────────────────────────────────────────────────────────────

def collapse_multirests_for_pdf_page(
    pdf_path,
    mxl_path,
    page_index: int = 0,
    log=None,
) -> Optional[dict]:
    """对 ``pdf_path`` 的第 ``page_index`` 页做多小节休止折叠。

    原地（原子替换）重写 ``mxl_path``。无法处理时返回 None，文件保持不变。
    """
    pymupdf = _load_pymupdf()
    if pymupdf is None:
        return None
    pdf_path = Path(pdf_path)
    mxl_path = Path(mxl_path)
    if not pdf_path.exists() or not mxl_path.exists():
        return None
    if pdf_path.suffix.lower() != '.pdf':
        return None

    try:
        doc = pymupdf.open(str(pdf_path))
        if page_index >= doc.page_count:
            doc.close()
            return None
        systems = page_system_slot_kinds(doc.load_page(page_index))
        doc.close()
    except Exception:
        return None
    if not systems or sum(len(s) for s in systems) > _MAX_MEASURES:
        return None

    try:
        tree = ET.parse(str(mxl_path))
    except Exception:
        return None
    root = tree.getroot()

    stats = {'systems': 0, 'skipped': 0, 'removed': 0, 'inserted': 0, 'forced': 0,
             'multirest': 0, 'reordered': 0, 'dropped': 0}
    changed = False
    for part in root.iter('part'):
        part_stats = collapse_page_multirests(part, systems)
        for key in stats:
            stats[key] += part_stats.get(key, 0)
        if (part_stats.get('removed') or part_stats.get('inserted')
                or part_stats.get('forced') or part_stats.get('reordered')):
            changed = True

    if not changed:
        return stats

    tmp = mxl_path.with_name(mxl_path.name + '.tmp')
    tree.write(str(tmp), encoding='UTF-8', xml_declaration=True)
    os.replace(str(tmp), str(mxl_path))
    if log is not None:
        try:
            log(f"  [多小节休止] 第{page_index + 1}页：多小节休止符号 {stats['multirest']} 处，"
                f"重排 {stats['systems']} 个系统（删除多余休止小节 "
                f"{stats['removed'] + stats['dropped']}，补入 {stats['inserted']}）")
        except Exception:
            pass
    return stats
