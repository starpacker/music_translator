"""Read precise MusicXML crescendo/diminuendo boundaries without music21 loss."""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile
import tempfile


def _root(path: Path):
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            container = ET.fromstring(archive.read('META-INF/container.xml'))
            filename = next((entry.get('full-path') for entry in container.iter()
                             if entry.tag.rsplit('}', 1)[-1] == 'rootfile'
                             and entry.get('full-path')), None)
            if not filename:
                raise ValueError('MXL 缺少根 MusicXML 文件')
            return ET.fromstring(archive.read(filename))
    return ET.parse(path).getroot()


def remove_continuation_markers_for_parser(path: Path, temporary_dir: Path) -> Path:
    """Remove layout-only wedge continuations in a temporary parser input.

    music21 rejects the valid MusicXML ``continue`` type. Start and stop
    positions are separately read from the untouched original MusicXML.
    """
    root = _root(path)
    removed = 0
    for parent in root.iter():
        for child in list(parent):
            if child.tag.rsplit('}', 1)[-1] == 'wedge' and child.get('type') == 'continue':
                parent.remove(child)
                removed += 1
    if not removed:
        return path
    out = temporary_dir / '__wedge_continuations_for_parser.musicxml'
    ET.ElementTree(root).write(out, encoding='utf-8', xml_declaration=True)
    return out


def parse_musicxml_compatible(path: Path):
    """Parse valid continuation markers without modifying their source file."""
    from music21 import converter
    with tempfile.TemporaryDirectory(prefix='musicxml_parse_') as tmp:
        compatible = remove_continuation_markers_for_parser(path, Path(tmp))
        return converter.parse(str(compatible))


def read_wedges(path: Path):
    """Return paired (start, end, kind) boundaries per part, plus issues.

    Boundaries are ``(zero_based_measure_index, Fraction(quarter_length))``.
    MusicXML's cursor, divisions, backups, forwards and direction offsets are
    interpreted directly rather than approximated with music21 spanner notes.
    """
    root = _root(path)
    ns = root.tag.split('}', 1)[0] + '}' if root.tag.startswith('{') else ''
    if root.tag != f'{ns}score-partwise':
        raise ValueError('仅支持逐声部 MusicXML 渐强渐弱线')
    result, issues = [], []
    for part_index, part in enumerate(root.findall(f'{ns}part')):
        pairs, active, divisions = [], {}, None
        for mi, measure in enumerate(part.findall(f'{ns}measure')):
            cursor = Fraction(0)
            cursor_valid = True
            for child in measure:
                if child.tag == f'{ns}attributes':
                    value = child.findtext(f'{ns}divisions')
                    if value:
                        try:
                            divisions = int(value)
                            if divisions <= 0:
                                divisions = None
                        except ValueError:
                            divisions = None
                elif child.tag in (f'{ns}note', f'{ns}forward', f'{ns}backup'):
                    if child.tag == f'{ns}note' and (child.find(f'{ns}chord') is not None
                                                      or child.find(f'{ns}grace') is not None):
                        continue
                    value = child.findtext(f'{ns}duration')
                    if value and divisions:
                        try:
                            amount = Fraction(int(value), divisions)
                        except ValueError:
                            issues.append((part_index, mi, '无效时值'))
                            cursor_valid = False
                            continue
                        cursor += -amount if child.tag == f'{ns}backup' else amount
                        if amount < 0 or cursor < 0:
                            cursor_valid = False
                    else:
                        cursor_valid = False
                elif child.tag == f'{ns}direction':
                    offset_text = child.findtext(f'{ns}offset')
                    try:
                        offset = Fraction(int(offset_text), divisions) if offset_text and divisions else Fraction(0)
                    except ValueError:
                        issues.append((part_index, mi, '无效方向偏移'))
                        continue
                    if offset_text and not divisions:
                        issues.append((part_index, mi, '方向偏移缺少 divisions'))
                        continue
                    position = (mi, cursor + offset)
                    staff = child.findtext(f'{ns}staff') or '1'
                    for dtype in child.findall(f'{ns}direction-type'):
                        for wedge in dtype.findall(f'{ns}wedge'):
                            kind = wedge.get('type')
                            identity = (staff, wedge.get('number') or '1')
                            if not cursor_valid or position[1] < 0:
                                issues.append((part_index, mi, '力度线位置无法确定'))
                                if kind in ('crescendo', 'diminuendo'):
                                    active[identity] = None
                                elif kind == 'stop':
                                    active.pop(identity, None)
                                continue
                            if kind in ('crescendo', 'diminuendo'):
                                if identity in active:
                                    issues.append((part_index, mi, '同编号力度线重复开始'))
                                    active[identity] = None
                                else:
                                    active[identity] = (position, kind)
                            elif kind == 'stop':
                                start = active.pop(identity, None)
                                if start is None:
                                    issues.append((part_index, mi, '力度线结束缺少唯一开始'))
                                elif position <= start[0] or position[1] < 0:
                                    issues.append((part_index, mi, '力度线端点顺序无效'))
                                else:
                                    pairs.append((start[0], position, start[1], staff))
                            elif kind != 'continue':
                                issues.append((part_index, mi, '不支持的力度线类型'))
        for _ in active:
            issues.append((part_index, len(part.findall(f'{ns}measure')) - 1, '力度线没有结束'))
        # LilyPond supports one active hairpin in a voice.  Do not silently
        # replace overlapping source wedges with a different shape.
        overlapping = set()
        for i, first in enumerate(pairs):
            for j, second in enumerate(pairs[i + 1:], i + 1):
                if first[0] < second[1] and second[0] < first[1]:
                    overlapping.update((i, j))
        if overlapping:
            issues.append((part_index, None, '重叠力度线无法唯一排版'))
        result.append([(a, b, kind) for i, (a, b, kind, _) in enumerate(pairs) if i not in overlapping])
    return result, issues
