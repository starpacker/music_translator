"""
jianpu_formatter.py
Format NoteUnit events into Jianpu (简谱) text strings.

This module produces the text representation written to
``output_jianpu.txt``. It uses ASCII-only suffixes so the file
renders identically in any terminal/editor and so the strings can
be drawn with cv2.putText. The visual jianpu (with proper 减时线
underlines and octave dots) is rendered separately by
``jianpu_visual.py``.

Suffix table:
    eighth         /2
    sixteenth      /4
    dotted eighth  /2.
    dotted quarter .
    half           -
    dotted half    --
    whole          ---
    triplet etc.   /3, /6, /12, /3*2
"""


def duration_to_suffix(duration):
    """Convert numeric duration to Jianpu ASCII suffix."""
    if abs(duration - 1.0 / 12.0) < 0.01:
        return "/12"
    if abs(duration - 1.0 / 6.0) < 0.02:
        return "/6"
    if abs(duration - 1.0 / 3.0) < 0.03:
        return "/3"
    if abs(duration - 2.0 / 3.0) < 0.05:
        return "/3*2"  # quarter-note triplet (2/3 beat each)
    if abs(duration - 0.25) < 0.01:
        return "/4"
    if abs(duration - 0.5) < 0.01:
        return "/2"
    if abs(duration - 0.75) < 0.05:
        return "/2."     # dotted eighth
    if abs(duration - 1.5) < 0.1:
        return "."       # dotted quarter
    if abs(duration - 2.0) < 0.1:
        return "-"       # half note
    if abs(duration - 3.0) < 0.1:
        return "--"      # dotted half note
    if abs(duration - 4.0) < 0.1:
        return "---"     # whole note
    return ""


def format_note(note, accidentals_map, persistent_accs=None, dy=21.0):
    """Format a single note from a NoteUnit as Jianpu string."""
    pitch = note['pitch']
    base = pitch[0]
    suffix = pitch[1:]

    cx = note['x'] + note.get('w', 0) // 2
    cy = note['y_center']
    key = (cx, cy)
    acc = accidentals_map.get(key, '')
    note_x = note['x']

    pitch_key = base + suffix
    max_persist_gap = dy * 5.0

    if acc == 'n':
        if persistent_accs is not None and pitch_key in persistent_accs:
            del persistent_accs[pitch_key]
        acc = ''
    elif acc:
        if persistent_accs is not None:
            persistent_accs[pitch_key] = (acc, note_x)
    else:
        if persistent_accs is not None and pitch_key in persistent_accs:
            stored_acc, stored_x = persistent_accs[pitch_key]
            if abs(note_x - stored_x) <= max_persist_gap:
                acc = stored_acc
                persistent_accs[pitch_key] = (stored_acc, note_x)

    return acc + base + suffix


def format_note_unit(unit, accidentals_map, persistent_accs=None, dy=21.0):
    """Format a NoteUnit (single note or chord) with duration suffix."""
    notes = unit['notes']
    duration = unit['duration']

    note_strs = [format_note(n, accidentals_map, persistent_accs, dy=dy)
                 for n in notes]

    if len(note_strs) == 1:
        chord_str = note_strs[0]
    else:
        chord_str = "[" + " ".join(note_strs) + "]"

    return chord_str + duration_to_suffix(duration)


def format_rest(event):
    """Format a rest event."""
    dur = event.get('duration', 1.0)
    if abs(dur - 4.0) < 0.05:
        return "0---"
    if abs(dur - 3.0) < 0.05:
        return "0--"
    if abs(dur - 2.0) < 0.05:
        return "0-"
    if abs(dur - 1.5) < 0.05:
        return "0."
    if abs(dur - 0.75) < 0.05:
        return "0/2."
    if abs(dur - 0.5) < 0.05:
        return "0/2"
    if abs(dur - 0.25) < 0.05:
        return "0/4"
    return "0"


def format_measure(measure, accidentals_map, measure_idx=0, dy=21.0):
    """Format a complete measure as Jianpu string."""
    if not measure:
        return "0 0"

    persistent_accs = {}
    parts = []

    multi_count = None
    for event in measure:
        if event['type'] == 'rest':
            parts.append(format_rest(event))
        elif event['type'] == 'note_unit':
            parts.append(format_note_unit(event['unit'], accidentals_map,
                                          persistent_accs, dy=dy))
        elif event['type'] == 'multi_rest_count':
            multi_count = event['count']

    result = " ".join(parts)
    if multi_count is not None:
        result += f" ×{multi_count}"
    return result


def format_output(measures, accidentals_map, dy=21.0,
                   skip_empty=False):
    """Format all measures into the final Jianpu output string."""
    lines = []
    for i, measure in enumerate(measures):
        if skip_empty and not measure:
            continue
        measure_str = format_measure(measure, accidentals_map,
                                     measure_idx=i, dy=dy)
        if skip_empty and measure_str == "0 0":
            continue
        lines.append("|" + measure_str + "|")
    return "\n".join(lines)
