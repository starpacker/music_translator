"""
jianpu_formatter.py
Format NoteUnit events into Jianpu (简谱) notation strings.

Handles:
- NoteUnit → jianpu number with octave suffix
- Accidental prefix (#, b) with measure-level persistence
- Natural sign (n) cancels persistence
- Chord brackets [note1 note2 ...]
- Duration suffix (/2, /4)
- Rest formatting (0, 0/2, 0/4)
"""


def duration_to_suffix(duration):
    """Convert numeric duration to Jianpu suffix."""
    if duration == 0.25:
        return "/4"
    elif duration == 0.5:
        return "/2"
    return ""


def format_note(note, accidentals_map, persistent_accs=None, dy=21.0):
    """
    Format a single note from a NoteUnit as Jianpu string.

    note: dict with 'pitch' (str like "3'"), 'accidental' (str or None),
          'x', 'y_center', 'w' (optional)
    accidentals_map: {(cx, cy): '#'/'b'/'n'} from global detection
    persistent_accs: mutable dict for measure-level accidental persistence
    """
    pitch = note['pitch']
    base = pitch[0]
    suffix = pitch[1:]

    # Look up accidental from the global map
    cx = note['x'] + note.get('w', 0) // 2
    cy = note['y_center']
    key = (cx, cy)
    acc = accidentals_map.get(key, '')
    note_x = note['x']

    pitch_key = base + suffix
    max_persist_gap = dy * 10

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
    if dur == 4.0:
        return "0 0 0 0"
    elif dur == 2.0:
        return "0 0"
    elif dur == 0.5:
        return "0/2"
    elif dur == 0.25:
        return "0/4"
    return "0"


def format_measure(measure, accidentals_map, measure_idx=0, dy=21.0):
    """Format a complete measure as Jianpu string."""
    if not measure:
        return "0 0"

    persistent_accs = {}
    parts = []

    for event in measure:
        if event['type'] == 'rest':
            parts.append(format_rest(event))
        elif event['type'] == 'note_unit':
            parts.append(format_note_unit(event['unit'], accidentals_map,
                                          persistent_accs, dy=dy))

    return " ".join(parts)


def format_output(measures, accidentals_map, dy=21.0):
    """Format all measures into the final Jianpu output string."""
    lines = []
    for i, measure in enumerate(measures):
        measure_str = format_measure(measure, accidentals_map,
                                     measure_idx=i, dy=dy)
        lines.append("|" + measure_str + "|")
    return "\n".join(lines)
