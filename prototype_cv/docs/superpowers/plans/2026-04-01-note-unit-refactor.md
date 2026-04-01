# Note Unit Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace x-coordinate chord grouping and spacing-based duration estimation with stem-tracking-based note unit recognition that determines chord membership and duration from physical stem connections.

**Architecture:** From each detected notehead, trace the stem on the staff-removed image to find its direction and tip. Group noteheads sharing the same stem into chords. Analyze the stem tip region (beams/flags) to determine duration. Package each group as a NoteUnit with complete pitch + duration info, then format directly to jianpu.

**Tech Stack:** Python 3.14, OpenCV, NumPy (existing stack, no new deps)

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `stem_tracking.py` | Create | Trace stem from notehead, return direction + tip + length |
| `note_unit.py` | Create | Group by shared stem → NoteUnit, detect duration from tip, segment measures |
| `jianpu_formatter.py` | Modify | Simplify to read pitch/accidental/duration from NoteUnit directly |
| `main.py` | Modify | Replace old pipeline steps 8-10 with stem_tracking + note_unit |
| `symbol_detection.py` | Modify | Remove `detect_note_durations()` and `_count_beams_improved()` |
| `chord_grouping.py` | Delete | Replaced by `note_unit.py` |
| `duration_estimation.py` | Delete | Replaced by `note_unit.py` |

---

### Task 1: Create `stem_tracking.py` — stem tracing from notehead

**Files:**
- Create: `prototype_cv/stem_tracking.py`

- [ ] **Step 1: Create `stem_tracking.py` with `track_stem()` function**

```python
"""
stem_tracking.py
Trace the stem of a notehead on the staff-removed binary image.
Returns stem direction, tip position, and length.
"""
import numpy as np


def track_stem(music_symbols, note, dy):
    """
    Trace the stem from a notehead.

    Args:
        music_symbols: Binary image with staff lines removed (uint8, white=255)
        note: Dict with keys x, y, w, h, y_center
        dy: Staff line spacing in pixels

    Returns:
        Dict with keys:
            stem_x (int): x-coordinate of the stem
            stem_dir ('up'|'down'|None): stem direction
            stem_tip_y (int): y of the stem tip (far end from notehead)
            stem_length (float): length in pixels
    """
    img_h, img_w = music_symbols.shape[:2]
    x, y, w, h = note['x'], note['y'], note['w'], note['h']
    y_center = note['y_center']
    cx = x + w // 2

    # Step 1: Find stem x by scanning columns near notehead edge
    stem_x = _find_stem_x(music_symbols, cx, y, h, dy, img_w)

    # Step 2: Trace upward from notehead top edge
    stem_width = max(3, int(dy * 0.15))
    stem_top_y = _trace_vertical(music_symbols, stem_x, y, -1, dy, stem_width, img_h)

    # Step 3: Trace downward from notehead bottom edge
    stem_bot_y = _trace_vertical(music_symbols, stem_x, y + h, +1, dy, stem_width, img_h)

    # Step 4: Determine direction
    up_length = y_center - stem_top_y
    down_length = stem_bot_y - y_center
    min_stem_length = dy * 1.5

    if up_length >= down_length and up_length >= min_stem_length:
        return {
            'stem_x': stem_x,
            'stem_dir': 'up',
            'stem_tip_y': stem_top_y,
            'stem_length': float(up_length),
        }
    elif down_length > up_length and down_length >= min_stem_length:
        return {
            'stem_x': stem_x,
            'stem_dir': 'down',
            'stem_tip_y': stem_bot_y,
            'stem_length': float(down_length),
        }
    else:
        return {
            'stem_x': stem_x,
            'stem_dir': None,
            'stem_tip_y': y_center,
            'stem_length': 0.0,
        }


def _find_stem_x(music_symbols, cx, y, h, dy, img_w):
    """Find the x-coordinate of the stem by scanning columns near the notehead."""
    search_range = int(dy * 0.3)
    scan_height = int(dy * 2)
    best_x = cx
    best_density = 0.0
    stem_width = max(3, int(dy * 0.15))

    for col_x in range(max(0, cx - search_range), min(img_w, cx + search_range + 1)):
        x1 = max(0, col_x - stem_width // 2)
        x2 = min(img_w, col_x + stem_width // 2 + 1)

        # Check above notehead
        y_top = max(0, y - scan_height)
        region_above = music_symbols[y_top:y, x1:x2]
        density_above = np.mean(region_above) / 255.0 if region_above.size > 0 else 0.0

        # Check below notehead
        y_bot = min(music_symbols.shape[0], y + h + scan_height)
        region_below = music_symbols[y + h:y_bot, x1:x2]
        density_below = np.mean(region_below) / 255.0 if region_below.size > 0 else 0.0

        density = max(density_above, density_below)
        if density > best_density:
            best_density = density
            best_x = col_x

    return best_x


def _trace_vertical(music_symbols, stem_x, start_y, direction, dy, stem_width, img_h):
    """
    Trace vertically from start_y in the given direction (+1=down, -1=up).
    Returns the y-coordinate where the stem ends.
    """
    half_w = stem_width // 2
    x1 = max(0, stem_x - half_w)
    x2 = min(music_symbols.shape[1], stem_x + half_w + 1)
    gap_tolerance = int(dy * 0.3)
    density_threshold = 0.3
    max_search = int(dy * 4)

    current_y = start_y
    last_good_y = start_y
    gap_count = 0

    for step in range(max_search):
        current_y += direction
        if current_y < 0 or current_y >= img_h:
            break

        row = music_symbols[current_y, x1:x2]
        density = np.mean(row) / 255.0 if row.size > 0 else 0.0

        if density >= density_threshold:
            last_good_y = current_y
            gap_count = 0
        else:
            gap_count += 1
            if gap_count >= gap_tolerance:
                break

    return last_good_y
```

- [ ] **Step 2: Verify stem tracking on test image**

Run:
```bash
cd prototype_cv
python -c "
import cv2, numpy as np
from staff_removal import extract_staff_lines
from pitch_detection import get_staff_systems, pair_grand_staves
from template_matching import find_noteheads
from note_assignment import assign_notes_to_staves, filter_false_positive_notes
from stem_tracking import track_stem

staff_lines, music_symbols, binary = extract_staff_lines('../input_page1.png')
systems = get_staff_systems(staff_lines)
grand_staff_pairs = pair_grand_staves(systems)
dy = np.mean([(s[4]-s[0])/4.0 for s in systems])
boxes, _, _ = find_noteheads(binary, dy, threshold=0.55, staff_systems=systems, music_symbols=music_symbols)
all_notes = [{'x':x,'y':y,'w':w,'h':h,'y_center':y+h//2,'score':sc} for x,y,w,h,sc in boxes]
img_w = binary.shape[1]
all_notes = [n for n in all_notes if n['x'] > int(img_w*0.20)]
treble, bass = assign_notes_to_staves(all_notes, grand_staff_pairs, dy)
treble = filter_false_positive_notes(treble, dy, clef='treble')

for note in treble[:10]:
    stem = track_stem(music_symbols, note, dy)
    print(f'  x={note[\"x\"]:4d} y={note[\"y_center\"]:4d} → dir={stem[\"stem_dir\"]:>5s} tip_y={stem[\"stem_tip_y\"]:4d} len={stem[\"stem_length\"]:.0f}')
"
```

Expected: Each note shows a stem direction (`up` or `down`) with reasonable tip_y and length (> dy*1.5 for notes with stems). Verify that stems point in the correct direction by checking a few known notes.

- [ ] **Step 3: Commit**

```bash
git add prototype_cv/stem_tracking.py
git commit -m "feat: add stem_tracking.py — trace stem from notehead on staff-removed image"
```

---

### Task 2: Create `note_unit.py` — chord grouping by shared stem + duration detection

**Files:**
- Create: `prototype_cv/note_unit.py`

- [ ] **Step 1: Create `note_unit.py` with `build_note_units()` and `detect_duration()`**

```python
"""
note_unit.py
Build NoteUnit objects: group noteheads by shared stem (= chord),
detect duration from stem tip morphology, compute pitch.
"""
import numpy as np
from pitch_detection import y_to_jianpu


def build_note_units(notes, music_symbols, dy):
    """
    Group notes by shared stem and detect duration for each group.

    Args:
        notes: List of note dicts (with stem info from track_stem, plus clef/system/pair_idx)
        music_symbols: Staff-removed binary image
        dy: Staff line spacing

    Returns:
        List of NoteUnit dicts, each with:
            notes: [{pitch, accidental (None initially), x, y_center, clef, system, pair_idx}]
            duration: float (beats)
            stem_dir: 'up'/'down'/None
            stem_x: int
            x: float (average notehead x, for sorting)
    """
    if not notes:
        return []

    # Group by shared stem
    groups = _group_by_stem(notes, dy)

    # Build NoteUnit for each group
    units = []
    for group in groups:
        unit = _build_single_unit(group, music_symbols, dy)
        units.append(unit)

    return units


def _group_by_stem(notes, dy):
    """
    Group notes that share the same stem.
    Rules: same stem_x (within dy*0.5), same stem_dir, same system.
    """
    # Sort by stem_x
    sorted_notes = sorted(notes, key=lambda n: n.get('stem', {}).get('stem_x', n['x']))

    groups = []
    current_group = [sorted_notes[0]]

    for note in sorted_notes[1:]:
        first = current_group[0]
        first_stem = first.get('stem', {})
        note_stem = note.get('stem', {})

        first_sx = first_stem.get('stem_x', first['x'])
        note_sx = note_stem.get('stem_x', note['x'])
        first_dir = first_stem.get('stem_dir')
        note_dir = note_stem.get('stem_dir')
        same_system = (first.get('system') is not None
                       and note.get('system') is not None
                       and first['system'] == note['system'])

        if (abs(first_sx - note_sx) < dy * 0.5
                and first_dir == note_dir
                and same_system):
            current_group.append(note)
        else:
            groups.append(current_group)
            current_group = [note]

    if current_group:
        groups.append(current_group)

    return groups


def _build_single_unit(group, music_symbols, dy):
    """Build a NoteUnit from a group of notes sharing a stem."""
    # Sort notes by y_center descending (low pitch first, matching existing format)
    group.sort(key=lambda n: n['y_center'], reverse=True)

    # Use the first note's stem info (all share the same stem)
    stem = group[0].get('stem', {})
    stem_dir = stem.get('stem_dir')
    stem_x = stem.get('stem_x', group[0]['x'])
    stem_tip_y = stem.get('stem_tip_y', group[0]['y_center'])

    # Detect duration
    duration = _detect_duration(group, music_symbols, dy, stem_dir, stem_tip_y, stem_x)

    # Build note entries with pitch
    note_entries = []
    for n in group:
        base, suffix = y_to_jianpu(n['y_center'], n['system'], n['clef'])
        note_entries.append({
            'pitch': base + suffix,
            'accidental': None,
            'x': n['x'],
            'y_center': n['y_center'],
            'clef': n['clef'],
            'system': n['system'],
            'pair_idx': n.get('pair_idx', 0),
            'w': n.get('w', 0),
        })

    avg_x = float(np.mean([n['x'] for n in group]))

    return {
        'notes': note_entries,
        'duration': duration,
        'stem_dir': stem_dir,
        'stem_x': stem_x,
        'x': avg_x,
    }


def _detect_duration(group, music_symbols, dy, stem_dir, stem_tip_y, stem_x):
    """
    Detect the duration of a note unit from its stem tip morphology.

    Returns duration in beats (4.0, 2.0, 1.0, 0.5, 0.25).
    """
    # Check if notehead is hollow (open)
    is_hollow = _is_hollow(group[0], music_symbols, dy)

    if stem_dir is None:
        # No stem
        return 4.0 if is_hollow else 1.0

    if is_hollow:
        # Stem + hollow = half note
        return 2.0

    # Filled notehead with stem: count beams at stem tip
    beam_count = _count_beams_at_tip(music_symbols, stem_tip_y, stem_x, stem_dir, dy)

    if beam_count == 0:
        # Check for flag (isolated note, not beamed)
        flag_count = _detect_flag(music_symbols, stem_tip_y, stem_x, stem_dir, dy)
        if flag_count >= 2:
            return 0.25
        elif flag_count == 1:
            return 0.5
        return 1.0
    elif beam_count == 1:
        return 0.5
    elif beam_count >= 2:
        return 0.25
    return 1.0


def _is_hollow(note, music_symbols, dy):
    """Check if a notehead is hollow (open) by examining fill ratio at center."""
    pad = max(2, int(dy * 0.2))
    y_center = note['y_center']
    cx = note['x'] + note.get('w', 0) // 2
    img_h, img_w = music_symbols.shape[:2]

    ny1 = max(0, y_center - pad)
    ny2 = min(img_h, y_center + pad)
    nx1 = max(0, cx - pad)
    nx2 = min(img_w, cx + pad)

    region = music_symbols[ny1:ny2, nx1:nx2]
    fill_ratio = np.mean(region) / 255.0 if region.size > 0 else 1.0
    return fill_ratio <= 0.4


def _count_beams_at_tip(music_symbols, stem_tip_y, stem_x, stem_dir, dy):
    """
    Count horizontal beams at the stem tip using horizontal projection.
    Beams are thick horizontal lines connecting multiple stems.
    """
    img_h, img_w = music_symbols.shape[:2]

    # ROI around stem tip
    roi_half_h = int(dy * 0.5)
    roi_half_w = int(dy * 1.5)

    y1 = max(0, stem_tip_y - roi_half_h)
    y2 = min(img_h, stem_tip_y + roi_half_h)
    x1 = max(0, stem_x - roi_half_w)
    x2 = min(img_w, stem_x + roi_half_w)

    roi = music_symbols[y1:y2, x1:x2]
    if roi.size == 0 or roi.shape[0] < 3 or roi.shape[1] < 3:
        return 0

    # Horizontal projection
    h_proj = np.sum(roi, axis=1).astype(float) / (255.0 * max(1, roi.shape[1]))

    # Count beam-like horizontal bands
    beam_threshold = 0.4
    min_thickness = max(2, int(dy * 0.08))
    max_thickness = int(dy * 0.45)

    beam_count = 0
    in_beam = False
    thickness = 0

    for val in h_proj:
        if val > beam_threshold:
            if not in_beam:
                beam_count += 1
                thickness = 1
                in_beam = True
            else:
                thickness += 1
        else:
            if in_beam:
                if thickness < min_thickness or thickness > max_thickness:
                    beam_count -= 1
                in_beam = False
                thickness = 0

    if in_beam:
        if thickness < min_thickness or thickness > max_thickness:
            beam_count -= 1

    return max(0, beam_count)


def _detect_flag(music_symbols, stem_tip_y, stem_x, stem_dir, dy):
    """
    Detect flag (符尾) on an isolated note by checking pixel density
    to the right of the stem tip.
    """
    img_h, img_w = music_symbols.shape[:2]

    roi_half_h = int(dy * 0.8)
    roi_w = int(dy * 1.0)

    y1 = max(0, stem_tip_y - roi_half_h)
    y2 = min(img_h, stem_tip_y + roi_half_h)
    x1 = stem_x
    x2 = min(img_w, stem_x + roi_w)

    roi = music_symbols[y1:y2, x1:x2]
    if roi.size == 0:
        return 0

    density = np.mean(roi) / 255.0

    # A flag has significant density to the right of the stem tip
    if density > 0.15:
        # Try to count distinct lobes (each flag is a curve)
        # Use vertical projection to see distinct bumps
        v_proj = np.sum(roi, axis=0).astype(float) / (255.0 * max(1, roi.shape[0]))
        # If average density is high, likely 2+ flags
        if density > 0.25:
            return 2
        return 1

    return 0


def segment_into_measures(note_units, rests, barline_xs, dy):
    """
    Segment NoteUnits and rests into measures based on barline x-positions.
    """
    events = []

    for unit in note_units:
        events.append({
            'type': 'note_unit',
            'x': unit['x'],
            'unit': unit,
        })

    for rest in rests:
        events.append({
            'type': 'rest',
            'x': rest['x'],
            'duration': rest['duration'],
        })

    events.sort(key=lambda e: e['x'])

    if not barline_xs or not events:
        return [events] if events else []

    boundaries = [0] + list(barline_xs) + [float('inf')]

    measures = []
    for i in range(len(boundaries) - 1):
        left = boundaries[i]
        right = boundaries[i + 1]
        measure_events = [e for e in events if left <= e['x'] < right]
        if dy is not None:
            measure_events = _clean_rests_in_measure(measure_events, dy)
        measures.append(measure_events)

    return measures


def _clean_rests_in_measure(events, dy):
    """Remove false positive rests overlapping with note units or duplicating each other."""
    unit_xs = [e['x'] for e in events if e['type'] == 'note_unit']
    if not unit_xs:
        return events

    cleaned = []
    kept_rest_xs = []

    for e in events:
        if e['type'] != 'rest':
            cleaned.append(e)
            continue

        if any(abs(e['x'] - ux) < dy * 1.5 for ux in unit_xs):
            continue

        if any(abs(e['x'] - rx) < dy * 2.0 for rx in kept_rest_xs):
            continue

        cleaned.append(e)
        kept_rest_xs.append(e['x'])

    return cleaned
```

- [ ] **Step 2: Verify note unit building on test image**

Run:
```bash
cd prototype_cv
python -c "
import cv2, numpy as np
from staff_removal import extract_staff_lines
from pitch_detection import get_staff_systems, pair_grand_staves
from template_matching import find_noteheads
from note_assignment import assign_notes_to_staves, filter_false_positive_notes
from stem_tracking import track_stem
from note_unit import build_note_units

staff_lines, music_symbols, binary = extract_staff_lines('../input_page1.png')
systems = get_staff_systems(staff_lines)
grand_staff_pairs = pair_grand_staves(systems)
dy = np.mean([(s[4]-s[0])/4.0 for s in systems])
boxes, _, _ = find_noteheads(binary, dy, threshold=0.55, staff_systems=systems, music_symbols=music_symbols)
all_notes = [{'x':x,'y':y,'w':w,'h':h,'y_center':y+h//2,'score':sc} for x,y,w,h,sc in boxes]
img_w = binary.shape[1]
all_notes = [n for n in all_notes if n['x'] > int(img_w*0.20)]
treble, bass = assign_notes_to_staves(all_notes, grand_staff_pairs, dy)
treble = filter_false_positive_notes(treble, dy, clef='treble')

# Track stems
for note in treble:
    note['stem'] = track_stem(music_symbols, note, dy)

# Build units for pair 0
pair_treble = [n for n in treble if n.get('pair_idx') == 0]
units = build_note_units(pair_treble, music_symbols, dy)

print(f'Pair 0 treble: {len(pair_treble)} notes -> {len(units)} units')
for i, u in enumerate(sorted(units, key=lambda u: u['x'])):
    pitches = [n['pitch'] for n in u['notes']]
    chord_str = '[' + ' '.join(pitches) + ']' if len(pitches) > 1 else pitches[0]
    print(f'  Unit {i}: x={u[\"x\"]:.0f} dur={u[\"duration\"]} stem={u[\"stem_dir\"]} {chord_str}')
"
```

Expected: Note units group correctly into chords (multiple pitches per unit when they share a stem). Durations should mostly match expected values (1.0 for quarter, 0.5 for eighth, 0.25 for sixteenth).

- [ ] **Step 3: Commit**

```bash
git add prototype_cv/note_unit.py
git commit -m "feat: add note_unit.py — stem-based chord grouping and duration detection"
```

---

### Task 3: Rewrite `jianpu_formatter.py` to use NoteUnit

**Files:**
- Modify: `prototype_cv/jianpu_formatter.py`

- [ ] **Step 1: Rewrite `jianpu_formatter.py`**

Replace the entire file with:

```python
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
    # base = digit part, suffix = octave marks
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
```

- [ ] **Step 2: Commit**

```bash
git add prototype_cv/jianpu_formatter.py
git commit -m "refactor: rewrite jianpu_formatter to use NoteUnit data structure"
```

---

### Task 4: Rewire `main.py` pipeline

**Files:**
- Modify: `prototype_cv/main.py`

- [ ] **Step 1: Update imports in `main.py`**

Replace the old imports:

```python
from chord_grouping import group_into_chords, segment_into_measures
from duration_estimation import estimate_durations_by_spacing
```

With:

```python
from stem_tracking import track_stem
from note_unit import build_note_units, segment_into_measures
```

- [ ] **Step 2: Replace pipeline steps 8-9 in `main.py`**

Find the current step 8 (detect note durations):
```python
    # ── 8. Detect Note Durations ──
    print("8. Detecting note durations...")
    treble_notes = detect_note_durations(binary, music_symbols, treble_notes, systems, dy)
    bass_notes = detect_note_durations(binary, music_symbols, bass_notes, systems, dy)
```

Replace with:
```python
    # ── 8. Track Stems ──
    print("8. Tracking stems...")
    for note in treble_notes + bass_notes:
        note['stem'] = track_stem(music_symbols, note, dy)
    print(f"   Tracked stems for {len(treble_notes) + len(bass_notes)} notes")
```

- [ ] **Step 3: Replace the chord grouping / duration estimation loop in step 9**

Find the current step 9 loop body (inside `for pair_idx, (treble_sys, bass_sys) in enumerate(grand_staff_pairs):`):
```python
        treble_chords = group_into_chords(pair_treble, dy)
        bass_chords = group_into_chords(pair_bass, dy)

        treble_measures = segment_into_measures(treble_chords, pair_t_rests, barlines, pair_idx, dy=dy)
        bass_measures = segment_into_measures(bass_chords, pair_b_rests, barlines, pair_idx, dy=dy)

        treble_measures = estimate_durations_by_spacing(treble_measures, beats_per_measure=2.0)
        bass_measures = estimate_durations_by_spacing(bass_measures, beats_per_measure=2.0)
```

Replace with:
```python
        treble_units = build_note_units(pair_treble, music_symbols, dy)
        bass_units = build_note_units(pair_bass, music_symbols, dy)

        treble_measures = segment_into_measures(treble_units, pair_t_rests, barlines, dy)
        bass_measures = segment_into_measures(bass_units, pair_b_rests, barlines, dy)
```

- [ ] **Step 4: Remove the `detect_note_durations` import from symbol_detection**

In the imports section at the top of `main.py`, find:
```python
from symbol_detection import (
    detect_barlines,
    detect_accidentals_global,
    assign_accidentals_to_notes,
    detect_rests,
    detect_note_durations,
)
```

Remove `detect_note_durations`:
```python
from symbol_detection import (
    detect_barlines,
    detect_accidentals_global,
    assign_accidentals_to_notes,
    detect_rests,
)
```

- [ ] **Step 5: Update `_print_and_save_output` to use new formatter**

Find the `_print_and_save_output` function and update the `format_output` calls. The new `format_output` no longer takes `barline_xs`:

```python
def _print_and_save_output(pair_data, accidentals_map, dy):
    """Format, print, and save the Jianpu output."""
    all_treble_lines = []
    all_bass_lines = []

    for pd in pair_data:
        t_line = format_output(pd['treble_measures'], accidentals_map, dy=dy)
        b_line = format_output(pd['bass_measures'], accidentals_map, dy=dy)
        all_treble_lines.append(t_line)
        all_bass_lines.append(b_line)
```

(The rest of `_print_and_save_output` stays the same.)

- [ ] **Step 6: Update visualization `format_measure` calls**

In `_generate_annotated_image`, the `format_measure` call needs updating. Find:
```python
                    text = format_measure(measures[mi], accidentals_map,
                                          measure_idx=mi, barline_xs=barlines, dy=dy)
```

Replace with:
```python
                    text = format_measure(measures[mi], accidentals_map,
                                          measure_idx=mi, dy=dy)
```

Similarly in `_generate_jianpu_pil` and `_generate_jianpu_cv2`, find all `format_measure` calls and remove the `barline_xs=barline_xs` argument (the new signature doesn't take it).

In `_generate_jianpu_pil`:
```python
        t_parts = [format_measure(m, accidentals_map, measure_idx=mi, dy=dy)
                   for mi, m in enumerate(pd['treble_measures'])]
        b_parts = [format_measure(m, accidentals_map, measure_idx=mi, dy=dy)
                   for mi, m in enumerate(pd['bass_measures'])]
```

In `_generate_jianpu_cv2`:
```python
        t_parts = [format_measure(m, accidentals_map, measure_idx=mi, dy=dy)
                   for mi, m in enumerate(pd['treble_measures'])]
        b_parts = [format_measure(m, accidentals_map, measure_idx=mi, dy=dy)
                   for mi, m in enumerate(pd['bass_measures'])]
```

- [ ] **Step 7: Commit**

```bash
git add prototype_cv/main.py
git commit -m "refactor: rewire main.py pipeline to use stem_tracking + note_unit"
```

---

### Task 5: Clean up — remove old modules and dead code

**Files:**
- Delete: `prototype_cv/chord_grouping.py`
- Delete: `prototype_cv/duration_estimation.py`
- Modify: `prototype_cv/symbol_detection.py` (remove `detect_note_durations` and `_count_beams_improved`)

- [ ] **Step 1: Delete `chord_grouping.py` and `duration_estimation.py`**

```bash
cd prototype_cv
rm chord_grouping.py duration_estimation.py
```

- [ ] **Step 2: Remove `detect_note_durations` and `_count_beams_improved` from `symbol_detection.py`**

Open `symbol_detection.py` and delete the two functions (lines 574-692 approximately):
- `def detect_note_durations(binary_img, music_symbols_img, noteheads, staff_systems, dy):` — entire function
- `def _count_beams_improved(binary_img, cx, cy, dy, system):` — entire function

- [ ] **Step 3: Commit**

```bash
git add -u prototype_cv/chord_grouping.py prototype_cv/duration_estimation.py
git add prototype_cv/symbol_detection.py
git commit -m "cleanup: remove chord_grouping.py, duration_estimation.py, and dead beam counting code"
```

---

### Task 6: End-to-end test — verify 100% accuracy on first line

**Files:**
- No new files

- [ ] **Step 1: Run the full pipeline**

```bash
cd prototype_cv
PYTHONIOENCODING=utf-8 python main.py
```

Expected: Pipeline completes without errors, produces `output_jianpu.txt`, `output_jianpu_on_staff.png`, `output_jianpu_clean.png`.

- [ ] **Step 2: Run evaluation**

```bash
PYTHONIOENCODING=utf-8 python evaluate.py
```

Expected:
```
TREBLE (高音): 5/5 measures exact match (100%)
BASS   (低音): 5/5 measures exact match (100%)
OVERALL:
  Measure accuracy: 10/10 (100%)
  Pitch accuracy:   94/94 (100%)
```

- [ ] **Step 3: If accuracy < 100%, debug and fix**

Run debug script to compare old vs new output per measure:
```bash
PYTHONIOENCODING=utf-8 python -c "
from evaluate import load_gt_from_file, load_output_from_file, compare_measures, print_report
gt_t, gt_b = load_gt_from_file('ground_truth.md')
out_t, out_b = load_output_from_file('output_jianpu.txt')
t_stats = compare_measures(gt_t, out_t[:len(gt_t)])
b_stats = compare_measures(gt_b, out_b[:len(gt_b)])
print_report(t_stats, 'TREBLE')
print_report(b_stats, 'BASS')
"
```

For any failing measure, inspect the NoteUnit debug output (add temporary prints in `note_unit.py`) to check:
1. Are notes grouped into chords correctly? (shared stem check)
2. Is the duration detected correctly? (beam count at stem tip)
3. Is the pitch correct? (y_to_jianpu mapping)

Fix any issues in `stem_tracking.py` or `note_unit.py`, then re-run steps 1-2.

- [ ] **Step 4: Commit final passing state**

```bash
git add -A prototype_cv/
git commit -m "verified: note unit refactor passes 100% accuracy on first line"
```

---

### Task 7: Update README.md

**Files:**
- Modify: `prototype_cv/README.md`

- [ ] **Step 1: Update the algorithm section in README.md**

Update the pipeline description to reflect the new architecture:
- Replace "第 8 阶段：时值检测" with "第 8 阶段：符干追踪"
- Replace "第 9 阶段：和弦分组 + 小节分割" with "第 9 阶段：音符单元构建 + 小节分割"
- Remove references to `chord_grouping.py`, `duration_estimation.py`
- Add descriptions for `stem_tracking.py` and `note_unit.py`
- Update the project structure listing
- Update the "泛化性分析" section to note the improved robustness of stem-based grouping

- [ ] **Step 2: Commit**

```bash
git add prototype_cv/README.md
git commit -m "docs: update README for note unit refactor"
```
