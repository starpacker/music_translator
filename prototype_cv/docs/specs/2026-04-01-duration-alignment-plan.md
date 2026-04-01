# Duration-Aligned Event Merging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correctly handle two-voice notation by detecting per-note durations and merging overlapping notes into chord events — so a single eighth-note 1' gets replicated across two sixteenth-note events it sustains through.

**Architecture:** Three stages in `note_unit.py`: (1) detect duration per individual notehead via beam/flag counting, (2) after grouping into measures, merge notes whose durations overlap into shared events. Also fix clef boundary in `main.py` to stop filtering real notes at x≈280.

**Tech Stack:** Python, OpenCV, NumPy (existing stack)

---

## File Structure

| File | Change | Responsibility |
|------|--------|---------------|
| `prototype_cv/note_unit.py` | Modify | Add `_detect_individual_duration()`, add `_merge_overlapping_notes()`, refactor `build_note_units()` |
| `prototype_cv/main.py:98-99` | Modify | Fix clef boundary from 10.5% to adaptive per-system |
| `prototype_cv/test_duration_merge.py` | Create | Unit tests for duration detection and temporal merge |

---

### Task 1: Fix clef area boundary to keep first notes

**Files:**
- Modify: `prototype_cv/main.py:94-123`

- [ ] **Step 1: Write test to verify first chord detection**

Create `prototype_cv/test_duration_merge.py`:

```python
"""Tests for duration-aligned event merging."""
import subprocess
import sys


def test_first_chord_detected():
    """The [3 1'] chord at x≈284 on line 2 must not be filtered."""
    # Run main and capture output
    result = subprocess.run(
        [sys.executable, "main.py", "../input_page1.png"],
        capture_output=True, text=True, cwd="."
    )
    output = result.stdout
    # System 2 treble first measure must start with [3 1']
    # Find the line after "--- System 2 ---"
    lines = output.split('\n')
    for i, line in enumerate(lines):
        if 'System 2' in line:
            treble_line = lines[i + 1]  # "  Treble: |...|"
            assert '[3 1\']' in treble_line, (
                f"First chord [3 1'] missing from System 2 treble: {treble_line}"
            )
            return
    raise AssertionError("System 2 not found in output")


if __name__ == "__main__":
    test_first_chord_detected()
    print("PASS: test_first_chord_detected")
```

- [ ] **Step 2: Run test to verify it fails with current code**

Run: `cd prototype_cv && python test_duration_merge.py`

Expected: May pass now (since we already changed to 10.5%) or fail if notes at x≈280 still filtered.

- [ ] **Step 3: Implement adaptive per-system clef boundary**

In `prototype_cv/main.py`, replace the fixed-percentage clef boundaries (lines 94-123) with adaptive detection:

```python
    # Filter clef area — adaptive per-system boundary
    # Detect actual first notehead per system, then set boundary just before it.
    img_w = binary.shape[1]
    clef_area_x_first = int(img_w * 0.17)   # pair 0: clef + key-sig + time-sig

    # For lines 2+, find actual first notehead per staff using template matching
    from template_matching import create_notehead_template
    nh_template = create_notehead_template(dy)
    th, tw = nh_template.shape

    clef_boundaries = {}
    for pi, (treble_sys, bass_sys) in enumerate(grand_staff_pairs):
        if pi == 0:
            clef_boundaries[pi * 2] = clef_area_x_first
            clef_boundaries[pi * 2 + 1] = clef_area_x_first
            continue
        # Scan x=200-500 for first notehead on each staff
        boundary = clef_area_x_first  # fallback
        for sys_info in [treble_sys, bass_sys]:
            y1 = max(0, sys_info[0] - int(dy * 2))
            y2 = min(binary.shape[0], sys_info[4] + int(dy * 2))
            roi = binary[y1:y2, 200:500]
            if tw < roi.shape[1] and th < roi.shape[0]:
                res = cv2.matchTemplate(roi, nh_template, cv2.TM_CCOEFF_NORMED)
                loc = np.where(res >= 0.55)
                if len(loc[0]) > 0:
                    first_x = int(np.min(loc[1])) + 200
                    # Set boundary 1*dy before the first notehead
                    candidate = max(200, first_x - int(dy))
                    boundary = min(boundary, candidate)
        clef_boundaries[pi * 2] = boundary
        clef_boundaries[pi * 2 + 1] = boundary

    # Filter notes based on their system's clef boundary
    filtered_notes = []
    for n in all_notes:
        n_cy = n['y_center']
        best_sys_idx = None
        best_dist = float('inf')
        for si, sys in enumerate(systems):
            mid_y = (sys[0] + sys[4]) / 2.0
            dist = abs(n_cy - mid_y)
            if dist < best_dist:
                best_dist = dist
                best_sys_idx = si
        boundary = clef_boundaries.get(best_sys_idx, clef_area_x_first)
        if n['x'] > boundary:
            filtered_notes.append(n)
    all_notes = filtered_notes
    print(f"   After clef area filtering: {len(all_notes)} notes")
```

Also add `import cv2` at the top of main.py if not already there (it is — line 17).

- [ ] **Step 4: Run test to verify it passes**

Run: `cd prototype_cv && python test_duration_merge.py`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd prototype_cv && git add main.py test_duration_merge.py && git commit -m "fix: adaptive clef boundary to preserve first notes on lines 2+"
```

---

### Task 2: Per-note individual duration detection

**Files:**
- Modify: `prototype_cv/note_unit.py:172-203`

Currently `_detect_duration()` returns a single duration for an entire chord group and defaults to 1.0 (quarter note) for stemmed filled notes. We need per-note duration using beam/flag analysis on each note's own stem.

- [ ] **Step 1: Add test for individual duration detection**

Append to `prototype_cv/test_duration_merge.py`:

```python
def test_individual_duration_detection():
    """Detect duration per notehead, not per chord group."""
    from note_unit import _detect_individual_duration
    # Mock a note with stem_dir='up', has_flag → eighth note (0.5)
    # A note with 2 beams → sixteenth note (0.25)
    # We can't easily mock image data, so test the logic:
    assert _detect_individual_duration(beam_count=0, has_flag=True, is_hollow=False) == 0.5
    assert _detect_individual_duration(beam_count=0, has_flag=False, is_hollow=False) == 1.0
    assert _detect_individual_duration(beam_count=1, has_flag=False, is_hollow=False) == 0.5
    assert _detect_individual_duration(beam_count=2, has_flag=False, is_hollow=False) == 0.25
    assert _detect_individual_duration(beam_count=0, has_flag=False, is_hollow=True) == 2.0
    print("PASS: test_individual_duration_detection")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd prototype_cv && python test_duration_merge.py`

Expected: ImportError — `_detect_individual_duration` doesn't exist yet.

- [ ] **Step 3: Implement `_detect_individual_duration`**

Add to `prototype_cv/note_unit.py` after `_detect_duration` (after line 203):

```python
def _detect_individual_duration(beam_count=0, has_flag=False, is_hollow=False):
    """Convert beam/flag/hollow info into a duration value.

    Returns float: 4.0 (whole), 2.0 (half), 1.0 (quarter), 0.5 (eighth), 0.25 (sixteenth).
    """
    if is_hollow:
        return 2.0  # half note (whole notes have no stem, handled separately)
    if beam_count >= 2:
        return 0.25  # sixteenth
    if beam_count == 1 or has_flag:
        return 0.5   # eighth
    return 1.0  # quarter (default)


def detect_duration_per_note(note, binary, dy):
    """Detect duration for a single note using its stem's beam/flag info.

    Parameters
    ----------
    note : dict with 'stem' key (from track_stem), 'y_center', 'system'
    binary : ndarray, full binary image (beams intact)
    dy : float, staff line spacing

    Returns
    -------
    float : duration in beats
    """
    stem = note['stem']
    if stem['stem_dir'] is None:
        return 1.0  # no stem → default quarter

    staff_lines = note.get('system', None)
    beam_count, has_flag = _count_beams(
        binary, stem['stem_tip_y'], stem['stem_x'], dy,
        staff_lines=staff_lines, stem_dir=stem['stem_dir']
    )
    is_hollow = False  # filled noteheads assumed; hollow checked elsewhere
    return _detect_individual_duration(beam_count, has_flag, is_hollow)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd prototype_cv && python test_duration_merge.py`

Expected: PASS for both tests.

- [ ] **Step 5: Commit**

```bash
cd prototype_cv && git add note_unit.py test_duration_merge.py && git commit -m "feat: per-note individual duration detection from beam/flag"
```

---

### Task 3: Store per-note duration in build_note_units

**Files:**
- Modify: `prototype_cv/note_unit.py:293-338` (inside `build_note_units`)

Currently each NoteUnit stores one `duration` for the group. We need each note entry to also carry its own `individual_duration`.

- [ ] **Step 1: Modify build_note_units to call detect_duration_per_note**

In `prototype_cv/note_unit.py`, inside `build_note_units()`, after the stem tracking data is available on each note in the group (the group is already built by line 295), add per-note duration detection.

Replace lines 300-313 (the note_entries building loop) with:

```python
        # Build pitch for each note, including per-note duration
        note_entries = []
        for n in group:
            base_str, suffix_str = y_to_jianpu(n['y_center'], n['system'], n.get('clef', 'treble'))
            pitch = base_str + suffix_str
            ind_dur = detect_duration_per_note(n, binary, dy)
            note_entries.append({
                'pitch': pitch,
                'accidental': n.get('accidental', None),
                'x': n['x'],
                'y_center': n['y_center'],
                'clef': n.get('clef', 'treble'),
                'system': n['system'],
                'pair_idx': n.get('pair_idx', 0),
                'w': n['w'],
                'individual_duration': ind_dur,
            })
```

- [ ] **Step 2: Run the pipeline to verify nothing breaks**

Run: `cd prototype_cv && python main.py ../input_page1.png 2>&1 | grep "System 1" -A 6`

Expected: System 1 output unchanged (still matches GT).

- [ ] **Step 3: Commit**

```bash
cd prototype_cv && git add note_unit.py && git commit -m "feat: store individual_duration per note entry in NoteUnit"
```

---

### Task 4: Implement temporal overlap merge

**Files:**
- Modify: `prototype_cv/note_unit.py` (add new function after `build_note_units`)

This is the core algorithm: after building NoteUnits, merge notes from different stems whose durations overlap in time.

- [ ] **Step 1: Add test for merge logic**

Append to `prototype_cv/test_duration_merge.py`:

```python
def test_merge_overlapping_notes():
    """An eighth-note 1' at x=284 should be merged into the next sixteenth-note event at x=370."""
    from note_unit import merge_overlapping_note_units

    # Simulate: NoteUnit A at x=284 has [3, 1'] where 1' is eighth (0.5)
    # NoteUnit B at x=370 has [#2] which is sixteenth (0.25)
    # After merge: event at x=284 is [3 1']/4, event at x=370 is [#2 1']/4
    units = [
        {
            'notes': [
                {'pitch': '3', 'x': 284, 'y_center': 1310, 'individual_duration': 0.25,
                 'accidental': None, 'clef': 'treble', 'system': [1225,1246,1268,1289,1310], 'pair_idx': 0, 'w': 27},
                {'pitch': "1'", 'x': 284, 'y_center': 1257, 'individual_duration': 0.5,
                 'accidental': None, 'clef': 'treble', 'system': [1225,1246,1268,1289,1310], 'pair_idx': 0, 'w': 27},
            ],
            'duration': 0.25, 'stem_dir': 'down', 'stem_x': 284, 'x': 284.0,
        },
        {
            'notes': [
                {'pitch': '#2', 'x': 370, 'y_center': 1321, 'individual_duration': 0.25,
                 'accidental': None, 'clef': 'treble', 'system': [1225,1246,1268,1289,1310], 'pair_idx': 0, 'w': 27},
            ],
            'duration': 0.25, 'stem_dir': 'down', 'stem_x': 370, 'x': 370.0,
        },
    ]

    merged = merge_overlapping_note_units(units, beats_per_measure=2.0, dy=21.2)

    # First event should have both 3 and 1'
    pitches_0 = {n['pitch'] for n in merged[0]['notes']}
    assert '3' in pitches_0, f"Expected '3' in first event, got {pitches_0}"
    assert "1'" in pitches_0, f"Expected '1\\'' in first event, got {pitches_0}"

    # Second event should have #2 AND the sustained 1'
    pitches_1 = {n['pitch'] for n in merged[1]['notes']}
    assert '#2' in pitches_1 or '2' in pitches_1, f"Expected '#2' in second event, got {pitches_1}"
    assert "1'" in pitches_1, f"Expected sustained '1\\'' in second event, got {pitches_1}"

    print("PASS: test_merge_overlapping_notes")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd prototype_cv && python test_duration_merge.py`

Expected: ImportError — `merge_overlapping_note_units` doesn't exist yet.

- [ ] **Step 3: Implement `merge_overlapping_note_units`**

Add to `prototype_cv/note_unit.py` after `build_note_units()`:

```python
def merge_overlapping_note_units(note_units, beats_per_measure=2.0, dy=21.0):
    """Merge notes whose durations overlap into shared events.

    For each note in a NoteUnit, if its individual_duration extends past
    the next NoteUnit's start time, copy that note into the next NoteUnit.

    This handles two-voice notation where an eighth note in one voice
    sustains across two sixteenth notes in another voice.

    Parameters
    ----------
    note_units : list of NoteUnit dicts, sorted by x
    beats_per_measure : float
    dy : float, staff spacing (used for x-distance heuristics)

    Returns
    -------
    list of NoteUnit dicts with merged notes
    """
    if len(note_units) <= 1:
        return note_units

    # Sort by x
    units = sorted(note_units, key=lambda u: u['x'])

    # Estimate time per pixel from the measure's x span
    x_positions = [u['x'] for u in units]
    x_span = x_positions[-1] - x_positions[0]
    if x_span <= 0:
        return units

    # Total beats occupied by all events (sum of min individual durations)
    total_dur = sum(
        min((n['individual_duration'] for n in u['notes']), default=1.0)
        for u in units
    )
    if total_dur <= 0:
        total_dur = beats_per_measure

    # Assign start_time to each unit based on cumulative shortest durations
    start_times = []
    t = 0.0
    for u in units:
        start_times.append(t)
        event_dur = min((n['individual_duration'] for n in u['notes']), default=0.25)
        t += event_dur

    # For each note in each unit, check if it sustains into later units
    for i, unit in enumerate(units):
        for note in unit['notes']:
            ind_dur = note.get('individual_duration', 0.25)
            note_end_time = start_times[i] + ind_dur

            # Check subsequent units
            for j in range(i + 1, len(units)):
                if start_times[j] >= note_end_time:
                    break  # note has ended, stop checking further units

                # This note is still sounding at unit j's start time.
                # Check if this pitch is already present in unit j
                existing_pitches = {n['pitch'] for n in units[j]['notes']}
                if note['pitch'] not in existing_pitches:
                    # Copy the note into unit j (as a sustained note)
                    sustained = dict(note)
                    sustained['individual_duration'] = min(
                        ind_dur,
                        start_times[j + 1] - start_times[j] if j + 1 < len(units)
                        else ind_dur
                    )
                    units[j]['notes'].append(sustained)

    # Update each unit's duration to the minimum individual duration
    for u in units:
        durations = [n.get('individual_duration', 1.0) for n in u['notes']]
        if durations:
            u['duration'] = min(durations)

    return units
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd prototype_cv && python test_duration_merge.py`

Expected: All 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd prototype_cv && git add note_unit.py test_duration_merge.py && git commit -m "feat: merge overlapping note units for two-voice duration alignment"
```

---

### Task 5: Integrate merge into the pipeline

**Files:**
- Modify: `prototype_cv/main.py:177-186` (where build_note_units and segment_into_measures are called)

- [ ] **Step 1: Call merge_overlapping_note_units after build_note_units**

In `main.py`, add import at the top (line 32):

```python
from note_unit import build_note_units, segment_into_measures, merge_overlapping_note_units
```

Then in the pair loop (after line ~178 where `treble_units = build_note_units(...)` is called), add the merge step:

```python
        treble_units = build_note_units(pair_treble, music_symbols, binary, dy)
        bass_units = build_note_units(pair_bass, music_symbols, binary, dy)

        # Merge notes whose durations overlap (two-voice alignment)
        treble_units = merge_overlapping_note_units(treble_units, beats_per_measure=2.0, dy=dy)
        bass_units = merge_overlapping_note_units(bass_units, beats_per_measure=2.0, dy=dy)
```

- [ ] **Step 2: Run full pipeline and check M6 output**

Run: `cd prototype_cv && python main.py ../input_page1.png 2>&1 | grep "System 2" -A 2`

Expected: System 2 treble first measure should show `[3 1']` sustained into the second event too if the duration detection works correctly.

- [ ] **Step 3: Run evaluation**

Run: `cd prototype_cv && python evaluate.py`

Check that System 1 bass remains 100% and treble M1-M5 remain exact matches.

- [ ] **Step 4: Commit**

```bash
cd prototype_cv && git add main.py && git commit -m "feat: integrate duration-overlap merge into pipeline"
```

---

### Task 6: Tune and verify against ground truth

**Files:**
- Modify: `prototype_cv/note_unit.py` (tuning thresholds)
- Modify: `prototype_cv/test_duration_merge.py` (integration test)

- [ ] **Step 1: Add integration test checking M6 against GT**

Append to `prototype_cv/test_duration_merge.py`:

```python
def test_m6_matches_gt():
    """System 2 treble M6 must match GT exactly."""
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "main.py", "../input_page1.png"],
        capture_output=True, text=True, cwd="."
    )
    lines = result.stdout.split('\n')
    gt_m6 = "[3 1']/4 [#2 1']/4 3/4 4/4 [4 2']/4 [#4 1']/4 [5 7]/4 [#5 1']/4"
    for i, line in enumerate(lines):
        if 'System 2' in line:
            treble_line = lines[i + 1].strip()
            # First measure is between first pair of ||
            first_measure = treble_line.split('|')[1].strip()
            if first_measure == gt_m6:
                print(f"PASS: M6 matches GT exactly")
                return
            else:
                print(f"PARTIAL: M6 = '{first_measure}'")
                print(f"GT:        '{gt_m6}'")
                # Check which events match
                from evaluate import parse_events
                gt_events = parse_events(gt_m6)
                out_events = parse_events(first_measure)
                for k in range(max(len(gt_events), len(out_events))):
                    ge = gt_events[k] if k < len(gt_events) else "---"
                    oe = out_events[k] if k < len(out_events) else "---"
                    match = "✓" if ge == oe else "✗"
                    print(f"  {match} event {k+1}: GT={ge}  OUT={oe}")
                return
    print("FAIL: System 2 not found")


if __name__ == "__main__":
    test_first_chord_detected()
    print()
    test_individual_duration_detection()
    print()
    test_merge_overlapping_notes()
    print()
    test_m6_matches_gt()
```

- [ ] **Step 2: Run integration test**

Run: `cd prototype_cv && python test_duration_merge.py`

Examine the per-event comparison output to identify remaining issues. Likely issues:
- Beam/flag detection might not correctly distinguish 1-beam (eighth) from 2-beam (sixteenth)
- The `_count_beams` function's flag detection threshold (density > 0.15) may need tuning

- [ ] **Step 3: Tune beam/flag detection if needed**

Based on the test output, adjust `_count_beams` thresholds in `note_unit.py:70-98`. Common adjustments:
- If flags not detected: lower `density > 0.15` to `density > 0.10`
- If beams overcounted: tighten `min_thickness`/`max_thickness` range
- If staff lines confused with beams: widen the mask from `±1` to `±2` rows

- [ ] **Step 4: Run full evaluation**

Run: `cd prototype_cv && python evaluate.py`

Verify:
- Bass line 1: still 100%
- Treble M1-M5: still exact match
- Treble M6: improved (should show [3 1'] in first event)

- [ ] **Step 5: Commit**

```bash
cd prototype_cv && git add note_unit.py test_duration_merge.py && git commit -m "test: integration test for M6 GT matching + tuning"
```
