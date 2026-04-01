# Duration-Aligned Event Merging Design

## Goal

Correctly translate two-voice staff notation to jianpu by detecting every notehead's position and duration, then merging overlapping notes into chord events based on temporal alignment.

## Problem

In the input score, two voices can share a staff:
- Voice A plays a longer note (e.g., eighth note, 0.5 beat)
- Voice B plays shorter notes simultaneously (e.g., two sixteenth notes, 0.25 beat each)

The image contains ONE notehead for voice A, but the jianpu output needs that note replicated across ALL shorter events it overlaps with:
- Image: `1'(0.5)` at x=284, `3(0.25)` at x=284, `#2(0.25)` at x=370
- Jianpu: `[3 1']/4  [#2 1']/4`

## Architecture

Three processing stages, each with a clear input/output:

### Stage 1: Complete Notehead Detection

**Input**: Binary image, staff systems, dy  
**Output**: List of `{x, y_center, score}` for every notehead in the image

Current issue: clef area filter at 10.5% still misses some notes. Need per-system boundary based on actual first-notehead detection, not a fixed percentage.

Fix: After detecting noteheads, find the actual key-sig right edge per system and use that + margin as the filter boundary. Alternatively, use the exclusion zone right edge.

### Stage 2: Per-Note Duration Detection

**Input**: Noteheads with stem info  
**Output**: Each notehead gets an independent `duration` value

Current approach detects duration per note-group (chord). Need to detect duration per INDIVIDUAL notehead based on:
- **Beam count**: 0 beams = quarter (1.0), 1 beam = eighth (0.5), 2 beams = sixteenth (0.25)
- **Flag detection**: Single flag = eighth, double flag = sixteenth
- **Hollow notehead**: Half note (2.0) or whole note (4.0)

Key change: a note with a single flag (eighth) gets duration=0.5, while beamed notes in the same beat group get duration=0.25. This distinction is critical for the alignment step.

### Stage 3: Temporal Alignment Merge

**Input**: All noteheads in a measure, each with `{x, y_center, pitch, duration}`  
**Output**: Ordered list of events, each event being a chord (list of pitches) with a duration

Algorithm:
1. Sort all noteheads by x position
2. Assign each notehead a `start_time` based on cumulative duration of preceding events in the same x-cluster
3. For each pair of noteheads: if `noteA.start_time + noteA.duration > noteB.start_time`, then noteA is still sounding when noteB starts
4. Group all notes sounding at each start_time into one event (chord)
5. The event duration = time until the next event's start_time

Simplified approach (sufficient for this score):
1. Cluster noteheads by x proximity into "columns"
2. Within each column, all notes form a chord; duration = minimum duration in the column
3. Between columns: if a note from column N has duration that extends past column N+1's x position, include it in column N+1's chord too
4. The "extends past" check: `note.duration > time_gap_to_next_column`

The time gap between columns is estimated from their x-spacing relative to the measure width.

## Files to Modify

- `main.py`: Fix clef area filtering (per-system boundary)
- `note_unit.py`: Refactor `build_note_units()` to detect per-note duration, add `merge_by_duration()` function for Stage 3
- `template_matching.py`: Ensure noteheads near clef boundary are not filtered

## Success Criteria

- M6 line 2: `[3 1']/4 [#2 1']/4 3/4 4/4 [4 2']/4 [#4 1']/4 [5 7]/4 [#5 1']/4` matches GT exactly
- All other measures that currently match GT continue to match
- System 1 treble+bass remains 100%
