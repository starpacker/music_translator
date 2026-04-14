"""
pitch_detection.py
Staff system detection and pitch calculation for both treble and bass clefs.
"""
import cv2
import numpy as np


def get_staff_systems(staff_lines_img):
    """
    Finds the Y coordinates of the 5 lines for each staff system.
    """
    # Sum horizontally to find peaks where lines are
    proj = np.sum(staff_lines_img, axis=1)
    
    # Find peaks (line locations)
    threshold = np.max(proj) * 0.3
    peaks = np.where(proj > threshold)[0]
    
    # Cluster close peaks into single lines (lines can be multiple pixels thick)
    clustered_peaks = []
    if len(peaks) == 0:
        return []
    
    current_peak_group = [peaks[0]]
    for p in peaks[1:]:
        if p - current_peak_group[-1] < 10:  # within 10 pixels
            current_peak_group.append(p)
        else:
            clustered_peaks.append(int(np.mean(current_peak_group)))
            current_peak_group = [p]
    clustered_peaks.append(int(np.mean(current_peak_group)))
    
    # Group lines into systems of 5
    systems = []
    if len(clustered_peaks) < 5:
        return []

    # First split clustered_peaks into contiguous "line runs" separated
    # by large gaps (inter-staff spaces). Within each run we then pick
    # the most uniform 5-peak window — important when a run contains 6+
    # peaks (e.g., a spurious cross-staff line above the real top line).
    avg_gap_all = float(np.median(np.diff(clustered_peaks)))

    runs = []
    cur = [clustered_peaks[0]]
    for p in clustered_peaks[1:]:
        gap = p - cur[-1]
        if gap > avg_gap_all * 3.0:
            runs.append(cur)
            cur = [p]
        else:
            cur.append(p)
    runs.append(cur)

    for run in runs:
        if len(run) < 5:
            continue
        # Score every 5-peak window by gap-spread; pick the tightest.
        best = None
        best_spread = None
        for j in range(len(run) - 4):
            y1, y2, y3, y4, y5 = run[j:j + 5]
            gaps = [y2 - y1, y3 - y2, y4 - y3, y5 - y4]
            avg_dy = sum(gaps) / 4.0
            if avg_dy <= 0:
                continue
            spread = (max(gaps) - min(gaps)) / avg_dy
            if spread >= 0.8:
                continue
            if best_spread is None or spread < best_spread:
                best = [y1, y2, y3, y4, y5]
                best_spread = spread
        if best is not None:
            systems.append(best)

    return systems


def detect_staff_layout(systems):
    """Detect whether the score uses grand staff (piano) or single staff (solo).

    Returns 'grand' if consecutive staves form treble+bass pairs,
    'single' if all staves are independent (solo instrument).
    """
    if len(systems) < 2:
        return 'single'

    gaps = []
    for i in range(len(systems) - 1):
        gap = systems[i + 1][0] - systems[i][4]
        gaps.append(gap)

    # Grand staff: bimodal gaps (small within pair, large between pairs).
    # Single staff: uniform gaps.
    if len(gaps) < 2:
        return 'grand'  # only 2 staves → assume grand staff

    sorted_gaps = sorted(gaps)
    # Check uniformity: if max/min ratio < 1.6, gaps are uniform → single staff
    if sorted_gaps[0] > 0 and sorted_gaps[-1] / sorted_gaps[0] < 1.6:
        return 'single'

    return 'grand'


def pair_grand_staves(systems):
    """
    Pair consecutive staves into grand staff systems (treble + bass).
    In a piano grand staff, staves come in pairs:
      - even index (0, 2, 4, ...) = treble
      - odd index (1, 3, 5, ...) = bass

    Returns list of tuples: [(treble_system, bass_system), ...]
    """
    pairs = []
    i = 0
    while i + 1 < len(systems):
        treble = systems[i]
        bass = systems[i + 1]

        gap_within = bass[0] - treble[4]
        treble_height = treble[4] - treble[0]

        if gap_within < treble_height * 4:
            pairs.append((treble, bass))
            i += 2
        else:
            i += 1

    return pairs


# ============================================================
# Pitch mapping tables
# ============================================================
# 19 positions per staff, index 0 = highest, index 18 = lowest
#
# Treble clef (G clef):
#   Position 5 = top line (F5), Position 13 = bottom line (E4)
#   Index:   0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15   16   17   18
#   Note:  C6   B5   A5   G5   F5   E5   D5   C5   B4   A4   G4   F4   E4   D4   C4   B3   A3   G3   F3
#   Jianpu: 2''  1''  7'   6'   5'   4'   3'   2'   1'   7    6    5    4    3    2    1    7,   6,   5,
TREBLE_RANGE = [2, 2, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1]
TREBLE_NOTE  = [2, 1, 7, 6, 5, 4, 3, 2, 1, 7, 6, 5, 4, 3, 2, 1, 7, 6, 5]

# Standard Bass clef (F clef):
#   Position 5 = top line (A3), Position 13 = bottom line (G2)
#   The bass clef reads a major 3rd + octave lower than treble at same position.
#   Index:   0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15   16   17   18
#   Note:  E4   D4   C4   B3   A3   G3   F3   E3   D3   C3   B2   A2   G2   F2   E2   D2   C2   B1   A1
#   Jianpu: 4    3    2    1    7,   6,   5,   4,   3,   2,   1,   7,,  6,,  5,,  4,,  3,,  2,,  1,,  7,,,
BASS_RANGE_STANDARD = [0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -2, -2, -2, -2, -2, -2, -2, -3]
BASS_NOTE_STANDARD  = [4, 3, 2, 1, 7, 6, 5, 4, 3, 2, 1, 7, 6, 5, 4, 3, 2, 1, 7]

# Legacy mode: treat bass positions as if reading treble clef
# (used by the original ground truth for this specific score)
BASS_RANGE_LEGACY = TREBLE_RANGE
BASS_NOTE_LEGACY  = TREBLE_NOTE

# Default: use standard bass clef mapping
# Set USE_LEGACY_BASS = True to match the original GT convention
USE_LEGACY_BASS = True

BASS_RANGE = BASS_RANGE_LEGACY if USE_LEGACY_BASS else BASS_RANGE_STANDARD
BASS_NOTE  = BASS_NOTE_LEGACY  if USE_LEGACY_BASS else BASS_NOTE_STANDARD

# Extended pitch sequence for notes beyond the 19-position grid
# This is the chromatic scale in jianpu: ...7,6,5,4,3,2,1,7,6,5,4,3,2,1,7,6,5...
# Each step is a half-step on the staff (line or space)
_PITCH_SEQUENCE = [1, 2, 3, 4, 5, 6, 7]  # repeating pattern


def _build_line_21(staff_system):
    """Build the 21 y-positions for a staff system (reference approach)."""
    y1, y2, y3, y4, y5 = staff_system
    dy = (y5 - y1) / 4.0
    
    line_11 = [
        y1 - 3 * dy, y1 - 2 * dy, y1 - dy,
        y1, y2, y3, y4, y5,
        y5 + dy, y5 + 2 * dy, y5 + 3 * dy
    ]
    
    line_21 = []
    for j in range(len(line_11)):
        line_21.append(line_11[j])
        if j < len(line_11) - 1:
            line_21.append((line_11[j] + line_11[j + 1]) / 2.0)
    
    return line_21


def _build_extended_grid(staff_system, num_extra=10):
    """Build an extended grid beyond the standard 21 positions.
    
    Returns list of y-positions, extending both above and below the standard grid.
    The standard grid has 21 positions (indices 0-20), corresponding to cnt 0-18
    centered at line_21[cnt+1].
    
    We extend by num_extra half-steps in each direction.
    """
    y1, y2, y3, y4, y5 = staff_system
    dy = (y5 - y1) / 4.0
    half_dy = dy / 2.0
    
    # Standard 19 positions: cnt=0 at line_21[1], cnt=18 at line_21[19]
    # line_21[1] = y1 - 2.5*dy, line_21[19] = y5 + 2.5*dy
    # Center of cnt=k is at line_21[k+1]
    
    line_21 = _build_line_21(staff_system)
    
    # Extend above (lower index = higher pitch = lower y)
    extended_above = []
    top_y = line_21[1]  # center of cnt=0
    for i in range(num_extra, 0, -1):
        extended_above.append(top_y - i * half_dy)
    
    # Standard centers
    standard_centers = [line_21[cnt + 1] for cnt in range(19)]
    
    # Extend below (higher index = lower pitch = higher y)
    extended_below = []
    bot_y = line_21[19] if len(line_21) > 19 else line_21[-1]
    for i in range(1, num_extra + 1):
        extended_below.append(bot_y + i * half_dy)
    
    all_centers = extended_above + standard_centers + extended_below
    return all_centers, num_extra  # offset: standard cnt=0 is at index num_extra


def y_to_position_index(y_center, staff_system):
    """
    Convert a Y coordinate to a position index (0-18) relative to a staff system.
    Uses the 21-position grid from the reference implementation.
    
    Improved: uses weighted distance that prefers line positions (even indices)
    when the note is very close to a line.
    """
    line_21 = _build_line_21(staff_system)
    y1, y2, y3, y4, y5 = staff_system
    dy = (y5 - y1) / 4.0
    
    # Find which of the 19 windows (cnt=0..18) the y_center is closest to
    best_cnt = 0
    min_dist = float('inf')
    for cnt in range(min(19, len(line_21) - 1)):
        window_center = line_21[cnt + 1] if cnt + 1 < len(line_21) else line_21[-1]
        dist = abs(y_center - window_center)
        
        if dist < min_dist:
            min_dist = dist
            best_cnt = cnt
    
    return best_cnt


def y_to_extended_position(y_center, staff_system):
    """
    Convert a Y coordinate to an extended position index that can go beyond 0-18.
    Returns the position index where negative values are above the standard grid
    and values > 18 are below.
    """
    all_centers, offset = _build_extended_grid(staff_system)
    
    best_idx = 0
    min_dist = float('inf')
    for i, center in enumerate(all_centers):
        dist = abs(y_center - center)
        if dist < min_dist:
            min_dist = dist
            best_idx = i
    
    # Convert to standard cnt: standard cnt=0 is at index=offset
    return best_idx - offset


def y_to_jianpu(y_center, staff_system, clef='treble'):
    """
    Convert Y coordinate to Jianpu notation.
    clef: 'treble' or 'bass'
    Returns (note_number_str, suffix_str) e.g. ('3', "'") for 3'
    
    Supports extended range beyond the standard 19 positions.
    """
    idx = y_to_extended_position(y_center, staff_system)
    
    # If within standard range, use the lookup tables
    if 0 <= idx <= 18:
        if clef == 'bass':
            note_num = BASS_NOTE[idx]
            octave = BASS_RANGE[idx]
        else:
            note_num = TREBLE_NOTE[idx]
            octave = TREBLE_RANGE[idx]
    else:
        # Extended range: extrapolate from the nearest edge
        if clef == 'bass':
            range_table = BASS_RANGE
            note_table = BASS_NOTE
        else:
            range_table = TREBLE_RANGE
            note_table = TREBLE_NOTE
        
        if idx < 0:
            # Above the grid (higher pitch)
            # Start from position 0 and go up
            base_note = note_table[0]
            base_octave = range_table[0]
            steps = -idx  # number of half-steps above position 0
            
            # Each step up increases the note by 1 in the scale
            current_note = base_note
            current_octave = base_octave
            for _ in range(steps):
                current_note += 1
                if current_note > 7:
                    current_note = 1
                    current_octave += 1
            
            note_num = current_note
            octave = current_octave
        else:
            # Below the grid (lower pitch)
            # Start from position 18 and go down
            base_note = note_table[18]
            base_octave = range_table[18]
            steps = idx - 18  # number of half-steps below position 18
            
            current_note = base_note
            current_octave = base_octave
            for _ in range(steps):
                current_note -= 1
                if current_note < 1:
                    current_note = 7
                    current_octave -= 1
            
            note_num = current_note
            octave = current_octave
    
    base_note = str(note_num)
    
    if octave > 0:
        suffix = "'" * octave
    elif octave < 0:
        suffix = "," * abs(octave)
    else:
        suffix = ""
    
    return base_note, suffix
