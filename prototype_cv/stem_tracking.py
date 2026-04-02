"""Stem tracking for musical noteheads.

Traces the stem of a musical note from its notehead bounding box
using the staff-removed binary image (music_symbols).
"""

import numpy as np


def track_stem(music_symbols, note, dy, binary=None):
    """Trace the stem of a musical note from its notehead.

    Parameters
    ----------
    music_symbols : np.ndarray
        Staff-removed binary image (uint8, white=255 is ink).
    note : dict
        Notehead info with keys: x, y, w, h, y_center, and optionally 'system'
        (list of 5 staff line y-positions).
    dy : float
        Staff line spacing in pixels.
    binary : np.ndarray, optional
        Original binary image (staff lines intact). Used as fallback when
        music_symbols has gaps from staff line removal.

    Returns
    -------
    dict with keys:
        stem_x      : int            – x coordinate of the stem
        stem_dir    : 'up'|'down'|None – stem direction
        stem_tip_y  : int            – y of the far end of the stem
        stem_length : float          – stem length in pixels
    """
    img_h, img_w = music_symbols.shape[:2]
    x, y, w, h, y_center = note['x'], note['y'], note['w'], note['h'], note['y_center']
    cx = x + w // 2

    # Build set of staff line rows (±2px) to forgive gaps from staff removal
    staff_line_rows = set()
    system = note.get('system', None)
    if system is not None:
        for line_y in system:
            for offset in range(-2, 3):
                row = int(line_y) + offset
                if 0 <= row < img_h:
                    staff_line_rows.add(row)

    stem_half = max(3, int(dy * 0.15))
    search_range = max(int(dy * 0.3), w // 2 + int(dy * 0.2))
    scan_dist = int(dy * 2)
    density_threshold = 0.3
    gap_tolerance = int(dy * 0.3)
    max_search = int(dy * 4)

    # --- Step 1: Find stem x ---
    # Search for the column with highest vertical density in a narrow strip.
    # Use max(above, below) density and scan far enough to get past chord noteheads.
    far_scan = int(dy * 3)
    best_x = cx
    best_density = -1.0

    for col in range(max(0, cx - search_range), min(img_w, cx + search_range + 1)):
        left = max(0, col - stem_half // 2)
        right = min(img_w, left + stem_half)
        if right <= left:
            continue

        # Check density above notehead (scan far to get past nearby noteheads)
        top_start = max(0, y - far_scan)
        top_end = y
        d_above = 0.0
        if top_end > top_start:
            region = music_symbols[top_start:top_end, left:right]
            d_above = np.count_nonzero(region) / max(1, region.size)

        # Check density below notehead
        bot_start = y + h
        bot_end = min(img_h, y + h + far_scan)
        d_below = 0.0
        if bot_end > bot_start:
            region = music_symbols[bot_start:bot_end, left:right]
            d_below = np.count_nonzero(region) / max(1, region.size)

        # Use max (stem goes in one direction) rather than combined
        density = max(d_above, d_below)
        if density > best_density:
            best_density = density
            best_x = col

    stem_x = best_x
    stem_left = max(0, stem_x - stem_half // 2)
    stem_right = min(img_w, stem_left + stem_half)

    # --- Step 2: Trace upward ---
    stem_top_y = y
    gap_count = 0
    for row in range(y - 1, max(0, y - max_search) - 1, -1):
        if stem_right <= stem_left:
            break
        strip = music_symbols[row, stem_left:stem_right]
        row_density = np.count_nonzero(strip) / max(1, strip.size)
        if row_density < density_threshold:
            # If this gap is at a staff line, check binary image instead
            if row in staff_line_rows and binary is not None:
                bin_strip = binary[row, stem_left:stem_right]
                bin_density = np.count_nonzero(bin_strip) / max(1, bin_strip.size)
                if bin_density >= density_threshold:
                    # Stem exists here in binary — staff removal caused the gap
                    gap_count = 0
                    stem_top_y = row
                    continue
            gap_count += 1
            if gap_count >= gap_tolerance:
                stem_top_y = row + gap_tolerance
                break
        else:
            gap_count = 0
            stem_top_y = row
    else:
        # Reached the search limit
        if gap_count < gap_tolerance:
            stem_top_y = max(0, y - max_search)

    # --- Step 3: Trace downward ---
    stem_bot_y = y + h
    gap_count = 0
    for row in range(y + h, min(img_h, y + h + max_search)):
        if stem_right <= stem_left:
            break
        strip = music_symbols[row, stem_left:stem_right]
        row_density = np.count_nonzero(strip) / max(1, strip.size)
        if row_density < density_threshold:
            # If this gap is at a staff line, check binary image instead
            if row in staff_line_rows and binary is not None:
                bin_strip = binary[row, stem_left:stem_right]
                bin_density = np.count_nonzero(bin_strip) / max(1, bin_strip.size)
                if bin_density >= density_threshold:
                    gap_count = 0
                    stem_bot_y = row
                    continue
            gap_count += 1
            if gap_count >= gap_tolerance:
                stem_bot_y = row - gap_tolerance
                break
        else:
            gap_count = 0
            stem_bot_y = row
    else:
        if gap_count < gap_tolerance:
            stem_bot_y = min(img_h - 1, y + h + max_search - 1)

    # --- Step 4: Determine direction ---
    up_length = y_center - stem_top_y
    down_length = stem_bot_y - y_center
    min_stem = dy * 1.5

    if up_length >= down_length:
        stem_length = float(up_length)
        stem_dir = 'up'
        stem_tip_y = stem_top_y
    else:
        stem_length = float(down_length)
        stem_dir = 'down'
        stem_tip_y = stem_bot_y

    if stem_length < min_stem:
        stem_dir = None
        stem_tip_y = y_center
        stem_length = 0.0

    return {
        'stem_x': int(stem_x),
        'stem_dir': stem_dir,
        'stem_tip_y': int(stem_tip_y),
        'stem_length': stem_length,
    }
