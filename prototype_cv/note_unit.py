"""
note_unit.py
Build NoteUnit groups (chords sharing a stem) with duration detection,
then segment into measures by barline positions.
"""
import cv2
import numpy as np
from pitch_detection import y_to_jianpu


def _is_hollow(music_symbols, note, dy):
    """Check if a notehead is hollow by examining fill ratio of center region."""
    cx = note['x'] + note['w'] // 2
    cy = note['y_center']
    r = max(1, int(dy * 0.2))
    h, w = music_symbols.shape[:2]
    y1 = max(0, cy - r)
    y2 = min(h, cy + r)
    x1 = max(0, cx - r)
    x2 = min(w, cx + r)
    if y2 <= y1 or x2 <= x1:
        return False
    region = music_symbols[y1:y2, x1:x2]
    if region.size == 0:
        return False
    fill_ratio = np.count_nonzero(region > 127) / region.size
    return fill_ratio <= 0.4


def _count_beams(binary, tip_y, stem_x, dy, staff_lines=None, stem_dir=None,
                 music_symbols=None, other_noteheads=None):
    """
    Count horizontal beams near the stem tip using horizontal projection.

    other_noteheads: list of (x, y, w, h) for noteheads of OTHER notes
        on the same staff. These are masked out of the projection to
        prevent adjacent noteheads from being counted as beams.

    Returns (beam_count, has_flag).
    """
    h_img, w_img = binary.shape[:2]

    if stem_dir == 'down':
        roi_y1 = max(0, int(tip_y - dy * 1.5))
        roi_y2 = min(h_img, int(tip_y + dy * 1.5))
    elif stem_dir == 'up':
        roi_y1 = max(0, int(tip_y - dy * 1.5))
        roi_y2 = min(h_img, int(tip_y + dy * 0.5))
    else:
        roi_y1 = max(0, int(tip_y - dy * 1.5))
        roi_y2 = min(h_img, int(tip_y + dy * 1.5))

    roi_x1 = max(0, int(stem_x - dy * 1.3))
    roi_x2 = min(w_img, int(stem_x + dy * 1.3))

    if roi_y2 <= roi_y1 or roi_x2 <= roi_x1:
        return 0, False

    roi = binary[roi_y1:roi_y2, roi_x1:roi_x2]
    roi_h, roi_w = roi.shape[:2]
    if roi_h == 0 or roi_w == 0:
        return 0, False

    # Build notehead mask for later per-band verification.
    # For each row, compute how much ink is from known noteheads vs beams.
    nh_mask = np.zeros((roi_h, roi_w), dtype=bool)
    if other_noteheads:
        # Use generous vertical pad to cover stem transitions and
        # the full notehead ink extent (detection bbox can be smaller).
        pad_y = max(4, int(dy * 0.4))
        pad_x = max(2, int(dy * 0.15))
        for nx, ny, nw, nh in other_noteheads:
            ly1 = max(0, ny - pad_y - roi_y1)
            ly2 = min(roi_h, ny + nh + pad_y - roi_y1)
            lx1 = max(0, nx - pad_x - roi_x1)
            lx2 = min(roi_w, nx + nw + pad_x - roi_x1)
            if ly1 < ly2 and lx1 < lx2:
                nh_mask[ly1:ly2, lx1:lx2] = True

    projection = np.count_nonzero(roi > 127, axis=1) / roi_w

    # Mask staff line rows on the binary projection
    mask_radius = max(3, int(dy * 0.2))
    masked_rows = set()
    if staff_lines is not None:
        for line_y in staff_lines:
            for offset in range(-mask_radius, mask_radius + 1):
                local_y = int(line_y) + offset - roi_y1
                if 0 <= local_y < roi_h:
                    projection[local_y] = 0.0
                    masked_rows.add(local_y)

    beam_threshold = 0.25
    min_thickness = max(1, int(dy * 0.08))
    max_thickness = max(min_thickness + 1, int(dy * 0.55))

    _debug_beam = False  # set True for beam detection debug

    # Find bands above threshold
    bands = []
    i = 0
    while i < len(projection):
        if projection[i] > beam_threshold:
            start = i
            while i < len(projection) and projection[i] > beam_threshold:
                i += 1
            bands.append((start, i))
        else:
            i += 1

    if _debug_beam:
        print(f"    [BEAM DEBUG] raw bands={bands} (abs_y={[(s+roi_y1, e+roi_y1) for s,e in bands]})")

    beam_count = 0
    counted_bands = []
    stem_local_x = stem_x - roi_x1  # stem position within ROI
    for start, end in bands:
        thickness = end - start
        if min_thickness <= thickness <= max_thickness:
            # Reject 1px bands adjacent to masked regions
            if thickness <= 1:
                if start - 1 in masked_rows or end in masked_rows:
                    continue
            # Verify: a beam should extend on both sides of the stem, or
            # at minimum have significant ink AT the stem. A nearby notehead
            # entirely on one side creates a band that happens to reach the
            # stem edge but has NO ink beyond it.
            mid_row = (start + end) // 2
            stem_col = max(0, min(roi_w - 1, stem_local_x))
            row_pixels = roi[mid_row, :]
            # Count ink pixels on each side of stem (excluding stem itself ±3px)
            margin = 3
            left_ink = np.sum(row_pixels[:max(0, stem_col - margin)] > 127)
            right_ink = np.sum(row_pixels[min(roi_w, stem_col + margin + 1):] > 127)
            # A beam extends at least ~0.3*dy (6px) from the stem on at
            # least one side. Very low ink (<0.3*dy) means the band is
            # just the stem or a tiny notehead fragment, not a real beam.
            min_beam_extent = max(3, int(dy * 0.3))
            if left_ink < min_beam_extent and right_ink < min_beam_extent:
                continue  # insufficient horizontal extent for a beam
            # Notehead check: a beam extends wider than any notehead.
            # If removing notehead-region ink leaves insufficient horizontal
            # extent, the band is a notehead, not a beam.
            if np.any(nh_mask[start:end, :]):
                mid_row = (start + end) // 2
                beam_ink = (roi[mid_row, :] > 127) & ~nh_mask[mid_row, :]
                # After removing notehead ink, the beam should still
                # extend at least min_beam_extent from the stem
                left_beam = np.sum(beam_ink[:max(0, stem_local_x - 3)])
                right_beam = np.sum(beam_ink[min(roi_w, stem_local_x + 4):])
                if left_beam < min_beam_extent and right_beam < min_beam_extent:
                    continue  # no beam remaining after notehead removal
            beam_count += 1
            counted_bands.append((start, end))
            if _debug_beam:
                print(f"    [BEAM DEBUG] counted band ({start},{end}) abs_y=({start+roi_y1},{end+roi_y1}) "
                      f"thick={thickness} left={left_ink} right={right_ink}")

    # Validate beam gaps: genuine double beams (sixteenths) are closely
    # spaced (~0.2-0.35*dy apart). Larger gaps indicate beam-angle artifacts
    # at beam group endpoints. Keep only the band closest to tip_y.
    if beam_count >= 2 and len(counted_bands) >= 2:
        max_beam_gap = dy * 0.4
        tip_local = tip_y - roi_y1
        filtered = [counted_bands[0]]
        for bi in range(1, len(counted_bands)):
            prev_end = counted_bands[bi - 1][1]
            curr_start = counted_bands[bi][0]
            gap = curr_start - prev_end
            if gap > max_beam_gap:
                # Keep the band closest to tip_y
                prev_mid = (counted_bands[bi - 1][0] + counted_bands[bi - 1][1]) / 2
                curr_mid = (counted_bands[bi][0] + counted_bands[bi][1]) / 2
                if abs(curr_mid - tip_local) < abs(prev_mid - tip_local):
                    filtered[-1] = counted_bands[bi]
                # else keep previous, skip current
            else:
                filtered.append(counted_bands[bi])
        if len(filtered) < len(counted_bands):
            beam_count = len(filtered)
            counted_bands = filtered

    if _debug_beam:
        print(f"    [BEAM DEBUG] beam_count after binary={beam_count} bands={counted_bands}")

    # Cross-check with music_symbols: if any staff line is within
    # the ROI, the mask may have hidden beams. Use music_symbols
    # (staff lines removed) to detect beams in the masked region.
    # Only count beams that are entirely WITHIN the masked zone
    # (not overlapping with already-detected binary bands).
    if music_symbols is not None and staff_lines is not None and masked_rows:
        ms_roi = music_symbols[roi_y1:roi_y2, roi_x1:roi_x2]
        ms_proj = np.count_nonzero(ms_roi > 127, axis=1) / roi_w
        # Only check the MASKED region
        margin = 2
        mask_min = max(0, min(masked_rows) - margin)
        mask_max = min(roi_h, max(masked_rows) + margin + 1)
        ms_threshold = 0.15
        ms_bands = []
        j = mask_min
        while j < mask_max:
            if ms_proj[j] > ms_threshold:
                s = j
                while j < mask_max and ms_proj[j] > ms_threshold:
                    j += 1
                ms_bands.append((s, j))
            else:
                j += 1
        # Merge bands with small gaps
        if len(ms_bands) > 1:
            merged = [ms_bands[0]]
            for bs, be in ms_bands[1:]:
                if bs - merged[-1][1] <= 3:
                    merged[-1] = (merged[-1][0], be)
                else:
                    merged.append((bs, be))
            ms_bands = merged
        if _debug_beam:
            print(f"    [BEAM DEBUG] MS masked zone=[{mask_min}:{mask_max}] ms_bands={ms_bands} "
                  f"abs_y={[(s+roi_y1, e+roi_y1) for s,e in ms_bands]}")
        # Only add at most 1 MS-recovered beam, and only when binary
        # found few beams (0-1). Verify the beam reaches the stem.
        if beam_count <= 1:
            ms_stem_local = stem_x - roi_x1
            for s, e in ms_bands:
                thickness = e - s
                if min_thickness <= thickness <= max_thickness:
                    overlaps = any(not (e <= bs or s >= be) for bs, be in counted_bands)
                    if not overlaps:
                        # Verify beam reaches the stem on music_symbols.
                        # Check ANY row in the band (not just mid — the
                        # mid may fall on a staff-line gap).
                        ms_check_hw = max(3, int(dy * 0.3))
                        ms_sx1 = max(0, ms_stem_local - ms_check_hw)
                        ms_sx2 = min(roi_w, ms_stem_local + ms_check_hw + 1)
                        reaches_stem = False
                        for ms_r in range(s, e):
                            if 0 <= ms_r < ms_roi.shape[0]:
                                if np.sum(ms_roi[ms_r, ms_sx1:ms_sx2] > 127) >= 2:
                                    reaches_stem = True
                                    break
                        if not reaches_stem:
                            continue  # beam doesn't reach the stem
                        # Validate gap to existing beams — reject if too far
                        if counted_bands:
                            mid_new = (s + e) / 2
                            nearest = min(abs(mid_new - (cb[0]+cb[1])/2)
                                          for cb in counted_bands)
                            if nearest > dy * 0.8:
                                continue  # too far from existing beams
                        beam_count += 1
                        if _debug_beam:
                            print(f"    [BEAM DEBUG] MS recovered band ({s},{e}) abs_y=({s+roi_y1},{e+roi_y1})")
                        break  # at most 1 recovery

    # Check for flag if no beams found.
    # Flags extend horizontally from the stem tip. To avoid counting the
    # stem itself as flag ink, mask out a narrow vertical strip centered
    # on the stem and measure density in the remaining area.
    has_flag = False
    if beam_count == 0:
        flag_x1 = max(0, int(stem_x - dy * 0.2))
        flag_x2 = min(w_img, int(stem_x + dy * 1.5))
        if stem_dir == 'up':
            flag_y1 = max(0, int(tip_y - dy * 0.2))
            flag_y2 = min(h_img, int(tip_y + dy * 1.5))
        elif stem_dir == 'down':
            flag_y1 = max(0, int(tip_y - dy * 1.5))
            flag_y2 = min(h_img, int(tip_y + dy * 0.2))
        else:
            flag_y1 = max(0, int(tip_y - dy * 0.6))
            flag_y2 = min(h_img, int(tip_y + dy * 0.6))
        if flag_y2 > flag_y1 and flag_x2 > flag_x1:
            flag_roi = binary[flag_y1:flag_y2, flag_x1:flag_x2].copy()
            if flag_roi.size > 0:
                # Mask out the stem column (±3px) to avoid counting stem ink
                stem_local = stem_x - flag_x1
                stem_hw = max(3, int(dy * 0.15))
                s_col1 = max(0, stem_local - stem_hw)
                s_col2 = min(flag_roi.shape[1], stem_local + stem_hw + 1)
                flag_roi[:, s_col1:s_col2] = 0
                total_pixels = flag_roi.size - flag_roi.shape[0] * (s_col2 - s_col1)
                if total_pixels > 0:
                    density = np.count_nonzero(flag_roi > 127) / total_pixels
                    from config import CFG
                    has_flag = density > CFG.beam.flag_density_threshold

    return beam_count, has_flag


def _count_beams_along_stem(binary, notes_in_group, stem_x, dy, staff_lines):
    """Count beams by scanning the full stem column, masking noteheads and staff lines.

    Looks at a narrow vertical strip centered on stem_x, spanning the full stem
    extent. Masks out notehead bounding boxes and staff line positions. Remaining
    horizontal bands with density > threshold are beams.
    """
    img_h, img_w = binary.shape[:2]
    half_w = int(dy * 3.0)  # wide ROI to capture beam from adjacent stems
    x1 = max(0, stem_x - half_w)
    x2 = min(img_w, stem_x + half_w)
    if x2 <= x1:
        return 0

    # Determine stem extent from all traced tips
    all_tips = [n['stem']['stem_tip_y'] for n in notes_in_group
                if n['stem']['stem_dir'] is not None]
    all_y_centers = [n['y_center'] for n in notes_in_group]

    if not all_tips:
        return 0

    y_min = min(min(all_tips), min(all_y_centers)) - int(dy * 0.5)
    y_max = max(max(all_tips), max(all_y_centers)) + int(dy * 0.5)
    y_min = max(0, y_min)
    y_max = min(img_h, y_max)

    if y_max <= y_min:
        return 0

    # Compute horizontal projection along the stem column
    roi = binary[y_min:y_max, x1:x2]
    projection = np.count_nonzero(roi > 127, axis=1) / max(1, roi.shape[1])

    # Do NOT mask noteheads — beam rows have higher density than notehead-only
    # rows because beams extend wider. Only mask staff lines (density ~1.0).
    # Mask staff lines
    if staff_lines is not None:
        for line_y in staff_lines:
            for offset in range(-1, 2):
                row = int(line_y) + offset - y_min
                if 0 <= row < len(projection):
                    projection[row] = 0.0

    # Count beam-like bands. With wide ROI (6*dy ≈ 127px):
    # - notehead-only rows: ~0.21 density (27px notehead in 127px ROI)
    # - beam rows: ~0.50-0.70 (beam spans 60-90px)
    # - staff lines: ~1.0 (already masked)
    # Threshold 0.35 reliably separates noteheads from beams.
    beam_threshold = 0.35
    min_thickness = max(1, int(dy * 0.08))
    max_thickness = int(dy * 0.45)

    beam_count = 0
    i = 0
    while i < len(projection):
        if projection[i] > beam_threshold:
            start = i
            while i < len(projection) and projection[i] > beam_threshold:
                i += 1
            thickness = i - start
            if min_thickness <= thickness <= max_thickness:
                beam_count += 1
        else:
            i += 1

    return beam_count


def _detect_duration(music_symbols, binary, notes_in_group, dy):
    """Detect duration of a note group from stem tip morphology.

    Strategy:
    1. Check hollow (open) noteheads -> whole/half note
    2. Determine chord stem direction from staff position
    3. Look for beams PAST the outermost notehead in the stem direction
    """
    has_stem = any(n['stem']['stem_dir'] is not None for n in notes_in_group)

    if not has_stem:
        hollow = any(_is_hollow(music_symbols, n, dy) for n in notes_in_group)
        return 4.0 if hollow else 1.0

    hollow = any(_is_hollow(music_symbols, n, dy) for n in notes_in_group)
    if hollow:
        return 2.0

    # Determine chord stem direction from staff position
    system = notes_in_group[0].get('system')
    if system is not None and len(system) >= 5:
        mid_line = system[2]
        avg_y = np.mean([n['y_center'] for n in notes_in_group])
        chord_stem_dir = 'down' if avg_y <= mid_line else 'up'
    else:
        stemmed = [n for n in notes_in_group if n['stem']['stem_dir'] is not None]
        chord_stem_dir = max(stemmed, key=lambda n: n['stem']['stem_length'])['stem']['stem_dir'] if stemmed else 'down'

    # Duration will be refined later by estimate_durations_in_measures()
    # which combines beam detection with proportional spacing.
    # For now, return a placeholder (1.0 = quarter note default).
    return 1.0



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


def _detect_dot(binary, note, dy, all_notes=None):
    """Detect augmentation dot to the right of a notehead.

    Augmentation dots are small filled circles (~0.3*dy diameter) placed
    in the space to the right of the notehead. They multiply the note
    duration by 1.5.

    Uses the full binary image (not music_symbols) and strict filtering
    to avoid false positives from staccato dots, accidentals, etc.

    Parameters
    ----------
    binary : ndarray, binary image
    note : dict with x, y_center, w, h
    dy : float, staff line spacing
    all_notes : list of note dicts, for overlap checking

    Returns True if a dot is found.
    """
    cx = note['x'] + note['w'] // 2
    cy = note['y_center']
    h_img, w_img = binary.shape[:2]

    # Search region: to the right of the notehead, with padding for
    # dots at the edge. Dot diameter is ~0.35*dy, so add 0.5*dy padding.
    x_start = cx + int(dy * 0.6)
    x_end = min(w_img, cx + int(dy * 2.0))
    # Vertical: dots are placed in the nearest space, so ±0.4*dy
    y_start = max(0, cy - int(dy * 0.5))
    y_end = min(h_img, cy + int(dy * 0.5))

    if x_end <= x_start or y_end <= y_start:
        return False

    # If another notehead overlaps the search area, skip
    # (the "dot" might be part of the next note's edge)
    if all_notes is not None:
        for other in all_notes:
            if other is note:
                continue
            # Use left edge of other notehead (not center)
            o_left = other['x']
            oy = other['y_center']
            if o_left <= x_end and o_left + other['w'] >= x_start and abs(oy - cy) < dy * 1.5:
                return False

    roi = binary[y_start:y_end, x_start:x_end]
    if roi.size == 0:
        return False

    roi_bin = (roi > 127).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        roi_bin, connectivity=8)

    for label_id in range(1, num_labels):
        bw = stats[label_id, cv2.CC_STAT_WIDTH]
        bh = stats[label_id, cv2.CC_STAT_HEIGHT]
        area = stats[label_id, cv2.CC_STAT_AREA]

        # Dot size: 0.25-0.45*dy, roughly circular
        min_size = max(4, int(dy * 0.25))
        max_size = max(min_size + 1, int(dy * 0.45))

        if not (min_size <= bw <= max_size and min_size <= bh <= max_size):
            continue

        # Circularity
        aspect = max(bw, bh) / max(1, min(bw, bh))
        if aspect > 1.6:
            continue

        # Compactness: dot is filled circle → high fill ratio
        bbox_area = bw * bh
        if bbox_area > 0 and area / bbox_area < 0.50:
            continue

        # Size upper bound
        if area > dy * dy * 0.15:
            continue

        # Reject blobs touching the RIGHT or BOTTOM boundary of the ROI —
        # they're likely clipped noteheads from adjacent notes.
        # Top/left boundaries are OK (dot can be above notehead center).
        blob_left = stats[label_id, cv2.CC_STAT_LEFT]
        blob_top = stats[label_id, cv2.CC_STAT_TOP]
        roi_h, roi_w = roi_bin.shape
        if blob_left + bw >= roi_w - 1 or blob_top + bh >= roi_h - 1:
            continue

        return True

    return False


def detect_duration_per_note(note, binary, dy, music_symbols=None, all_notes=None,
                             other_noteheads=None):
    """Detect duration for a single note using its stem's beam/flag info.

    Parameters
    ----------
    note : dict with 'stem' key (from track_stem), 'y_center', 'system'
    binary : ndarray, full binary image (beams intact)
    dy : float, staff line spacing
    all_notes : list of note dicts, for dot overlap checking
    other_noteheads : list of (x, y, w, h) for OTHER noteheads on same staff,
        used to mask them from beam detection ROI

    Returns
    -------
    float : duration in beats
    """
    # Hollow template detections have score in [0.80, 0.85) range.
    # For stemless notes (whole notes), trust the score — _is_hollow may
    # fail because staff lines pass through the center of the oval.
    # For stemmed notes, always verify with _is_hollow to avoid
    # misclassifying low-confidence filled noteheads as hollow.
    score = note.get('score', 1.0)
    likely_hollow_template = 0.78 <= score < 0.85

    stem = note['stem']
    if stem['stem_dir'] is None:
        if likely_hollow_template:
            return 4.0  # whole note (no stem + hollow template match)
        is_hollow = _is_hollow(music_symbols, note, dy) if music_symbols is not None else False
        return 4.0 if is_hollow else 1.0

    staff_lines = note.get('system', None)
    # Use effective beam-search center. The tracked stem tip can overshoot
    # through beams and adjacent noteheads. For beam counting, search
    # near where the stem exits the staff — that's where beams are.
    tip_y = stem['stem_tip_y']
    if staff_lines is not None and len(staff_lines) >= 5:
        staff_top, staff_bot = staff_lines[0], staff_lines[4]
        note_y = note['y_center']
        if stem['stem_dir'] == 'down':
            # Beams are below the notehead, near or past the bottom staff line.
            # Use the midpoint between the notehead and the bottom line,
            # or the tracked tip if it's within the staff range.
            expected_beam_y = max(note_y, staff_bot)
            tip_y = min(tip_y, expected_beam_y + int(dy * 0.5))
        elif stem['stem_dir'] == 'up':
            expected_beam_y = min(note_y, staff_top)
            tip_y = max(tip_y, expected_beam_y - int(dy * 0.5))
    beam_count, has_flag = _count_beams(
        binary, tip_y, stem['stem_x'], dy,
        staff_lines=staff_lines, stem_dir=stem['stem_dir'],
        music_symbols=music_symbols, other_noteheads=other_noteheads
    )
    note['beam_count'] = beam_count
    note['has_flag'] = has_flag
    is_hollow = _is_hollow(music_symbols, note, dy) if music_symbols is not None else False
    dur = _detect_individual_duration(beam_count, has_flag, is_hollow)

    # Check for augmentation dot (multiplies duration by 1.5)
    if _detect_dot(binary, note, dy, all_notes=all_notes):
        dur *= 1.5

    return dur


def build_note_units(notes, music_symbols, binary, dy, single_staff=False):
    """
    Group notes by shared stem and detect duration for each group.

    Parameters
    ----------
    notes : list of dict
        Each note has: x, y, w, h, y_center, score, clef, system, pair_idx,
        and a 'stem' dict with stem_x, stem_dir, stem_tip_y, stem_length.
    music_symbols : ndarray
        Binary image with staff lines removed (white=255 foreground).
    dy : float
        Staff line spacing in pixels.

    Returns
    -------
    list of NoteUnit dicts
    """
    if not notes:
        return []

    stem_threshold = dy * 0.5
    x_threshold = dy * 0.9  # notehead x-proximity

    # Pre-filter false positives BEFORE grouping to avoid corrupting chord groups.
    # Notes with very low fill on music_symbols are accidentals/symbols, not noteheads.
    # Notes far from the staff require higher fill (dynamic markings, ornaments, etc.).
    verified_notes = []
    for n in notes:
        cx = n['x'] + n['w'] // 2
        cy = n['y_center']
        # Use a larger pad to capture the outline of hollow noteheads.
        # Center fill may be low (hollow), but outline fill should be non-zero.
        pad = max(2, int(dy * 0.15))
        pad_large = max(4, int(dy * 0.4))
        ny1 = max(0, cy - pad)
        ny2 = min(music_symbols.shape[0], cy + pad)
        nx1 = max(0, cx - pad)
        nx2 = min(music_symbols.shape[1], cx + pad)
        region = music_symbols[ny1:ny2, nx1:nx2]
        if region.size > 0:
            fill = np.mean(region) / 255.0
            # Check larger region as secondary fill check
            fill_large = fill  # default to same as center fill

            # Position-aware threshold: notes far from staff need higher fill
            sys = n.get('system')
            min_fill = 0.25
            if sys is not None:
                below = cy - sys[4]
                above = sys[0] - cy
                if below > dy * 1.5 or above > dy * 3:
                    min_fill = 0.45
            # Accept if center fill OR outline fill is sufficient.
            # Also check for hollow noteheads (whole/half notes): low center
            # fill but significant ring density from the notehead outline.
            if fill < min_fill and fill_large < min_fill:
                ring_pad = max(5, int(dy * 0.5))
                ry1 = max(0, cy - ring_pad)
                ry2 = min(music_symbols.shape[0], cy + ring_pad)
                rx1 = max(0, cx - ring_pad)
                rx2 = min(music_symbols.shape[1], cx + ring_pad)
                ring_region = music_symbols[ry1:ry2, rx1:rx2]
                ring_fill = np.mean(ring_region) / 255.0 if ring_region.size > 0 else 0
                # Hollow noteheads have ring_fill ≥ 0.20; reject lower values
                # Hollow noteheads detected by template matching (score≈0.80)
                # may have very low ring fill due to staff line removal.
                # Use a lower threshold for notes with hollow-detection scores.
                ring_threshold = 0.15 if n.get('score', 1.0) < 0.85 else 0.24
                if ring_fill < ring_threshold:
                    continue

            # Beam artifact filter: beams near staff lines look like noteheads
            # but extend horizontally much further than a real notehead (~1.3*dy).
            # Measure horizontal run-length of filled pixels at the note center;
            # if it exceeds 2*dy, the detection is a beam, not a notehead.
            if sys is not None:
                near_staff = any(abs(cy - sl) < dy * 0.5 for sl in sys)
                if near_staff:
                    img_w = music_symbols.shape[1]
                    row_y1 = max(0, cy - 1)
                    row_y2 = min(music_symbols.shape[0], cy + 2)
                    # Measure run-length left
                    lx = cx
                    while lx > 0:
                        col_strip = music_symbols[row_y1:row_y2, lx - 1:lx]
                        if col_strip.size == 0 or np.mean(col_strip) / 255.0 < 0.3:
                            break
                        lx -= 1
                    # Measure run-length right
                    rx = cx
                    while rx < img_w - 1:
                        col_strip = music_symbols[row_y1:row_y2, rx:rx + 1]
                        if col_strip.size == 0 or np.mean(col_strip) / 255.0 < 0.3:
                            break
                        rx += 1
                    run_length = rx - lx
                    if run_length > dy * 2.0:
                        continue  # beam artifact, not a notehead

            verified_notes.append(n)
        else:
            verified_notes.append(n)
    notes = verified_notes

    # Build chord groups by stem_x proximity OR notehead x-proximity, same system
    used = [False] * len(notes)
    groups = []

    # Sort by notehead x for stable grouping
    indexed = sorted(enumerate(notes), key=lambda t: t[1]['x'])

    for i, (idx_a, note_a) in enumerate(indexed):
        if used[idx_a]:
            continue
        group = [note_a]
        used[idx_a] = True
        sys_a = tuple(note_a['system']) if note_a.get('system') is not None else None

        for j in range(i + 1, len(indexed)):
            idx_b, note_b = indexed[j]
            if used[idx_b]:
                continue
            sys_b = tuple(note_b['system']) if note_b.get('system') is not None else None
            if sys_a != sys_b:
                continue

            # Check stem_x proximity (primary) OR notehead x proximity (fallback)
            stem_close = abs(note_b['stem']['stem_x'] - note_a['stem']['stem_x']) < stem_threshold
            x_close = abs(note_b['x'] - note_a['x']) < x_threshold

            # In single-staff scores (solo instruments), opposite stem
            # directions indicate sequential notes, not chords. In grand
            # staff scores, two-voice chords have opposite stems.
            if single_staff:
                dir_a = note_a['stem']['stem_dir']
                dir_b = note_b['stem']['stem_dir']
                if dir_a is not None and dir_b is not None and dir_a != dir_b:
                    continue

            # Check y-distance: new note must be within chord_y_range of at
            # least one existing member (not just the first).
            from config import CFG
            min_dist = min(abs(note_b['y_center'] - g['y_center']) for g in group)
            y_ok = min_dist <= dy * CFG.chord.y_range_max_dy

            if (stem_close or x_close) and y_ok:
                group.append(note_b)
                used[idx_b] = True
            elif note_b['x'] - note_a['x'] >= x_threshold:
                break  # sorted by x, no more matches

        groups.append(group)

    # Second pass: merge single-note groups into nearby larger groups.
    # Handles ordering issues where a note formed its own group because
    # the target group hadn't grown enough when it was first checked.
    for g_small in list(groups):
        if len(g_small) != 1:
            continue
        note_n = g_small[0]
        sys_n = tuple(note_n['system']) if note_n.get('system') is not None else None
        best_group = None
        best_dist = float('inf')
        for g_big in groups:
            if g_big is g_small or len(g_big) < 2:
                continue
            sys_g = tuple(g_big[0]['system']) if g_big[0].get('system') is not None else None
            if sys_n != sys_g:
                continue
            x_dist = abs(note_n['x'] - np.mean([n['x'] for n in g_big]))
            if x_dist >= x_threshold:
                continue
            # In single-staff mode, don't merge opposite stem directions
            if single_staff:
                dir_n = note_n['stem']['stem_dir']
                if dir_n is not None:
                    g_dirs = {n['stem']['stem_dir'] for n in g_big}
                    if dir_n not in g_dirs and None not in g_dirs:
                        continue
            y_dist = min(abs(note_n['y_center'] - n['y_center']) for n in g_big)
            if y_dist <= dy * CFG.chord.y_range_max_dy and y_dist < best_dist:
                best_dist = y_dist
                best_group = g_big
        if best_group is not None:
            best_group.append(note_n)
            groups.remove(g_small)

    # Build NoteUnit for each group
    note_units = []
    for group in groups:
        # Sort by y_center descending (low pitch first = higher y value first)
        group.sort(key=lambda n: -n['y_center'])

        # Build list of OTHER noteheads (not in this group) for beam masking
        group_ids = {id(n) for n in group}
        other_noteheads = [(n['x'], n['y'], n['w'], n['h'])
                           for n in notes if id(n) not in group_ids]

        # Build pitch for each note
        note_entries = []
        for n in group:
            base_str, suffix_str = y_to_jianpu(n['y_center'], n['system'], n.get('clef', 'treble'))
            pitch = base_str + suffix_str
            ind_dur = detect_duration_per_note(n, binary, dy, music_symbols=music_symbols,
                                                all_notes=notes, other_noteheads=other_noteheads)
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
                'stem_dir': n['stem']['stem_dir'],
                'beam_count': n.get('beam_count', -1),
                'has_flag': n.get('has_flag', False),
            })

        # Detect duration
        duration = _detect_duration(music_symbols, binary, group, dy)

        # Determine stem_dir from the note with longest stem
        stemmed = [n for n in group if n['stem']['stem_dir'] is not None]
        if stemmed:
            best_stem_note = max(stemmed, key=lambda n: n['stem']['stem_length'])
            stem_dir = best_stem_note['stem']['stem_dir']
            stem_x = best_stem_note['stem']['stem_x']
        else:
            stem_dir = None
            stem_x = int(np.mean([n['stem']['stem_x'] for n in group]))

        avg_x = float(np.mean([n['x'] for n in group]))

        note_units.append({
            'notes': note_entries,
            'duration': duration,
            'stem_dir': stem_dir,
            'stem_x': stem_x,
            'x': avg_x,
        })

    return note_units


def merge_overlapping_note_units(note_units, beats_per_measure=2.0, dy=21.0):
    """Merge notes whose durations overlap into shared events (two-voice alignment).

    Detects two-voice chords where one voice (stem_dir=up) plays a longer note
    (e.g., eighth) while the other voice (stem_dir=down) plays shorter notes
    (e.g., sixteenths). The longer note is propagated to subsequent events.

    The key signal is stem_dir: in a two-voice chord, notes have opposing
    stem directions. A note is only propagated if:
    1. Its event contains notes with BOTH stem_dir=up and stem_dir=down
    2. It has a DIFFERENT stem_dir from the majority of notes in its event
    3. Its individual_duration > the majority's duration
    4. The target event has no note with the same stem_dir

    Parameters
    ----------
    note_units : list of NoteUnit dicts, sorted by x
    beats_per_measure : float
    dy : float, staff spacing

    Returns
    -------
    list of NoteUnit dicts with merged notes
    """
    if len(note_units) <= 1:
        return note_units

    # Sort by x
    units = sorted(note_units, key=lambda u: u['x'])

    # Assign start_time to each unit based on cumulative shortest durations
    start_times = []
    t = 0.0
    for u in units:
        start_times.append(t)
        event_dur = min((n.get('individual_duration', 0.25) for n in u['notes']), default=0.25)
        t += event_dur

    for i in range(len(units)):
        notes = units[i]['notes']
        if len(notes) < 2:
            continue

        # Check for two-voice: must have both stem_dir=up and stem_dir=down
        dirs = [n.get('stem_dir') for n in notes]
        has_up = 'up' in dirs
        has_down = 'down' in dirs
        if not (has_up and has_down):
            continue

        # Find the minority stem_dir (the sustained voice)
        n_up = sum(1 for d in dirs if d == 'up')
        n_down = sum(1 for d in dirs if d == 'down')
        minority_dir = 'up' if n_up <= n_down else 'down'

        # Propagate minority-dir notes to subsequent events
        for note in list(notes):
            if note.get('stem_dir') != minority_dir:
                continue
            ind_dur = note.get('individual_duration', 0.25)
            # Must have longer duration than the majority
            majority_durs = [n.get('individual_duration', 0.25)
                             for n in notes if n.get('stem_dir') != minority_dir]
            if not majority_durs:
                continue
            majority_min = min(majority_durs)
            if ind_dur <= majority_min:
                continue
            # Cap propagation: beam detection is unreliable for edge-of-group
            # notes (especially ledger-line notes). Limit to 2× majority duration.
            ind_dur = min(ind_dur, majority_min * 2)

            note_end_time = start_times[i] + ind_dur

            for j in range(i + 1, len(units)):
                if start_times[j] >= note_end_time:
                    break

                # Check if target event OR any nearby NoteUnit already has
                # a note with the minority stem_dir (within dy*1.5 x tolerance).
                # Also treat sd=None as a potential match — stem tracking failed,
                # so we can't be sure the note isn't in the minority voice.
                target_x = units[j]['x']
                has_minority_nearby = False
                for k in range(len(units)):
                    if k == i:
                        continue
                    if abs(units[k]['x'] - target_x) < dy * 1.5:
                        k_dirs = {n.get('stem_dir') for n in units[k]['notes']}
                        if minority_dir in k_dirs or (None in k_dirs and len(units[k]['notes']) >= 2):
                            has_minority_nearby = True
                            break

                if has_minority_nearby:
                    break  # voice-note already present, stop propagating

                existing_pitches = {n['pitch'] for n in units[j]['notes']}
                if note['pitch'] not in existing_pitches:
                    sustained = dict(note)
                    sustained['individual_duration'] = min(
                        (n.get('individual_duration', 0.25) for n in units[j]['notes']),
                        default=0.25
                    )
                    units[j]['notes'].append(sustained)

    return units


def segment_into_measures(note_units, rests, barline_xs, dy,
                          beats_per_measure=2.0, is_first_system=True,
                          tuplet_markers=None):
    """
    Segment NoteUnits and rests into measures by barline x-positions.

    Parameters
    ----------
    note_units : list of NoteUnit dicts (from build_note_units)
    rests : list of dict with 'x' and 'duration' keys
    barline_xs : list of float/int, sorted x-positions of barlines
    dy : float, staff line spacing
    beats_per_measure : float, beats per measure (default 2.0 for 2/4 time)
    is_first_system : bool, if True allow pickup measure detection
    tuplet_markers : list of dicts with 'x', 'n' (tuplet number), optional

    Returns
    -------
    list of lists, each inner list contains events sorted by x:
        {'type': 'note_unit', 'x': float, 'unit': NoteUnit}
        {'type': 'rest', 'x': float, 'duration': float}
    """
    if tuplet_markers is None:
        tuplet_markers = []
    # Build events
    events = []
    for unit in note_units:
        events.append({'type': 'note_unit', 'x': unit['x'], 'unit': unit})
    for rest in rests:
        events.append({'type': 'rest', 'x': rest['x'], 'duration': rest.get('duration', 1.0)})

    # Sort by x
    events.sort(key=lambda e: e['x'])

    # Clean rests: remove rests within dy*1.5 of any note_unit x
    note_unit_xs = [e['x'] for e in events if e['type'] == 'note_unit']
    cleaned = []
    for e in events:
        if e['type'] == 'rest':
            too_close = any(abs(e['x'] - nx) < dy * 1.5 for nx in note_unit_xs)
            if too_close:
                continue
        cleaned.append(e)

    # Deduplicate rests within dy*2.0 of each other
    final_events = []
    for e in cleaned:
        if e['type'] == 'rest':
            duplicate = any(
                f['type'] == 'rest' and abs(f['x'] - e['x']) < dy * 2.0
                for f in final_events
            )
            if duplicate:
                continue
        final_events.append(e)

    sorted_barlines = sorted(barline_xs)

    # Segment by barline boundaries
    boundaries = [0] + sorted_barlines + [float('inf')]

    measures = []
    for i in range(len(boundaries) - 1):
        left = boundaries[i]
        right = boundaries[i + 1]
        measure = [e for e in final_events if left <= e['x'] < right]
        measure.sort(key=lambda e: e['x'])
        measures.append(measure)

    # The last barline is the end-of-system barline; any content after it
    # belongs to the final measure, not a new measure.
    # When merging, deduplicate note units near the barline boundary that
    # have matching y_centers (double-detections spanning the barline).
    if len(measures) >= 2 and measures[-1]:
        existing = measures[-2]
        trailing = measures[-1]
        for te in trailing:
            if te['type'] != 'note_unit':
                continue  # skip rests from trailing area (barline gap artifacts)
            t_ycs = sorted(n['y_center'] for n in te['unit']['notes'])
            is_dup = False
            for ee in existing:
                if ee['type'] != 'note_unit':
                    continue
                if abs(te['x'] - ee['x']) > dy * 5:
                    continue
                e_ycs = sorted(n['y_center'] for n in ee['unit']['notes'])
                if len(t_ycs) == len(e_ycs) and all(
                    abs(a - b) <= dy * 0.25 for a, b in zip(t_ycs, e_ycs)
                ):
                    is_dup = True
                    break
            if not is_dup:
                existing.append(te)
        existing.sort(key=lambda e: e['x'])
    if measures:
        measures.pop()

    # Strip trailing empty measures
    while measures and not measures[-1]:
        measures.pop()

    # Estimate durations using proportional spacing
    for mi, measure in enumerate(measures):
        _estimate_durations_in_measure(measure, beats_per_measure=beats_per_measure,
                                       measure_idx=mi, barline_xs=sorted_barlines,
                                       is_first_system=is_first_system)

    # Apply tuplet markers: each marker indicates N notes near its x-position
    # should have duration 1/6 (tuplet subdivision).
    if tuplet_markers:
        _apply_tuplet_markers(measures, tuplet_markers, beats_per_measure, dy=dy)

    # Post-processing: if a measure has a whole note (duration >= 3.5),
    # remove all rests — the whole note fills the measure.
    for measure in measures:
        has_whole = any(
            e['type'] == 'note_unit' and e['unit']['duration'] >= 3.5
            for e in measure
        )
        if has_whole:
            measure[:] = [e for e in measure if e['type'] != 'rest']

    # Post-processing: adjust rest durations so each measure totals
    # beats_per_measure. Strategy:
    # 1. If detected rest durations sum close to remaining, try upgrading
    #    individual rests to fill the gap exactly.
    # 2. If that fails, try distributing remaining beats equally.
    # 3. Keep whichever approach is closest to beats_per_measure.
    for measure in measures:
        note_events = [e for e in measure if e['type'] == 'note_unit']
        rest_events = [e for e in measure if e['type'] == 'rest']
        if not rest_events:
            continue
        note_beats = sum(e['unit']['duration'] for e in note_events)
        remaining = beats_per_measure - note_beats
        if remaining <= 0:
            continue

        n_rests = len(rest_events)
        detected_durs = [e['duration'] for e in rest_events]
        detected_total = sum(detected_durs)

        # Strategy 1: upgrade individual rests to fill the gap.
        # When multiple rests share the same detected duration, prefer
        # upgrading the one with more horizontal space (proportional
        # spacing heuristic: longer rests occupy more x-distance).
        gap = remaining - detected_total
        best_durs = list(detected_durs)
        if 0 < gap <= 2.0:
            # Compute x-span for each rest: distance to next event
            rest_spans = []
            for ri, re_ in enumerate(rest_events):
                rx = re_['x']
                # Find next event after this rest in the full measure
                next_x = None
                for e in measure:
                    if e['x'] > rx + 1:
                        next_x = e['x']
                        break
                span = (next_x - rx) if next_x else 0
                rest_spans.append(span)
            # Sort by (duration, -span): smallest duration first,
            # largest span first among equal durations
            indices = sorted(range(n_rests),
                             key=lambda i: (detected_durs[i], -rest_spans[i]))
            trial = list(detected_durs)
            gap_left = gap
            for idx in indices:
                if gap_left <= 0.01:
                    break
                upgrade = min(gap_left, trial[idx])  # at most double
                trial[idx] = _snap_duration(trial[idx] + upgrade)
                gap_left = remaining - sum(trial)
            if abs(sum(trial) - remaining) < abs(detected_total - remaining):
                best_durs = trial

        # Strategy 2: equal distribution
        per_rest = _snap_duration(remaining / n_rests)
        equal_total = per_rest * n_rests

        # Pick whichever is closest to remaining
        best_total = sum(best_durs)
        if abs(equal_total - remaining) < abs(best_total - remaining):
            for e in rest_events:
                e['duration'] = per_rest
        elif abs(best_total - remaining) < abs(detected_total - remaining):
            for i, e in enumerate(rest_events):
                e['duration'] = best_durs[i]

    return measures


def _apply_tuplet_markers(measures, tuplet_markers, beats_per_measure, dy=21.0):
    """Apply detected tuplet markers to override note durations.

    The actual tuplet duration = base_duration × 2/3, where base_duration
    comes from beam detection (individual_duration):
      - 2 beams (base=/4=0.25) → 0.25 × 2/3 = 1/6
      - 1 beam  (base=/2=0.5)  → 0.5  × 2/3 = 1/3
      - 0 beams (base=1.0)     → 1.0  × 2/3 = 2/3

    This works for both triplets (3 in time of 2) and sextuplets
    (6 in time of 4) since both use the 2:3 ratio.
    """
    search_margin = dy * 2.5

    for marker in tuplet_markers:
        mx = marker['x']
        n = marker['n']

        # Find which measure this marker belongs to (by x position)
        best_mi = None
        for mi, measure in enumerate(measures):
            note_events = [e for e in measure if e['type'] == 'note_unit']
            if not note_events:
                continue
            xs = [e['x'] for e in note_events]
            if min(xs) - search_margin <= mx <= max(xs) + search_margin:
                best_mi = mi
                break

        if best_mi is None:
            continue

        measure = measures[best_mi]
        note_events = [e for e in measure if e['type'] == 'note_unit']

        # Find N contiguous events nearest to the marker x.
        # First find the anchor (closest event), then expand to N contiguous.
        candidates = [e for e in note_events if len(e['unit']['notes']) == 1]
        if len(candidates) < n:
            candidates = list(note_events)
        candidates.sort(key=lambda e: e['x'])

        if len(candidates) <= n:
            tuplet_group = candidates
        else:
            # Find anchor index (closest to marker x)
            anchor = min(range(len(candidates)),
                         key=lambda i: abs(candidates[i]['x'] - mx))
            # Expand window around anchor to pick N contiguous events
            best_start = max(0, anchor - n + 1)
            best_end = min(len(candidates), best_start + n)
            if best_end - best_start < n:
                best_start = best_end - n
            # Among all valid windows containing the anchor, pick the one
            # whose center is closest to marker x
            best_dist = float('inf')
            for s in range(max(0, anchor - n + 1), min(anchor + 1, len(candidates) - n + 1)):
                window = candidates[s:s + n]
                center = sum(e['x'] for e in window) / n
                d = abs(center - mx)
                if d < best_dist:
                    best_dist = d
                    best_start = s
            tuplet_group = candidates[best_start:best_start + n]

        # Determine base duration from beam detection.
        # Use the mode (most common) individual_duration across the group.
        idurs = []
        for e in tuplet_group:
            for note in e['unit']['notes']:
                idurs.append(note.get('individual_duration', 0.25))
        if idurs:
            from collections import Counter
            base_dur = Counter(idurs).most_common(1)[0][0]
        else:
            base_dur = 0.25

        # Tuplet actual duration = base × 2/3
        tuplet_dur = base_dur * 2.0 / 3.0

        # Validate: tuplet + remaining notes must fit in the measure.
        # Each remaining note needs at least 0.25 beats (sixteenth).
        # If beam detection over-estimated the base, halve and retry.
        tuplet_ids = {id(e) for e in tuplet_group}
        remaining = [e for e in note_events if id(e) not in tuplet_ids]
        rest_beats = sum(e.get('duration', 1.0) for e in measure if e['type'] == 'rest')
        min_remaining = len(remaining) * 0.25

        while (tuplet_dur * n + min_remaining + rest_beats
               > beats_per_measure + 0.1 and base_dur > 0.125):
            base_dur *= 0.5
            tuplet_dur = base_dur * 2.0 / 3.0

        for e in tuplet_group:
            e['unit']['duration'] = tuplet_dur

        # Re-estimate durations for remaining (non-tuplet) events.
        # Equal division: spacing-based proportions are unreliable here since
        # tuplet notes' spacing was mixed in during the initial estimate.
        tuplet_beats_total = tuplet_dur * n
        leftover = max(0.25, beats_per_measure - tuplet_beats_total - rest_beats)

        if remaining:
            per_note = leftover / len(remaining)
            snapped = _snap_duration(per_note)
            for e in remaining:
                e['unit']['duration'] = snapped


# ============================================================
# Duration estimation by proportional spacing
# ============================================================
STANDARD_DURATIONS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]


def _snap_duration(dur):
    """Snap a duration value to the nearest standard duration."""
    return min(STANDARD_DURATIONS, key=lambda d: abs(d - dur))


def _estimate_durations_in_measure(measure, beats_per_measure=2.0, measure_idx=0,
                                   barline_xs=None, is_first_system=True):
    """Estimate note durations within a measure.

    Strategy (beam-first):
    1. Use individual_duration from beam detection as primary source.
    2. If total matches beats_per_measure, done.
    3. If total is off, use proportional spacing as fallback.
    """
    note_events = [e for e in measure if e['type'] == 'note_unit']
    rest_events = [e for e in measure if e['type'] == 'rest']

    n = len(note_events)
    if n == 0:
        return

    # Detect pickup measure (only on the first system's first measure)
    if is_first_system and measure_idx == 0 and n >= 2 and barline_xs and len(barline_xs) > 0:
        first_barline = barline_xs[0]
        event_xs = [e['x'] for e in note_events]
        if first_barline > 0 and all(x > first_barline * 0.4 for x in event_xs):
            for e in note_events:
                e['unit']['duration'] = 0.5
            measure.insert(0, {'type': 'rest', 'x': 0, 'duration': 1.0})
            return

    rest_beats = sum(e.get('duration', 1.0) for e in rest_events)
    remaining = max(0.25, beats_per_measure - rest_beats)

    if n == 1:
        # Trust individual_duration from beam detection. For hollow notes
        # (whole/half), idur is already set correctly (4.0 or 2.0).
        # For beamed notes, idur reflects the actual beam count.
        # Only fall back to remaining-beats fill when no beam info exists.
        idur = note_events[0]['unit']['notes'][0].get('individual_duration', 1.0)
        if idur >= 2.0 or (idur < 1.0 and rest_beats > 0):
            # Trust beam detection: whole/half notes, or short notes with rests
            note_events[0]['unit']['duration'] = idur
        else:
            dur = _snap_duration(remaining)
            note_events[0]['unit']['duration'] = dur
        return

    # --- Beam-first approach (for longer measures where spacing is unreliable) ---
    # Collect individual_duration from beam detection for each event.
    beam_durs = []
    for e in note_events:
        idurs = [ne.get('individual_duration', 1.0) for ne in e['unit']['notes']]
        if idurs:
            from collections import Counter
            beam_durs.append(Counter(idurs).most_common(1)[0][0])
        else:
            beam_durs.append(1.0)

    # Snap non-standard durations (e.g. 0.375 from detection artifacts)
    beam_durs = [_snap_duration(d) for d in beam_durs]
    beam_note_total = sum(beam_durs)
    beam_total = beam_note_total + rest_beats

    # For longer measures (4/4+), beam detection is more reliable than
    # proportional spacing. Use beam durations directly — they give correct
    # note-level durations even when the measure total doesn't match
    # (due to missing notes or wrong barlines).
    # For 2/4 time, proportional spacing is well-tuned, so keep it.
    if beats_per_measure > 2.5:
        # If beam total < beats_per_measure, some notes may need longer
        # durations. Use proportional spacing to identify which notes
        # have wider gaps (= longer duration).
        if beam_note_total + rest_beats < beats_per_measure - 0.1 and n >= 3:
            deficit = beats_per_measure - rest_beats - beam_note_total
            xs = [e['x'] for e in note_events]
            gaps = [xs[i+1] - xs[i] for i in range(n-1)]
            # Add gap for last note (to right barline)
            right_bl = float('inf')
            if barline_xs:
                for bx in sorted(barline_xs):
                    if bx > xs[-1]:
                        right_bl = bx
                        break
            last_gap = right_bl - xs[-1] if right_bl < float('inf') else (
                gaps[-1] if gaps else 0)
            gaps.append(last_gap)
            if gaps:
                median_gap = sorted(gaps)[len(gaps)//2]
                # Upgrade notes with gaps > 1.25× median to the next duration
                for i in range(n):
                    if gaps[i] > median_gap * 1.25 and deficit >= 0.4:
                        old = beam_durs[i]
                        beam_durs[i] = min(old * 2, 4.0)
                        deficit -= (beam_durs[i] - old)
                # If deficit remains, upgrade shortest-duration notes
                # with widest gaps (most likely under-detected)
                if deficit >= 0.2:
                    candidates = [(i, gaps[i]) for i in range(n)
                                  if beam_durs[i] <= 0.25]
                    candidates.sort(key=lambda t: -t[1])  # widest gap first
                    for idx, _ in candidates:
                        if deficit < 0.2:
                            break
                        old = beam_durs[idx]
                        beam_durs[idx] = min(old * 2, 1.0)
                        deficit -= (beam_durs[idx] - old)

        # If beam total > beats_per_measure, some notes may have been
        # mis-detected as quarter when they should be eighth (beam not
        # found).  Downgrade quarter notes with the narrowest gaps first.
        beam_note_total2 = sum(beam_durs)
        surplus = beam_note_total2 + rest_beats - beats_per_measure
        if surplus >= 0.2 and n >= 3:
            xs = [e['x'] for e in note_events]
            # Compute gap for each note to its right neighbor
            gaps_r = []
            for i in range(n):
                if i < n - 1:
                    gaps_r.append(xs[i + 1] - xs[i])
                else:
                    # Last note: gap to right barline
                    right_bl = float('inf')
                    if barline_xs:
                        for bx in sorted(barline_xs):
                            if bx > xs[-1]:
                                right_bl = bx
                                break
                    gaps_r.append(right_bl - xs[-1] if right_bl < float('inf') else 0)
            # Sort quarter notes by gap (smallest gap = most likely eighth)
            candidates = [(i, gaps_r[i]) for i in range(n)
                          if beam_durs[i] == 1.0]
            candidates.sort(key=lambda t: t[1])
            for idx, _ in candidates:
                if surplus < 0.4:
                    break
                beam_durs[idx] = 0.5
                surplus -= 0.5

        for i, e in enumerate(note_events):
            e['unit']['duration'] = beam_durs[i]
        return

    # --- Fallback: proportional spacing ---
    xs = [e['x'] for e in note_events]
    gaps = [xs[i + 1] - xs[i] for i in range(n - 1)]

    if not gaps or sum(gaps) == 0:
        dur = _snap_duration(remaining / n)
        for e in note_events:
            e['unit']['duration'] = dur
        return

    last_x = xs[-1]
    last_gap = gaps[-1]
    if barline_xs and len(gaps) >= 2:
        next_barlines = [bx for bx in barline_xs if bx > last_x]
        if next_barlines:
            bl_gap = next_barlines[0] - last_x
            median_gap = sorted(gaps)[len(gaps) // 2]
            if median_gap > 0 and 0.3 * median_gap <= bl_gap <= 3.0 * median_gap:
                last_gap = bl_gap
    gaps.append(last_gap)

    total_gap = sum(gaps)
    raw = [g / total_gap * remaining for g in gaps]
    snapped = [_snap_duration(d) for d in raw]

    snapped_total = sum(snapped)
    if abs(snapped_total - remaining) > 0.1:
        diff = remaining - snapped_total
        max_idx = max(range(n), key=lambda i: gaps[i])
        snapped[max_idx] = _snap_duration(snapped[max_idx] + diff)

    for i, e in enumerate(note_events):
        e['unit']['duration'] = snapped[i]
