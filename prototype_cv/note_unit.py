"""
note_unit.py
Build NoteUnit groups (chords sharing a stem) with duration detection,
then segment into measures by barline positions.
"""
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


def _count_beams(binary, tip_y, stem_x, dy, staff_lines=None, stem_dir=None):
    """
    Count horizontal beams near the stem tip using horizontal projection analysis.
    Uses the full binary image (beams intact) and filters out staff line positions.
    The ROI is directional: looks PAST the tip (away from noteheads).
    Returns (beam_count, has_flag).
    """
    h_img, w_img = binary.shape[:2]
    # Look past the tip in the stem direction (away from noteheads)
    if stem_dir == 'down':
        roi_y1 = max(0, int(tip_y - dy * 0.2))
        roi_y2 = min(h_img, int(tip_y + dy * 1.0))
    elif stem_dir == 'up':
        roi_y1 = max(0, int(tip_y - dy * 1.0))
        roi_y2 = min(h_img, int(tip_y + dy * 0.2))
    else:
        roi_y1 = max(0, int(tip_y - dy * 0.5))
        roi_y2 = min(h_img, int(tip_y + dy * 0.5))
    roi_x1 = max(0, int(stem_x - dy * 1.5))
    roi_x2 = min(w_img, int(stem_x + dy * 1.5))

    if roi_y2 <= roi_y1 or roi_x2 <= roi_x1:
        return 0, False

    roi = binary[roi_y1:roi_y2, roi_x1:roi_x2]
    roi_h, roi_w = roi.shape[:2]
    if roi_h == 0 or roi_w == 0:
        return 0, False

    # Horizontal projection: fraction of white pixels per row
    projection = np.count_nonzero(roi > 127, axis=1) / roi_w

    # Mask out staff line rows (they look like beams but aren't)
    if staff_lines is not None:
        for line_y in staff_lines:
            local_y = int(line_y) - roi_y1
            for offset in range(-1, 2):  # mask ±1 row around each line
                row = local_y + offset
                if 0 <= row < roi_h:
                    projection[row] = 0.0

    # Find bands with density > 0.4
    min_thickness = max(1, int(dy * 0.08))
    max_thickness = max(min_thickness + 1, int(dy * 0.45))

    beam_count = 0
    i = 0
    while i < len(projection):
        if projection[i] > 0.4:
            start = i
            while i < len(projection) and projection[i] > 0.4:
                i += 1
            thickness = i - start
            if min_thickness <= thickness <= max_thickness:
                beam_count += 1
        else:
            i += 1

    # Check for flag if no beams found
    has_flag = False
    if beam_count == 0:
        flag_x1 = max(0, int(stem_x))
        flag_x2 = min(w_img, int(stem_x + dy * 1.0))
        flag_y1 = max(0, int(tip_y - dy * 0.4))
        flag_y2 = min(h_img, int(tip_y + dy * 0.4))
        if flag_y2 > flag_y1 and flag_x2 > flag_x1:
            flag_roi = binary[flag_y1:flag_y2, flag_x1:flag_x2]
            if flag_roi.size > 0:
                density = np.count_nonzero(flag_roi > 127) / flag_roi.size
                has_flag = density > 0.15

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
    is_hollow = False  # filled noteheads assumed; hollow detected elsewhere
    return _detect_individual_duration(beam_count, has_flag, is_hollow)


def build_note_units(notes, music_symbols, binary, dy):
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
    x_threshold = dy * 0.9  # fallback: notehead x-proximity

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

            # Also check y-distance: chord notes should be within ~staff height
            y_range = max(n['y_center'] for n in group) - min(n['y_center'] for n in group)
            new_y_range = max(y_range, abs(note_b['y_center'] - group[0]['y_center']))
            y_ok = new_y_range <= dy * 4.0  # within ~staff height

            if (stem_close or x_close) and y_ok:
                group.append(note_b)
                used[idx_b] = True
            elif note_b['x'] - note_a['x'] >= x_threshold:
                break  # sorted by x, no more matches

        groups.append(group)

    # Filter false positives: remove notes with very low density on music_symbols
    # (these are accidentals/other symbols detected as noteheads by morphology)
    verified_groups = []
    for group in groups:
        good_notes = []
        for n in group:
            cx = n['x'] + n['w'] // 2
            cy = n['y_center']
            pad = max(2, int(dy * 0.15))
            ny1 = max(0, cy - pad)
            ny2 = min(music_symbols.shape[0], cy + pad)
            nx1 = max(0, cx - pad)
            nx2 = min(music_symbols.shape[1], cx + pad)
            region = music_symbols[ny1:ny2, nx1:nx2]
            if region.size > 0:
                fill = np.mean(region) / 255.0
                if fill >= 0.25:  # real noteheads have high fill on music_symbols
                    good_notes.append(n)
            else:
                good_notes.append(n)
        if good_notes:
            verified_groups.append(good_notes)
    groups = verified_groups

    # Build NoteUnit for each group
    note_units = []
    for group in groups:
        # Sort by y_center descending (low pitch first = higher y value first)
        group.sort(key=lambda n: -n['y_center'])

        # Build pitch for each note
        note_entries = []
        for n in group:
            base_str, suffix_str = y_to_jianpu(n['y_center'], n['system'], n.get('clef', 'treble'))
            pitch = base_str + suffix_str
            note_entries.append({
                'pitch': pitch,
                'accidental': n.get('accidental', None),
                'x': n['x'],
                'y_center': n['y_center'],
                'clef': n.get('clef', 'treble'),
                'system': n['system'],
                'pair_idx': n.get('pair_idx', 0),
                'w': n['w'],
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


def segment_into_measures(note_units, rests, barline_xs, dy,
                          beats_per_measure=2.0, is_first_system=True):
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

    Returns
    -------
    list of lists, each inner list contains events sorted by x:
        {'type': 'note_unit', 'x': float, 'unit': NoteUnit}
        {'type': 'rest', 'x': float, 'duration': float}
    """
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

    # Segment by barline boundaries
    sorted_barlines = sorted(barline_xs)
    boundaries = [0] + sorted_barlines + [float('inf')]

    measures = []
    for i in range(len(boundaries) - 1):
        left = boundaries[i]
        right = boundaries[i + 1]
        measure = [e for e in final_events if left <= e['x'] < right]
        measure.sort(key=lambda e: e['x'])
        measures.append(measure)

    # Estimate durations using proportional spacing
    for mi, measure in enumerate(measures):
        _estimate_durations_in_measure(measure, beats_per_measure=beats_per_measure,
                                       measure_idx=mi, barline_xs=sorted_barlines,
                                       is_first_system=is_first_system)

    # Strip trailing empty measures (region after last barline)
    while measures and not measures[-1]:
        measures.pop()

    return measures


# ============================================================
# Duration estimation by proportional spacing
# ============================================================
STANDARD_DURATIONS = [0.25, 0.5, 1.0, 2.0, 4.0]


def _snap_duration(dur):
    """Snap a duration value to the nearest standard duration."""
    return min(STANDARD_DURATIONS, key=lambda d: abs(d - dur))


def _estimate_durations_in_measure(measure, beats_per_measure=2.0, measure_idx=0,
                                   barline_xs=None, is_first_system=True):
    """Estimate note durations using proportional x-spacing within a measure."""
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
            # Pickup measure: set all to eighth notes and add leading rest
            for e in note_events:
                e['unit']['duration'] = 0.5
            measure.insert(0, {'type': 'rest', 'x': 0, 'duration': 1.0})
            return

    rest_beats = sum(e.get('duration', 1.0) for e in rest_events)
    remaining = max(0.25, beats_per_measure - rest_beats)

    if n == 1:
        dur = _snap_duration(remaining)
        note_events[0]['unit']['duration'] = dur
        return

    # Proportional spacing
    xs = [e['x'] for e in note_events]
    gaps = [xs[i + 1] - xs[i] for i in range(n - 1)]

    if not gaps or sum(gaps) == 0:
        dur = _snap_duration(remaining / n)
        for e in note_events:
            e['unit']['duration'] = dur
        return

    # Last event gets same gap as previous (heuristic)
    gaps.append(gaps[-1])
    total_gap = sum(gaps)
    raw = [g / total_gap * remaining for g in gaps]
    snapped = [_snap_duration(d) for d in raw]

    # Adjust if total doesn't match
    snapped_total = sum(snapped)
    if abs(snapped_total - remaining) > 0.1:
        diff = remaining - snapped_total
        max_idx = max(range(n), key=lambda i: gaps[i])
        snapped[max_idx] = _snap_duration(snapped[max_idx] + diff)

    for i, e in enumerate(note_events):
        e['unit']['duration'] = snapped[i]
