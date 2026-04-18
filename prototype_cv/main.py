"""
main.py - Music score to Jianpu (简谱) converter.

Pipeline:
1. Extract staff lines and binary image
2. Detect staff systems and pair into grand staves (treble + bass)
3. Detect barlines to segment measures
4. Detect noteheads (filled and hollow)
5. Assign notes to treble/bass staves
6. Detect accidentals (sharps/flats/naturals)
7. Detect rests
8. Track stems from noteheads
9. Build note units (chord grouping + duration detection)
10. Segment into measures
11. Format output in Jianpu notation
"""
import cv2
import numpy as np
import os
import sys

# Combining low-line marks used by jianpu_formatter (U+0332 / U+0333) and
# the middle dot (U+00B7) need a Unicode-capable stdout. On Windows the
# default GBK console will raise UnicodeEncodeError when printing them.
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except (AttributeError, Exception):
    pass

from staff_removal import extract_staff_lines
from pitch_detection import get_staff_systems, pair_grand_staves, detect_staff_layout
from template_matching import find_noteheads
from symbol_detection import (
    detect_barlines,
    detect_accidentals_global,
    assign_accidentals_to_notes,
    detect_rests,
    detect_tuplet_markers,
    detect_time_signature,
    detect_time_signatures_along_system,
    detect_multi_rest_count,
)


def _timesig_to_bpm(num, den):
    """Convert a detected (num, den) time signature to beats-per-measure
    expressed in quarter-note units (1.0 = 1 quarter note).
    """
    return num * (4.0 / den)


def _build_system_timesig_anchors(binary, system, barlines, dy, current_bpm,
                                   clef_roi_x1=None, clef_roi_x2=None):
    """Build time-signature anchors for one staff system.

    Returns (anchors, new_current_bpm) where anchors is a list of
    (x_threshold, bpm) pairs suitable for passing to segment_into_measures
    via timesig_anchors, and new_current_bpm is the bpm in effect at the
    end of this system (to be inherited by the next system).
    """
    detections = detect_time_signatures_along_system(
        binary, system, barlines, dy,
        clef_roi_x1=clef_roi_x1, clef_roi_x2=clef_roi_x2)

    if detections:
        print(f"   [timesig] raw detections: " + ", ".join(
            f"{d['source']}@x={d['x']:.0f} {d['num']}/{d['den']} "
            f"score={d['score']:.2f}" for d in detections))

    anchors = []
    bpm = current_bpm
    anchors.append((0.0, bpm))

    for d in detections:
        new_bpm = _timesig_to_bpm(d['num'], d['den'])
        if d['source'] == 'clef':
            anchors[0] = (0.0, new_bpm)
            bpm = new_bpm
        else:
            anchors.append((d['x'], new_bpm))
            bpm = new_bpm

    return anchors, bpm
from note_assignment import assign_notes_to_staves, filter_false_positive_notes
from stem_tracking import track_stem
from note_unit import (build_note_units, segment_into_measures,
                        merge_overlapping_note_units, _fill_rests_for_gap)
from jianpu_formatter import format_measure, format_output


def main(image_path):
    print(f"Processing image: {image_path}")

    # ── 1. Staff Line Extraction ──
    print("1. Extracting staff lines...")
    staff_lines, music_symbols, binary = extract_staff_lines(image_path)
    if staff_lines is None:
        print("Error: Could not extract staff lines!")
        return

    # ── 2. Detect Staff Systems & Pair Grand Staves ──
    print("2. Detecting staff systems...")
    systems = get_staff_systems(staff_lines)
    if not systems:
        print("Error: No staff systems found!")
        return

    print(f"   Found {len(systems)} individual staves")

    # Calculate average dy (staff line spacing)
    dy = np.mean([(s[4] - s[0]) / 4.0 for s in systems])
    print(f"   Average staff line spacing (dy): {dy:.1f}px")

    # Detect layout: grand staff (piano) vs single staff (solo instrument)
    layout = detect_staff_layout(systems)
    print(f"   Layout: {layout}")

    if layout == 'single':
        return _main_single_staff(image_path, systems, staff_lines, music_symbols,
                                   binary, dy)

    grand_staff_pairs = pair_grand_staves(systems)
    print(f"   Paired into {len(grand_staff_pairs)} grand staff systems (treble+bass)")

    if not grand_staff_pairs:
        print("Error: Could not pair staves into grand staff systems!")
        return

    # ── 2b. Auto-detect time signature ──
    from config import CFG
    time_sig = detect_time_signature(binary, systems[0], dy)
    if time_sig:
        num, den = time_sig
        bpm_detected = _timesig_to_bpm(num, den)
        CFG.duration.beats_per_measure = bpm_detected
        print(f"   Time signature: {num}/{den} → beats_per_measure={bpm_detected}")
    else:
        print(f"   Time signature: not detected, using default {CFG.duration.beats_per_measure}")

    # ── 3. Detect Barlines ──
    print("3. Detecting barlines...")
    barlines_per_system = detect_barlines(binary, systems, dy)

    barlines_per_pair = []
    for pair_idx, (treble_sys, bass_sys) in enumerate(grand_staff_pairs):
        treble_idx = pair_idx * 2
        bass_idx = pair_idx * 2 + 1
        treble_bl = barlines_per_system[treble_idx] if treble_idx < len(barlines_per_system) else []
        bass_bl = barlines_per_system[bass_idx] if bass_idx < len(barlines_per_system) else []

        merged = _merge_barlines(treble_bl, bass_bl, dy, binary=binary,
                                  treble_sys=treble_sys, bass_sys=bass_sys)

        barlines_per_pair.append(merged)
        print(f"   Grand staff {pair_idx}: {len(merged)} barlines at x={merged}")

    # ── 4. Detect Noteheads ──
    print("4. Detecting noteheads...")
    boxes, template, exclusion_zones = find_noteheads(
        binary, dy, threshold=0.55, staff_systems=systems, music_symbols=music_symbols)

    all_notes = [
        {'x': x, 'y': y, 'w': w, 'h': h, 'y_center': y + h // 2, 'score': score}
        for x, y, w, h, score in boxes
    ]
    print(f"   Detected {len(all_notes)} noteheads total")

    # Filter clef area — adaptive per-system boundary
    # Line 1 (pair 0) has clef + key-sig + time-sig → wider exclusion (17%)
    # Lines 2+ → detect actual first notehead and set boundary just before it
    img_w = binary.shape[1]
    clef_area_x = int(img_w * 0.17)

    from template_matching import create_notehead_template
    nh_template = create_notehead_template(dy)
    th, tw = nh_template.shape

    clef_boundaries = {}
    for pi, (treble_sys, bass_sys) in enumerate(grand_staff_pairs):
        if pi == 0:
            clef_boundaries[pi * 2] = clef_area_x
            clef_boundaries[pi * 2 + 1] = clef_area_x
            continue
        # Scan x=200-500 for first notehead on each staff
        boundary = clef_area_x  # fallback
        for sys_info in [treble_sys, bass_sys]:
            y1 = max(0, sys_info[0] - int(dy * 2))
            y2 = min(binary.shape[0], sys_info[4] + int(dy * 2))
            roi = binary[y1:y2, 200:500]
            if tw < roi.shape[1] and th < roi.shape[0]:
                res = cv2.matchTemplate(roi, nh_template, cv2.TM_CCOEFF_NORMED)
                loc = np.where(res >= 0.55)
                if len(loc[0]) > 0:
                    first_x = int(np.min(loc[1])) + 200
                    candidate = max(200, first_x - int(dy))
                    boundary = min(boundary, candidate)
        clef_boundaries[pi * 2] = boundary
        clef_boundaries[pi * 2 + 1] = boundary

    # Filter notes based on their system's clef boundary
    all_notes = [n for n in all_notes
                 if n['x'] > clef_boundaries.get(_nearest_staff(n['y_center'], systems),
                                                  clef_area_x)]
    print(f"   After clef area filtering: {len(all_notes)} notes")

    # ── 5. Assign Notes to Treble/Bass ──
    print("5. Assigning notes to treble/bass staves...")
    treble_notes, bass_notes = assign_notes_to_staves(all_notes, grand_staff_pairs, dy)
    treble_notes = filter_false_positive_notes(treble_notes, dy, clef='treble')
    bass_notes = filter_false_positive_notes(bass_notes, dy, clef='bass')

    # Note: no per-pair clef filter — notes near key-sig are kept as they
    # may include early measure notes. False key-sig detections are handled
    # by the music_symbols fill check in note_unit.py.

    print(f"   Treble: {len(treble_notes)} notes, Bass: {len(bass_notes)} notes")

    # ── 6. Detect Accidentals ──
    print("6. Detecting accidentals...")
    global_accs = detect_accidentals_global(binary, systems, dy,
                                               clef_boundaries=clef_boundaries,
                                               music_symbols=music_symbols)
    all_detected_notes = treble_notes + bass_notes
    accidentals_map = assign_accidentals_to_notes(global_accs, all_detected_notes, dy)
    n_sharps = sum(1 for v in accidentals_map.values() if v == '#')
    n_flats = sum(1 for v in accidentals_map.values() if v == 'b')
    print(f"   Total accidentals: {len(accidentals_map)} ({n_sharps} sharps, {n_flats} flats)")

    # ── 7. Detect Rests ──
    print("7. Detecting rests...")
    all_rests = detect_rests(binary, systems, dy)
    all_rests = [r for r in all_rests if r['x'] > int(img_w * 0.10)]
    all_rests = _filter_rests(all_rests, barlines_per_pair, treble_notes + bass_notes, dy,
                              music_symbols=music_symbols)
    print(f"   Found {len(all_rests)} rests (after filtering)")

    treble_rests, bass_rests = _split_rests_by_clef(all_rests, grand_staff_pairs)

    # ── 7b. Detect Tuplet Markers ──
    print("7b. Detecting tuplet markers...")
    original_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    tuplet_markers = detect_tuplet_markers(original_gray, grand_staff_pairs, dy,
                                           clef_boundaries=clef_boundaries)
    print(f"   Found {len(tuplet_markers)} tuplet markers")
    for m in tuplet_markers:
        print(f"      {m['n']} at x={m['x']}, y={m['y']} (pair {m['pair_idx']}, {m['clef']})")

    # ── 8. Track Stems ──
    print("8. Tracking stems...")
    for note in treble_notes + bass_notes:
        note['stem'] = track_stem(music_symbols, note, dy, binary=binary)
    print(f"   Tracked stems for {len(treble_notes) + len(bass_notes)} notes")

    # ── 9. Group into Chords, Segment Measures, Estimate Durations ──
    print("9. Grouping notes and formatting output...")
    pair_data = []

    # Running bpm state — inherited across systems unless a new clef-area
    # time sig is detected.
    from config import CFG as _CFG
    current_bpm = _CFG.duration.beats_per_measure

    for pair_idx, (treble_sys, bass_sys) in enumerate(grand_staff_pairs):
        pair_treble = [n for n in treble_notes if n.get('pair_idx') == pair_idx]
        pair_bass = [n for n in bass_notes if n.get('pair_idx') == pair_idx]
        pair_t_rests = [r for r in treble_rests if r.get('pair_idx') == pair_idx]
        pair_b_rests = [r for r in bass_rests if r.get('pair_idx') == pair_idx]

        barlines = barlines_per_pair[pair_idx] if pair_idx < len(barlines_per_pair) else []

        treble_units = build_note_units(pair_treble, music_symbols, binary, dy)
        bass_units = build_note_units(pair_bass, music_symbols, binary, dy)

        # Build per-system time-signature anchors. Scan treble (time sig
        # is shared with bass in grand staff), and inherit current_bpm
        # across systems.
        anchors, current_bpm = _build_system_timesig_anchors(
            binary, treble_sys, barlines, dy, current_bpm)
        bpm = anchors[0][1]  # bpm at start of this system, for merge heuristics
        if len(anchors) > 1 or anchors[0][1] != _CFG.duration.beats_per_measure:
            print(f"   Grand staff {pair_idx}: timesig anchors = "
                  f"{[(round(x,1), b) for x, b in anchors]}")

        treble_units = merge_overlapping_note_units(treble_units, beats_per_measure=bpm, dy=dy)
        bass_units = merge_overlapping_note_units(bass_units, beats_per_measure=bpm, dy=dy)

        is_first = (pair_idx == 0)
        pair_treble_markers = [m for m in tuplet_markers
                               if m['pair_idx'] == pair_idx and m['clef'] == 'treble']
        pair_bass_markers = [m for m in tuplet_markers
                             if m['pair_idx'] == pair_idx and m['clef'] == 'bass']
        treble_measures = segment_into_measures(treble_units, pair_t_rests, barlines, dy,
                                                beats_per_measure=bpm,
                                                is_first_system=is_first,
                                                tuplet_markers=pair_treble_markers,
                                                timesig_anchors=anchors)
        bass_measures = segment_into_measures(bass_units, pair_b_rests, barlines, dy,
                                              beats_per_measure=bpm,
                                              is_first_system=is_first,
                                              tuplet_markers=pair_bass_markers,
                                              timesig_anchors=anchors)

        # Strip leading empty measures for lines 2+ (clef/key-sig area before first barline)
        if pair_idx >= 1:
            while treble_measures and not treble_measures[0]:
                treble_measures.pop(0)
            while bass_measures and not bass_measures[0]:
                bass_measures.pop(0)

        pair_data.append({
            'treble_measures': treble_measures,
            'bass_measures': bass_measures,
            'barlines': barlines,
            'treble_sys': treble_sys,
            'bass_sys': bass_sys,
            'pair_idx': pair_idx,
            'timesig_anchors': anchors,
        })
        print(f"   Grand staff {pair_idx + 1}: "
              f"{len(treble_measures)} treble measures, {len(bass_measures)} bass measures")

    # ── 10. Format & Save Output ──
    _print_and_save_output(pair_data, accidentals_map, dy)

    # ── 11. Visualization ──
    print("\n11. Generating visualizations...")
    _generate_annotated_image(image_path, pair_data, accidentals_map,
                              treble_notes, bass_notes, grand_staff_pairs,
                              barlines_per_pair, dy)
    _generate_jianpu_only_image(pair_data, accidentals_map, dy)


# ============================================================
# Single-staff mode (solo instrument)
# ============================================================

def _barline_consensus(barlines_per_system, dy, binary=None,
                       systems=None, music_symbols=None):
    """Rescue missing barlines using measure-width consistency.

    If a staff has one measure much wider than others (>1.8× median),
    there's likely a missing barline inside it. Search for a high-density
    vertical line in the middle of the wide measure.
    """
    if binary is None or systems is None:
        return barlines_per_system

    # Compute reference median span from the staff with the most barlines.
    # This staff has the most reliable detection, so its measure widths
    # represent the expected pattern.
    all_spans = []
    best_staff_spans = []
    best_count = 0
    for bl_list in barlines_per_system:
        sorted_bl = sorted(bl_list)
        spans = [sorted_bl[i + 1] - sorted_bl[i] for i in range(len(sorted_bl) - 1)]
        all_spans.extend(spans)
        if len(bl_list) > best_count:
            best_count = len(bl_list)
            best_staff_spans = spans
    ref_median = sorted(best_staff_spans)[len(best_staff_spans) // 2] if best_staff_spans else 300

    result = []
    for si, bl_list in enumerate(barlines_per_system):
        augmented = list(bl_list)
        if len(bl_list) < 2:
            result.append(augmented)
            continue

        sorted_bl = sorted(bl_list)
        spans = [sorted_bl[i + 1] - sorted_bl[i] for i in range(len(sorted_bl) - 1)]
        if not spans:
            result.append(augmented)
            continue
        median_span = ref_median  # use global reference
        if median_span < dy * 5:
            result.append(augmented)
            continue

        # Check for overly wide measures
        for i in range(len(spans)):
            if spans[i] > median_span * 1.8:
                # This measure is too wide — search for a barline inside
                left_x = sorted_bl[i]
                right_x = sorted_bl[i + 1]
                search_x1 = left_x + int(median_span * 0.4)
                search_x2 = right_x - int(median_span * 0.4)
                if search_x2 <= search_x1:
                    continue

                # Search for barline candidates that split the wide measure
                # into sub-measures close to the median width.
                sys_info = systems[si] if si < len(systems) else None
                if sys_info is None:
                    continue
                top, bot = int(sys_info[0]), int(sys_info[4])
                margin_y = int(dy * 0.3)
                y1 = max(0, top - margin_y)
                y2 = min(binary.shape[0], bot + margin_y)
                src = music_symbols if music_symbols is not None else binary
                roi = src[y1:y2, left_x:right_x]
                col_max = float(y2 - y1)
                if col_max < 1:
                    continue
                proj = np.sum(roi > 127, axis=0).astype(float) / col_max

                # Find all narrow peaks above threshold
                pk_threshold = 0.55
                candidates = []
                j = 0
                while j < len(proj):
                    if proj[j] > pk_threshold:
                        pk_start = j
                        while j < len(proj) and proj[j] > pk_threshold:
                            j += 1
                        pk_w = j - pk_start
                        if pk_w < dy * 0.5:
                            pk_cx = left_x + (pk_start + j) // 2
                            pk_val = float(np.max(proj[pk_start:j]))
                            candidates.append((pk_cx, pk_val))
                    else:
                        j += 1

                # Pick candidates that best split the wide measure into
                # sub-measures close to the median width. Try each candidate
                # and score by how close the resulting sub-measures are.
                best_additions = []
                for cx, val in candidates:
                    lg = cx - left_x
                    rg = right_x - cx
                    # Each sub-measure should be within 0.5-2× median
                    if (median_span * 0.5 < lg < median_span * 2.0 and
                            median_span * 0.5 < rg < median_span * 2.0):
                        fit = abs(lg - median_span) + abs(rg - median_span)
                        best_additions.append((fit, cx))

                if best_additions:
                    best_additions.sort()
                    augmented.append(best_additions[0][1])

        augmented.sort()
        result.append(augmented)

    return result


def _detect_barlines_single_staff(binary, systems, dy,
                                   notehead_xs_per_staff=None,
                                   music_symbols=None,
                                   clef_boundaries=None):
    """Detect barlines for single-staff scores.

    Uses vertical projection on music_symbols (staff lines removed) to find
    columns with strong vertical ink spanning the full staff.  A barline is
    an isolated thin vertical line; a stem is attached to a notehead.
    """
    from config import CFG
    bc = CFG.barline
    img_h, img_w = binary.shape
    all_barlines = []

    for si, sys_info in enumerate(systems):
        top, bot = int(sys_info[0]), int(sys_info[4])
        staff_h = bot - top

        # --- Step 1: Vertical projection on music_symbols ---
        src = music_symbols if music_symbols is not None else binary
        margin = int(dy * 0.3)
        y1 = max(0, top - margin)
        y2 = min(img_h, bot + margin)
        roi = src[y1:y2, :]
        proj = np.sum(roi > 127, axis=0).astype(float)

        col_max = float(y2 - y1)
        if col_max < 1:
            all_barlines.append([])
            continue
        proj_norm = proj / col_max

        # --- Step 2: Find peaks in the projection ---
        # Adaptive threshold: barlines+stems are the highest peaks on
        # the staff, so scale by per-staff max fill. This handles
        # staves whose top/bottom line was mis-detected, inflating
        # the ROI and depressing absolute fill ratios.
        # Empirically:
        #   Clean staff  — max≈0.88, barline≈0.79, stem≈0.88
        #   Warped staff — max≈0.79, barline≈0.55, stem≈0.79
        #   Flat symbol  — ≈0.48 in both regimes
        # 0.60 * max separates barlines from flats in both cases.
        max_fill = float(proj_norm.max()) if len(proj_norm) else 0.0
        threshold = max_fill * 0.60
        peaks = []
        in_peak = False
        peak_start = 0
        for x in range(len(proj_norm)):
            if proj_norm[x] > threshold:
                if not in_peak:
                    peak_start = x
                    in_peak = True
            else:
                if in_peak:
                    peak_end = x
                    peak_w = peak_end - peak_start
                    if peak_w < dy * 0.5:
                        center_x = (peak_start + peak_end) // 2
                        max_val = float(np.max(proj_norm[peak_start:peak_end]))
                        peaks.append((center_x, max_val, peak_w))
                    in_peak = False

        # --- Step 3: Score each peak ---
        # Use overlap-based stem detection: a candidate that falls within
        # a notehead bounding box is a stem, not a barline.
        scored = []
        nh_xs = notehead_xs_per_staff[si] if (notehead_xs_per_staff and
                                               si < len(notehead_xs_per_staff)) else []
        for cx, val, pw in peaks:
            # Check if candidate overlaps any notehead bbox
            on_notehead = False
            for nx_left, nx_right in nh_xs:
                if nx_left - 2 <= cx <= nx_right + 2:
                    on_notehead = True
                    break

            half_score = _half_staff_score(binary, sys_info, cx)

            score = val * 0.5 + half_score * 0.5
            if on_notehead:
                score *= 0.2  # heavy penalty for stems

            scored.append((cx, score))

        # --- Step 4: Filter by spacing and boundaries ---
        scored.sort(key=lambda p: p[0])

        if len(scored) >= 2:
            min_gap = dy * 8
            deduped = [scored[0]]
            for cx, sc in scored[1:]:
                if cx - deduped[-1][0] > min_gap:
                    deduped.append((cx, sc))
                elif sc > deduped[-1][1]:
                    deduped[-1] = (cx, sc)
            scored = deduped

        scored = [(cx, sc) for cx, sc in scored if sc > 0.35]

        # Use adaptive clef boundary if available, else config
        clef_end = (clef_boundaries.get(si, int(img_w * bc.clef_end_ratio))
                    if clef_boundaries else int(img_w * bc.clef_end_ratio))
        right_bound = int(img_w * bc.right_boundary_ratio)
        result_scored = [(int(cx), sc) for cx, sc in scored
                         if clef_end < cx < right_bound]

        # Spacing consistency: remove barlines that create gaps much
        # shorter than the median gap.  Such barlines are usually stems
        # that slipped through the notehead-overlap filter.
        #
        # Guard: only remove if the candidate is also notably WEAKER
        # than its neighbors. A real barline has a score comparable to
        # other barlines; a stem-leftover scores ~0.2× due to the
        # notehead-overlap penalty. Without this guard we would remove
        # legitimate barlines whenever a staff has a bimodal measure
        # layout (e.g. short whole-note measures next to dense ones).
        if len(result_scored) >= 3:
            xs = [x for x, _ in result_scored]
            scs = [s for _, s in result_scored]
            gaps = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
            median_gap = float(sorted(gaps)[len(gaps) // 2])
            min_gap_allowed = median_gap * bc.min_gap_median_ratio
            keep = [True] * len(result_scored)
            for i in range(1, len(result_scored) - 1):
                left_gap = xs[i] - xs[i - 1]
                right_gap = xs[i + 1] - xs[i]
                if left_gap < min_gap_allowed and right_gap < min_gap_allowed:
                    merged = xs[i + 1] - xs[i - 1]
                    if abs(merged - median_gap) < abs(left_gap - median_gap):
                        # Only remove if this candidate is clearly
                        # weaker than both neighbors — real barlines
                        # cluster in score; stems sit much lower.
                        nbr_min = min(scs[i - 1], scs[i + 1])
                        if scs[i] < nbr_min * 0.7:
                            keep[i] = False
            result_scored = [rs for rs, k in zip(result_scored, keep) if k]

        result = [x for x, _ in result_scored]

        # Keep end-of-line barline — segment_into_measures merges
        # content after the last barline into the final measure.

        all_barlines.append(result)

    return all_barlines


def _nearest_staff(y_center, systems):
    """Return the index of the staff system closest to y_center."""
    return min(range(len(systems)),
               key=lambda i: abs(y_center - (systems[i][0] + systems[i][4]) / 2.0))


def _detect_clef_boundaries(all_notes, systems, dy, clef_area_x):
    """Detect per-staff clef boundaries using the leftmost high-confidence notehead.

    Returns dict mapping staff_index -> x boundary.
    """
    clef_boundaries = {}
    for si, sys_info in enumerate(systems):
        sys_mid = (sys_info[0] + sys_info[4]) / 2.0
        raw = [
            n for n in all_notes
            if abs(n['y_center'] - sys_mid) < dy * 5
            and n.get('score', 1.0) >= 0.70
            and n['x'] > int(dy * 4)
        ]
        candidates = []
        for n in raw:
            if n['x'] < clef_area_x:
                has_vertical_sibling = any(
                    m is not n
                    and abs(m['x'] - n['x']) <= 2
                    and abs(m['y_center'] - n['y_center']) > dy * 2
                    for m in raw
                )
                if has_vertical_sibling:
                    continue
            candidates.append(n)
        if candidates:
            first_x = min(n['x'] for n in candidates)
            boundary = max(int(dy * 4), first_x - int(dy))
        else:
            boundary = clef_area_x
        clef_boundaries[si] = boundary
    return clef_boundaries


def _filter_rests_single_staff(all_rests, systems, dy, img_w, barlines_per_system,
                                clef_boundaries, all_detected_notes, music_symbols):
    """Filter rests for single-staff scores: remove clef-area, barline-near,
    note-overlapping, and invalid block/small rests."""
    filtered = []
    for rest in all_rests:
        si = rest['system_idx']
        if si < len(barlines_per_system):
            barlines = barlines_per_system[si]
            near_barline = any(abs(rest['x'] - bx) < dy * 3.5 for bx in barlines)
            clef_x = (clef_boundaries.get(si, int(img_w * 0.17))
                      if clef_boundaries else int(img_w * 0.17))
            if rest['x'] < clef_x:
                near_barline = True
            if near_barline:
                continue
        has_nearby = any(
            note.get('pair_idx', -1) == si
            and abs(rest['x'] - note['x']) < dy * 2.0
            for note in all_detected_notes
        )
        if not has_nearby:
            filtered.append(rest)
    filtered = [r for r in filtered if _validate_block_rest(music_symbols, r, dy)]
    filtered = [r for r in filtered if _validate_small_rest(music_symbols, r, dy)]
    return filtered


def _main_single_staff(image_path, systems, staff_lines, music_symbols, binary, dy):
    """Process a single-staff score (solo instrument like 二胡, 笛子, etc.)."""
    from config import CFG
    img_w = binary.shape[1]

    # ── 3. Detect Noteheads (before barlines — needed for stem filtering) ──
    print("3. Detecting noteheads...")
    boxes, template, exclusion_zones = find_noteheads(
        binary, dy, threshold=0.55, staff_systems=systems,
        music_symbols=music_symbols, detect_hollow=True)

    all_notes = [
        {'x': x, 'y': y, 'w': w, 'h': h, 'y_center': y + h // 2, 'score': score}
        for x, y, w, h, score in boxes
    ]
    print(f"   Detected {len(all_notes)} noteheads total")

    # Filter clef area — use first system boundary for line 1, adaptive for rest
    from template_matching import create_notehead_template
    nh_template = create_notehead_template(dy)
    th, tw = nh_template.shape

    clef_area_x = int(img_w * 0.17)
    clef_boundaries = _detect_clef_boundaries(all_notes, systems, dy, clef_area_x)

    # Filter notes by clef boundary
    all_notes = [n for n in all_notes
                 if n['x'] > clef_boundaries.get(_nearest_staff(n['y_center'], systems),
                                                  clef_area_x)]
    print(f"   After clef area filtering: {len(all_notes)} notes")

    # Build per-staff notehead x-ranges for barline filtering
    notehead_xs_per_staff = [[] for _ in systems]
    for n in all_notes:
        best_si = _nearest_staff(n['y_center'], systems)
        notehead_xs_per_staff[best_si].append((n['x'], n['x'] + n['w']))

    # ── 4. Detect Barlines (using noteheads to filter stems) ──
    print("4. Detecting barlines...")
    barlines_per_system = _detect_barlines_single_staff(binary, systems, dy,
                                                         notehead_xs_per_staff,
                                                         music_symbols=music_symbols,
                                                         clef_boundaries=clef_boundaries)
    # ── 4b. Remove barlines at time-signature positions ──
    # The fraction line of a time signature (e.g. "4/4") is a short
    # horizontal bar that the barline detector can match. Remove any
    # barline within 2*dy of a detected TS digit position.
    for si, sys_info in enumerate(systems):
        bl = barlines_per_system[si]
        if not bl:
            continue
        ts_dets = detect_time_signatures_along_system(binary, sys_info, bl, dy)
        ts_match_xs = []
        for d in ts_dets:
            if d.get('time_sig') is None:
                continue  # skip failed detections
            mx = d.get('match_x')
            if mx is not None:
                ts_match_xs.append(mx)
            elif d.get('x', 0) > 0:
                ts_match_xs.append(d['x'])
        if ts_match_xs:
            before = len(bl)
            bl = [b for b in bl
                  if not any(abs(b - tx) < dy * 2 for tx in ts_match_xs)]
            removed = before - len(bl)
            if removed:
                barlines_per_system[si] = bl

    for si, bl in enumerate(barlines_per_system):
        print(f"   Staff {si}: {len(bl)} barlines at x={bl}")

    # ── 5. Assign Notes to Staves ──
    print("5. Assigning notes to staves...")
    notes_per_staff = [[] for _ in systems]
    for n in all_notes:
        best_si = _nearest_staff(n['y_center'], systems)
        sys_info = systems[best_si]
        if abs(n['y_center'] - (sys_info[0] + sys_info[4]) / 2.0) < dy * 8:
            n['system'] = sys_info
            n['pair_idx'] = best_si
            n['clef'] = 'treble'
            notes_per_staff[best_si].append(n)

    # Filter false positives
    for si in range(len(systems)):
        notes_per_staff[si] = filter_false_positive_notes(notes_per_staff[si], dy, clef='treble')
        # Additional filter for single-staff: remove notes above the staff
        # near the left edge (measure numbers, rehearsal marks).
        sys_info = systems[si]
        staff_top = sys_info[0]
        boundary = clef_boundaries.get(si, clef_area_x)
        clean = []
        for n in notes_per_staff[si]:
            # Notes in the immediate clef area that are above the staff
            # AND small (< 0.5 score) are likely measure numbers.
            if (n['x'] < boundary + dy * 1
                    and n['y_center'] < staff_top - dy * 0.3
                    and n['score'] < 0.70):
                continue
            # Filter notes far above the staff (> 2.5*dy) — these are tempo
            # markings, rehearsal marks, or other text, not real notes.
            if n['y_center'] < staff_top - dy * 2.5:
                continue
            clean.append(n)
        notes_per_staff[si] = clean

    # ── 5b. Single-staff deduplication ──
    # Solo instruments can't play chords — if two noteheads are at nearly
    # the same x, one is a false positive.  Keep the higher-scored one.
    # Exception: when both notes have equal scores and different y-positions,
    # they are likely real sequential notes in a dense beam group (e.g.,
    # sixteenths 5-7 px apart), not duplicates.
    for si in range(len(systems)):
        notes = notes_per_staff[si]
        # Sort by x for pairwise comparison
        notes.sort(key=lambda n: n['x'])
        keep = [True] * len(notes)
        for i in range(len(notes)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(notes)):
                if notes[j]['x'] - notes[i]['x'] > dy * 1.0:
                    break
                if not keep[j]:
                    continue
                # If both notes have equal scores and different y-positions,
                # they are likely different notes in a dense beam group.
                y_diff = abs(notes[i]['y_center'] - notes[j]['y_center'])
                if abs(notes[i]['score'] - notes[j]['score']) < 0.01 and y_diff > dy * 0.5:
                    continue
                # Two notes within 1*dy in x — remove the lower-scored one
                if notes[i]['score'] >= notes[j]['score']:
                    keep[j] = False
                else:
                    keep[i] = False
                    break
        before = len(notes)
        notes_per_staff[si] = [n for n, k in zip(notes, keep) if k]
        removed = before - len(notes_per_staff[si])
        if removed:
            print(f"   Staff {si+1}: removed {removed} duplicate note(s)")

    total = sum(len(ns) for ns in notes_per_staff)
    print(f"   Total notes after filtering: {total}")

    # ── 5c. Reject hollow notehead false positives at time-sig digits ──
    # A "2"/"3"/"5" numeral in a time signature has rounded curves that
    # the hollow oval template matches at raw score ~0.35; these are
    # stored with score floor 0.80. Reject any floor-score hollow
    # detection whose x is within 1.5*dy of a detected time-signature
    # digit position. This covers BOTH mid-staff TS changes (where d['x']
    # is the snapped-barline anchor) AND the clef-area TS (where d['x']=0
    # is the anchor and d['match_x'] holds the actual digit position).
    for si, sys_info in enumerate(systems):
        bl = barlines_per_system[si] if si < len(barlines_per_system) else []
        ts_dets = detect_time_signatures_along_system(binary, sys_info, bl, dy)
        ts_xs = []
        for d in ts_dets:
            if d.get('source') == 'clef':
                mx = d.get('match_x')
                if mx is not None:
                    ts_xs.append(mx)
            else:
                ts_xs.append(d['x'])
        if not ts_xs:
            continue
        before = len(notes_per_staff[si])
        kept = []
        for n in notes_per_staff[si]:
            if (n.get('score', 1.0) <= 0.81
                    and any(abs(n['x'] - tx) < dy * 1.5 for tx in ts_xs)):
                continue
            kept.append(n)
        removed = before - len(kept)
        if removed:
            notes_per_staff[si] = kept
            print(f"   Staff {si+1}: removed {removed} hollow notehead(s) "
                  f"near time-sig")

    # ── 5d. Hollow chord companion scan ──
    # For confirmed hollow noteheads (score ≈ 0.80), search the same x on
    # music_symbols for additional notehead outlines. Only runs when
    # filled-note chords exist (multi-voice evidence after all filtering).
    has_chords = False
    for ns in notes_per_staff:
        xs_sorted = sorted(n['x'] for n in ns)
        for xi in range(len(xs_sorted) - 1):
            if xs_sorted[xi + 1] - xs_sorted[xi] < dy * 0.8:
                pair = [n for n in ns if abs(n['x'] - xs_sorted[xi]) < dy * 0.8]
                ys = [n['y_center'] for n in pair]
                if len(ys) >= 2 and max(ys) - min(ys) > dy * 0.8:
                    has_chords = True
                    break
        if has_chords:
            break
    if has_chords:
        added_total = 0
        for si in range(len(systems)):
            sys_info = systems[si]
            s_dy = (sys_info[4] - sys_info[0]) / 4.0
            hollow_notes = [n for n in notes_per_staff[si]
                            if n.get('score', 1.0) <= 0.81]
            new_notes = []
            for hn in hollow_notes:
                # Skip if this anchor already has a chord partner
                has_partner = any(
                    abs(n['x'] - hn['x']) < dy * 0.8
                    and abs(n['y_center'] - hn['y_center']) > dy * 0.8
                    for n in notes_per_staff[si] if n is not hn
                )
                if has_partner:
                    continue
                cx = hn['x'] + hn.get('w', 0) // 2
                cy = hn['y_center']
                search_r = int(s_dy * 4.0)
                hw = int(s_dy * 0.7)
                ry1 = max(0, int(cy - search_r))
                ry2 = min(music_symbols.shape[0], int(cy + search_r))
                rx1 = max(0, int(cx - hw))
                rx2 = min(music_symbols.shape[1], int(cx + hw))
                patch = music_symbols[ry1:ry2, rx1:rx2]
                if patch.size == 0:
                    continue
                row_ink = np.sum(patch > 127, axis=1).astype(float)
                has_ink_arr = (row_ink > 4).astype(np.uint8)
                runs = np.diff(np.concatenate([[0], has_ink_arr, [0]]))
                starts = np.where(runs == 1)[0]
                ends = np.where(runs == -1)[0]
                if len(starts) == 0:
                    continue
                max_gap = max(3, int(s_dy * 0.25))
                ms_list, me_list = [starts[0]], [ends[0]]
                for ri in range(1, len(starts)):
                    if starts[ri] - me_list[-1] <= max_gap:
                        me_list[-1] = ends[ri]
                    else:
                        ms_list.append(starts[ri])
                        me_list.append(ends[ri])
                for mi in range(len(ms_list)):
                    rs, re_ = ms_list[mi], me_list[mi]
                    seg_ink = row_ink[rs:re_]
                    seg_total = np.sum(seg_ink)
                    run_len = re_ - rs
                    if seg_total <= 30 or run_len < s_dy * 0.3 or run_len > s_dy * 1.5:
                        continue
                    ys_arr = np.arange(len(seg_ink))
                    comp_cy = ry1 + rs + int(np.sum(ys_arr * seg_ink) / seg_total)
                    if abs(comp_cy - cy) < s_dy * 0.8:
                        continue
                    # Verify hollow on music_symbols (staff lines removed)
                    vr = max(2, int(s_dy * 0.2))
                    vy1 = max(0, int(comp_cy - s_dy * 0.3))
                    vy2 = min(music_symbols.shape[0], int(comp_cy + s_dy * 0.3))
                    vx1, vx2 = max(0, cx - vr), min(music_symbols.shape[1], cx + vr)
                    cfill = np.mean(music_symbols[vy1:vy2, vx1:vx2] > 127) if (vy2 > vy1 and vx2 > vx1) else 1.0
                    if cfill >= 0.80:
                        continue
                    check_list = list(notes_per_staff[si]) + new_notes
                    already = any(abs(n['x'] - hn['x']) < dy * 0.5
                                  and abs(n['y_center'] - comp_cy) < dy * 0.5
                                  for n in check_list)
                    if already:
                        continue
                    new_notes.append({
                        'x': hn['x'], 'y': int(comp_cy - hn.get('h', int(dy)) // 2),
                        'w': hn.get('w', int(dy)), 'h': hn.get('h', int(dy)),
                        'y_center': comp_cy, 'score': 0.80,
                        'pair_idx': si, 'system': sys_info,
                        'companion': True
                    })
            if new_notes:
                notes_per_staff[si].extend(new_notes)
                added_total += len(new_notes)
        if added_total:
            print(f"   Added {added_total} hollow chord companion(s)")

    # ── 5e. Remove barlines that create empty measures ──
    # Only for multi-voice scores where false barlines are more common
    removed_barlines_total = 0
    if not has_chords:
        pass  # skip for single-voice scores
    for si in range(len(systems)) if has_chords else []:
        blines = barlines_per_system[si]
        if len(blines) < 2:
            continue
        notes_xs = sorted(n['x'] for n in notes_per_staff[si])
        if not notes_xs:
            continue
        clef_end = clef_boundaries.get(si, 0) if clef_boundaries else 0
        boundaries = [clef_end] + blines
        # Compute median barline gap for this staff
        all_gaps = [blines[i+1] - blines[i] for i in range(len(blines)-1)]
        if clef_end > 0 and blines:
            all_gaps.insert(0, blines[0] - clef_end)
        median_gap = float(sorted(all_gaps)[len(all_gaps)//2]) if all_gaps else 0

        to_remove = set()
        for bi in range(1, len(boundaries)):
            seg_left = boundaries[bi - 1]
            seg_right = boundaries[bi]
            seg_w = seg_right - seg_left
            notes_in = [n for n in notes_per_staff[si]
                        if seg_left < n['x'] < seg_right]
            # A lone hollow note (score=0.80) in a narrow segment is
            # likely a false detection; don't let it block barline removal
            non_hollow = [n for n in notes_in
                          if n.get('score', 1.0) != 0.80]
            real_notes = non_hollow if non_hollow else []
            if not non_hollow and len(notes_in) >= 2:
                real_notes = notes_in
            has_notes = len(real_notes) > 0
            if has_notes or bi >= len(boundaries) - 1:
                continue
            # Case 1: segment has lone hollow note(s) only — likely
            # a false hollow detection (score=0.80 baseline)
            has_only_hollow = (len(notes_in) > 0 and len(real_notes) == 0)
            # Case 2: TRULY narrow segment (false barline from stem
            # artifact). Real empty rest measures occupy ~75-100% of the
            # median gap, so threshold must be well below that. Stems
            # create segments < 50% of median.
            is_narrow = (median_gap > 0 and seg_w < median_gap * 0.50)
            if is_narrow:
                # narrow → false barline regardless of contents
                to_remove.add(boundaries[bi])
            elif has_only_hollow:
                # measure-sized segment with only an isolated hollow
                # detection → legitimate empty measure with a false
                # hollow positive. Keep barline; drop the false note.
                fp_xs = {(n['x'], n['y_center']) for n in notes_in}
                notes_per_staff[si] = [
                    n for n in notes_per_staff[si]
                    if (n['x'], n['y_center']) not in fp_xs
                ]
        if to_remove:
            barlines_per_system[si] = [b for b in blines if b not in to_remove]
            removed_barlines_total += len(to_remove)
            # Remove companion notes that were alone in removed segments
            empty_segs = []
            for bi2 in range(1, len(boundaries)):
                if boundaries[bi2] in to_remove:
                    empty_segs.append((boundaries[bi2 - 1], boundaries[bi2]))
            notes_per_staff[si] = [
                n for n in notes_per_staff[si]
                if not (n.get('companion', False)
                        and any(sl < n['x'] < sr for sl, sr in empty_segs))
            ]
    if removed_barlines_total:
        print(f"   Removed {removed_barlines_total} barline(s) creating empty measures")

    # ── 6. Detect Accidentals ──
    print("6. Detecting accidentals...")
    global_accs = detect_accidentals_global(binary, systems, dy,
                                             clef_boundaries=clef_boundaries,
                                             music_symbols=music_symbols)
    all_detected_notes = [n for ns in notes_per_staff for n in ns]
    accidentals_map = assign_accidentals_to_notes(global_accs, all_detected_notes, dy)
    print(f"   Accidentals: {len(accidentals_map)}")

    # ── 7. Detect Rests ──
    print("7. Detecting rests...")
    all_rests = detect_rests(binary, systems, dy)
    all_rests = [r for r in all_rests if r['x'] > int(img_w * 0.10)]
    for r in all_rests:
        r['pair_idx'] = r['system_idx']
    all_rests = _filter_rests_single_staff(
        all_rests, systems, dy, img_w, barlines_per_system,
        clef_boundaries, all_detected_notes, music_symbols)
    print(f"   Found {len(all_rests)} rests")

    rests_per_staff = [[] for _ in systems]
    for r in all_rests:
        si = r['system_idx']
        if si < len(rests_per_staff):
            r['pair_idx'] = si
            rests_per_staff[si].append(r)

    # ── 7b. Detect Tuplet Markers ──
    print("7b. Detecting tuplet markers...")
    original_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # For single staff, create fake grand staff pairs for tuplet detection
    fake_pairs = [(sys_info, sys_info) for sys_info in systems]
    tuplet_markers = detect_tuplet_markers(original_gray, fake_pairs, dy,
                                           clef_boundaries=clef_boundaries)
    print(f"   Found {len(tuplet_markers)} tuplet markers")

    # ── 8. Track Stems ──
    print("8. Tracking stems...")
    for ns in notes_per_staff:
        for note in ns:
            note['stem'] = track_stem(music_symbols, note, dy, binary=binary)

    # ── 8b. Auto-detect time signature ──
    time_sig = detect_time_signature(binary, systems[0], dy)
    if time_sig:
        num, den = time_sig
        bpm_detected = _timesig_to_bpm(num, den)
        CFG.duration.beats_per_measure = bpm_detected
        print(f"   Time signature: {num}/{den} → beats_per_measure={bpm_detected}")
    else:
        print(f"   Time signature: not detected, using default {CFG.duration.beats_per_measure}")

    # ── 9. Build Note Units & Segment Measures ──
    print("9. Building note units and segmenting measures...")
    current_bpm = CFG.duration.beats_per_measure
    staff_data = []

    for si, sys_info in enumerate(systems):
        notes = notes_per_staff[si]
        rests = rests_per_staff[si]
        barlines = barlines_per_system[si] if si < len(barlines_per_system) else []

        units = build_note_units(notes, music_symbols, binary, dy, single_staff=True)
        # No merge_overlapping for single-staff (no two-voice)

        anchors, current_bpm = _build_system_timesig_anchors(
            binary, sys_info, barlines, dy, current_bpm)
        bpm = anchors[0][1]
        if len(anchors) > 1 or anchors[0][1] != CFG.duration.beats_per_measure:
            print(f"   Staff {si + 1}: timesig anchors = "
                  f"{[(round(x,1), b) for x, b in anchors]}")

        is_first = (si == 0)
        staff_tuplets = [m for m in tuplet_markers
                         if m['pair_idx'] == si and m['clef'] == 'treble']
        measures = segment_into_measures(units, rests, barlines, dy,
                                          beats_per_measure=bpm,
                                          is_first_system=is_first,
                                          tuplet_markers=staff_tuplets,
                                          timesig_anchors=anchors,
                                          fill_to_measure=True)

        # Multi-measure rest expansion. A segment whose physical width
        # represents N empty measures (drawn as a thick bar with an
        # Arabic numeral above) is detected here and expanded into N
        # beat-filled empty measures so downstream formatting matches
        # the engraver's intent.
        sorted_bls = sorted(barlines)
        expanded = []
        for mi_x, m in enumerate(measures):
            left = sorted_bls[mi_x - 1] if 0 < mi_x <= len(sorted_bls) else 0
            right = (sorted_bls[mi_x] if mi_x < len(sorted_bls)
                     else (sorted_bls[-1] if sorted_bls else 0))
            n_rest = detect_multi_rest_count(binary, sys_info, left, right, dy)
            # When a clear multi-rest pattern (thick bar + numeral) is
            # detected we trust it: any note_units inside the segment are
            # almost certainly false positives from the bar / numeral
            # tripping the notehead matcher. Drop them via the expansion.
            if n_rest is not None and n_rest >= 2:
                fills = _fill_rests_for_gap(0.0, bpm)
                # For large multi-rests (≥5), store as a single compact
                # measure with a count tag instead of N identical empties.
                if n_rest >= 5:
                    proto = []
                    for j, fdur in enumerate(fills):
                        proto.append({
                            'type': 'rest',
                            'x': float(left) + 1 + j,
                            'duration': fdur,
                            'duration_source': 'multi_rest',
                        })
                    proto.append({
                        'type': 'multi_rest_count',
                        'x': float(left),
                        'count': n_rest,
                    })
                    expanded.append(proto)
                else:
                    for k in range(n_rest):
                        proto = []
                        for j, fdur in enumerate(fills):
                            proto.append({
                                'type': 'rest',
                                'x': float(left) + 1 + k * 1000 + j,
                                'duration': fdur,
                                'duration_source': 'multi_rest',
                            })
                        expanded.append(proto)
                print(f"   Staff {si + 1}: multi-rest at measure {mi_x}"
                      f" → expanded to {n_rest} measures")
                continue
            expanded.append(m)
        measures = expanded

        # Strip leading empty measures for lines 2+
        if si >= 1:
            while measures and not measures[0]:
                measures.pop(0)

        staff_data.append({
            'measures': measures,
            'barlines': barlines,
            'system': sys_info,
            'staff_idx': si,
            'timesig_anchors': anchors,
        })
        print(f"   Staff {si + 1}: {len(measures)} measures")

    # ── 10. Format & Save Output ──
    # skip_empty=True: legitimate empty measures are filled with rests
    # by fill_to_measure (so they format as "0 0 0 0" and pass through),
    # while false-barline narrow empties remain as "0 0" and get dropped.
    _print_and_save_single_staff_output(staff_data, accidentals_map, dy,
                                        skip_empty=has_chords)

    # ── 11. Visualization ──
    print("\n11. Generating visualizations...")
    _generate_single_staff_annotated(image_path, staff_data, accidentals_map, dy)
    # Visual jianpu (matches reference repo's PIL-rendered style:
    # red digits + 减时线 + octave dots + dashes for half/whole)
    from jianpu_visual import render_full_image
    render_full_image(image_path, staff_data, accidentals_map, dy,
                      "output_jianpu_visual.png")
    print("   Saved: output_jianpu_visual.png")


def _print_and_save_single_staff_output(staff_data, accidentals_map, dy,
                                         skip_empty=False):
    """Format, print, and save output for single-staff scores."""
    all_lines = []
    for sd in staff_data:
        line = format_output(sd['measures'], accidentals_map, dy=dy,
                              skip_empty=skip_empty)
        all_lines.append(line)

    print(f"\n{'=' * 60}")
    print("JIANPU OUTPUT")
    print(f"{'=' * 60}")

    for i, line in enumerate(all_lines):
        print(f"\n--- 第{i + 1}行 ---")
        print(line)

    # Save to file
    with open("output_jianpu.txt", "w", encoding="utf-8") as f:
        f.write("简谱翻译结果\n")
        f.write("=" * 60 + "\n\n")
        for i, line in enumerate(all_lines):
            f.write(f"--- 第{i + 1}行 ---\n")
            f.write(f"{line}\n\n")
        f.write("=" * 60 + "\n")
    print("\n   Saved text: output_jianpu.txt")

    # Confidence report
    from confidence import format_confidence_report
    from config import CFG
    staff_measures = [(f"第{i + 1}行", sd['measures'])
                      for i, sd in enumerate(staff_data)]
    staff_anchors = [sd.get('timesig_anchors') for sd in staff_data]
    report = format_confidence_report(staff_measures, accidentals_map,
                                      beats_per_measure=CFG.duration.beats_per_measure,
                                      dy=dy,
                                      staff_anchors=staff_anchors)
    with open("output_confidence.txt", "w", encoding="utf-8") as f:
        f.write(report)
    print("   Saved confidence: output_confidence.txt")


def _generate_single_staff_annotated(image_path, staff_data, accidentals_map, dy):
    """Generate annotated image for single-staff scores."""
    img_orig = cv2.imread(image_path)
    if img_orig is None:
        return
    orig_h, orig_w = img_orig.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    annotation_height = int(dy * 2.5)
    num_staves = len(staff_data)

    new_h = orig_h + num_staves * annotation_height
    img_out = np.ones((new_h, orig_w, 3), dtype=np.uint8) * 255

    y_offset = 0
    prev_copy_end = 0

    for si, sd in enumerate(staff_data):
        sys_info = sd['system']
        barlines = sd['barlines']
        measures = sd['measures']

        cut_y = sys_info[4] + int(dy * 1.5)
        if si + 1 < num_staves:
            next_top = staff_data[si + 1]['system'][0]
            cut_y = min(cut_y, next_top - int(dy * 0.5))
        else:
            cut_y = min(cut_y, orig_h)

        strip = img_orig[prev_copy_end:cut_y, :]
        img_out[y_offset:y_offset + strip.shape[0], :strip.shape[1]] = strip
        y_offset += strip.shape[0]

        ann_y1 = y_offset
        ann_y2 = y_offset + annotation_height
        img_out[ann_y1:ann_y2, :] = (245, 245, 255)

        text_y = ann_y1 + int(annotation_height * 0.6)
        boundaries = [0] + list(barlines) + [orig_w]

        for mi in range(len(measures)):
            if mi >= len(boundaries) - 1:
                break
            left = boundaries[mi]
            right = boundaries[mi + 1]
            region_w = right - left

            if mi > 0:
                cv2.line(img_out, (left, ann_y1 + 2), (left, ann_y2 - 2), (150, 150, 150), 1)

            text = format_measure(measures[mi], accidentals_map, measure_idx=mi, dy=dy)
            if text:
                scale = 0.35
                (tw, _), _ = cv2.getTextSize(text, font, scale, 1)
                if tw > region_w - 8:
                    scale = max(0.2, scale * (region_w - 8) / tw)
                (tw, _), _ = cv2.getTextSize(text, font, scale, 1)
                tx = left + max(2, (region_w - tw) // 2)
                cv2.putText(img_out, text, (tx, text_y), font, scale,
                            (180, 0, 0), 1, cv2.LINE_AA)

        y_offset += annotation_height
        prev_copy_end = cut_y

    if prev_copy_end < orig_h:
        remaining = img_orig[prev_copy_end:, :]
        end = min(y_offset + remaining.shape[0], img_out.shape[0])
        h_to_copy = end - y_offset
        img_out[y_offset:end, :remaining.shape[1]] = remaining[:h_to_copy]

    cv2.imwrite("output_jianpu_on_staff.png", img_out)
    print("   Saved: output_jianpu_on_staff.png")


# ============================================================
# Helper functions
# ============================================================

def _col_density(binary, y1, y2, bx, half_w=2):
    """Max mean-column density in binary[y1:y2, bx-half_w:bx+half_w+1]."""
    col = binary[y1:y2, max(0, bx - half_w):bx + half_w + 1]
    if col.size == 0:
        return 0.0
    return float(np.max(np.mean(col, axis=0)) / 255)


def _half_staff_score(binary, sys, bx, half_w=2):
    """Score a vertical line by the MIN density of the two halves of a staff.
    A real barline covers the full staff height → high min.
    A note stem covers only part → low min.
    """
    col = binary[sys[0]:sys[4], max(0, bx - half_w):bx + half_w + 1]
    if col.size == 0:
        return 0.0
    mid = col.shape[0] // 2
    upper = np.max(np.mean(col[:mid], axis=0)) / 255 if mid > 0 else 0.0
    lower = np.max(np.mean(col[mid:], axis=0)) / 255 if col.shape[0] - mid > 0 else 0.0
    return float(min(upper, lower))


def _merge_barlines(treble_bl, bass_bl, dy, binary=None,
                    treble_sys=None, bass_sys=None):
    """Merge and verify barlines from treble and bass staves."""
    from config import CFG
    bc = CFG.barline

    if not treble_bl and not bass_bl:
        return []

    # Step 1: Match treble/bass barlines pairwise
    match_tolerance = dy * bc.match_tolerance_dy
    t_sorted = sorted(treble_bl)
    b_sorted = sorted(bass_bl)
    t_used = [False] * len(t_sorted)
    b_used = [False] * len(b_sorted)
    matched, unmatched = [], []

    for ti, tx in enumerate(t_sorted):
        best_bi, best_dist = -1, match_tolerance
        for bi, bx in enumerate(b_sorted):
            if b_used[bi]:
                continue
            dist = abs(tx - bx)
            if dist < best_dist:
                best_dist = dist
                best_bi = bi
        if best_bi >= 0:
            matched.append(int((tx + b_sorted[best_bi]) / 2))
            t_used[ti] = True
            b_used[best_bi] = True
        else:
            unmatched.append(int(tx))
            t_used[ti] = True
    for bi, bx in enumerate(b_sorted):
        if not b_used[bi]:
            unmatched.append(int(bx))

    all_candidates = sorted(matched + unmatched)
    deduped = []
    for bx in all_candidates:
        if not deduped or abs(bx - deduped[-1]) > dy:
            deduped.append(bx)
        else:
            deduped[-1] = (deduped[-1] + bx) // 2

    if binary is None or treble_sys is None or bass_sys is None:
        return deduped

    # Step 2: Verify — search around each candidate using half-staff
    # scoring so full-height barlines beat partial note stems.
    base_radius = int(dy * bc.verify_base_radius_dy)
    density_thr = bc.verify_density_threshold
    verified = set()
    for bx in deduped:
        # Dynamic radius: bridge any treble/bass misalignment
        t_dists = [abs(bx - tx) for tx in t_sorted] if t_sorted else [0]
        b_dists = [abs(bx - bxx) for bxx in b_sorted] if b_sorted else [0]
        max_dist = max(min(t_dists), min(b_dists))
        radius = max(base_radius, int(max_dist + dy * bc.verify_extra_radius_dy))

        best_x, best_score = None, 0.0
        for test_x in range(int(bx) - radius, int(bx) + radius + 1):
            if test_x < 2 or test_x >= binary.shape[1] - 2:
                continue
            d_t = _col_density(binary, treble_sys[0], treble_sys[4], test_x)
            d_b = _col_density(binary, bass_sys[0], bass_sys[4], test_x)
            if d_t > density_thr and d_b > density_thr:
                score = _half_staff_score(binary, treble_sys, test_x) + \
                        _half_staff_score(binary, bass_sys, test_x)
                if score > best_score:
                    best_score = score
                    best_x = test_x
        if best_x is not None:
            verified.add(best_x)

    # Step 3: Scan for missed barlines
    clef_end = int(binary.shape[1] * bc.clef_end_ratio)
    all_spacings = []
    for bl_list in [t_sorted, b_sorted]:
        if len(bl_list) >= 2:
            all_spacings.extend([bl_list[i+1]-bl_list[i] for i in range(len(bl_list)-1)])
    min_gap = np.median(all_spacings) * bc.min_gap_median_ratio if all_spacings else dy * 15

    for bx in range(clef_end, binary.shape[1], bc.scan_step_px):
        d_t = _col_density(binary, treble_sys[0], treble_sys[4], bx)
        d_b = _col_density(binary, bass_sys[0], bass_sys[4], bx)
        if d_t > density_thr and d_b > density_thr:
            if all(abs(bx - v) > min_gap for v in verified):
                verified.add(bx)

    # Step 4: Spacing filter — when two barlines are too close, keep the
    # one with higher half-staff quality (real barline > note stem).
    def _quality(bx):
        return _half_staff_score(binary, treble_sys, bx) + \
               _half_staff_score(binary, bass_sys, bx)

    if verified:
        sorted_v = sorted(verified)
        filtered = [sorted_v[0]]
        for bx in sorted_v[1:]:
            if bx - filtered[-1] > min_gap:
                filtered.append(bx)
            elif _quality(bx) > _quality(filtered[-1]):
                filtered[-1] = bx
        filtered = [bx for bx in filtered
                    if clef_end < bx < binary.shape[1] * bc.right_boundary_ratio]
        if len(filtered) >= 2:
            return filtered

    # Fallback
    filtered = [deduped[0]] if deduped else []
    for bx in deduped[1:]:
        if bx - filtered[-1] > min_gap:
            filtered.append(bx)
    filtered = [bx for bx in filtered
                if clef_end < bx < binary.shape[1] * bc.right_boundary_ratio]
    return filtered


def _validate_block_rest(music_symbols, rest, dy):
    """Verify a half/whole rest match actually contains a rest block.

    The stop_1/stop_2 templates include surrounding staff lines, so template
    matching can score high anywhere along the middle staff line. We confirm
    a real match by sampling the expected rest-block position in the
    staff-lines-removed image: a genuine block is an opaque rectangle, a
    false positive shows nothing.
    """
    # Block y-offset from match center (the template center aligns with the
    # staff middle line). stop_2 (half) sits on the middle line; stop_1
    # (whole) hangs above it, farther from center.
    if rest['type'] == 'stop_2.jpg':
        block_dy_offset = -0.15
    elif rest['type'] == 'stop_1.jpg':
        block_dy_offset = -0.55
    else:
        return True

    cx = int(rest['x'])
    cy = int(rest['y_center'] + block_dy_offset * dy)
    half_h = max(2, int(0.25 * dy))
    half_w = max(3, int(0.55 * dy))

    h, w = music_symbols.shape
    y1 = max(0, cy - half_h)
    y2 = min(h, cy + half_h)
    x1 = max(0, cx - half_w)
    x2 = min(w, cx + half_w)
    if y2 <= y1 or x2 <= x1:
        return False

    roi = music_symbols[y1:y2, x1:x2]
    fill = float((roi > 127).sum()) / float(roi.size)
    return fill >= 0.35


def _validate_small_rest(music_symbols, rest, dy):
    """Verify a quarter/eighth rest match has a real symbol body.

    stop_4/stop_8 templates include staff-line context, so they can
    fire on slur arcs, chord-area noise, or stray decorations. A
    genuine quarter or eighth rest has a substantial connected
    component (>=~half the symbol height squared) at the match
    center; spurious matches show only thin lines.
    """
    if rest.get('type') not in ('stop_4.jpg', 'stop_8.jpg'):
        return True
    cx = int(rest['x'])
    cy = int(rest['y_center'])
    half_w = max(3, int(0.5 * dy))
    half_h = max(3, int(0.6 * dy))
    h, w = music_symbols.shape
    y1 = max(0, cy - half_h); y2 = min(h, cy + half_h)
    x1 = max(0, cx - half_w); x2 = min(w, cx + half_w)
    if y2 <= y1 or x2 <= x1:
        return False
    roi = (music_symbols[y1:y2, x1:x2] > 127).astype(np.uint8)
    nl, _, st, _ = cv2.connectedComponentsWithStats(roi, connectivity=8)
    if nl <= 1:
        return False
    biggest = max(int(st[i, cv2.CC_STAT_AREA]) for i in range(1, nl))
    # Quarter rests need a substantial body (~0.18*dy²); eighth rests
    # are smaller (~0.10*dy²). Slur tails / accent marks come in
    # below the eighth threshold (typically <0.08*dy²).
    if rest['type'] == 'stop_4.jpg':
        return biggest >= int(dy * dy * 0.18)
    return biggest >= int(dy * dy * 0.10)


def _filter_rests(all_rests, barlines_per_pair, all_notes, dy, music_symbols=None):
    """Filter rests near barlines and overlapping with notes."""
    # Remove rests near barlines
    filtered = []
    for rest in all_rests:
        rest_pair_idx = rest['system_idx'] // 2
        if rest_pair_idx < len(barlines_per_pair):
            barlines = barlines_per_pair[rest_pair_idx]
            near_barline = any(abs(rest['x'] - bx) < dy * 3.5 for bx in barlines)
            if barlines and rest['x'] < barlines[0]:
                near_barline = True
            if near_barline:
                continue
        filtered.append(rest)

    # Remove rests overlapping with notes. Dense beamed runs (e.g. eight
    # 16ths in one measure) leave narrow gaps between noteheads where rest
    # templates can false-match; the wider x-proximity catches those.
    result = []
    for rest in filtered:
        rest_pair_idx = rest['system_idx'] // 2
        has_nearby = any(
            note.get('pair_idx', -1) == rest_pair_idx
            and abs(rest['x'] - note['x']) < dy * 2.5
            and abs(rest['y_center'] - note['y_center']) < dy * 2.5
            for note in all_notes
        )
        if not has_nearby:
            result.append(rest)

    # Block-rest validation: stop_1/stop_2 templates include staff lines and
    # match along the middle line even without a block. Verify on the
    # staff-lines-removed image.
    if music_symbols is not None:
        result = [r for r in result if _validate_block_rest(music_symbols, r, dy)]

    return result


def _split_rests_by_clef(all_rests, grand_staff_pairs):
    """Split rests into treble/bass lists."""
    treble_rests = []
    bass_rests = []
    for rest in all_rests:
        sys_idx = rest['system_idx']
        pair_idx = sys_idx // 2
        if pair_idx < len(grand_staff_pairs):
            rest['pair_idx'] = pair_idx
            if sys_idx % 2 == 0:
                treble_rests.append(rest)
            else:
                bass_rests.append(rest)
    return treble_rests, bass_rests


def _print_and_save_output(pair_data, accidentals_map, dy):
    """Format, print, and save the Jianpu output."""
    all_treble_lines = []
    all_bass_lines = []

    for pd in pair_data:
        t_line = format_output(pd['treble_measures'], accidentals_map, dy=dy)
        b_line = format_output(pd['bass_measures'], accidentals_map, dy=dy)
        all_treble_lines.append(t_line)
        all_bass_lines.append(b_line)

    print(f"\n{'=' * 60}")
    print("JIANPU OUTPUT")
    print(f"{'=' * 60}")

    for i, pd in enumerate(pair_data):
        print(f"\n--- System {i + 1} ---")
        print(f"  Treble: {all_treble_lines[i]}")
        print(f"  Bass:   {all_bass_lines[i]}")

    print(f"\n{'=' * 60}")
    print("\n高音部分 (Treble):")
    for line in all_treble_lines:
        print(line)
    print("\n低音部分 (Bass):")
    for line in all_bass_lines:
        print(line)

    # Save to file
    with open("output_jianpu.txt", "w", encoding="utf-8") as f:
        f.write("简谱翻译结果\n")
        f.write("=" * 60 + "\n\n")
        for i in range(len(pair_data)):
            f.write(f"--- 第{i + 1}行 ---\n")
            f.write(f"高音: {all_treble_lines[i]}\n")
            f.write(f"低音: {all_bass_lines[i]}\n\n")
        f.write("\n" + "=" * 60 + "\n")
        f.write("高音部分:\n")
        for line in all_treble_lines:
            f.write(line + "\n")
        f.write("\n低音部分:\n")
        for line in all_bass_lines:
            f.write(line + "\n")
    print("\n   Saved text: output_jianpu.txt")

    # Confidence report
    from confidence import format_confidence_report
    from config import CFG
    staff_measures = []
    staff_anchors = []
    for pi, pd in enumerate(pair_data):
        staff_measures.append((f"第{pi + 1}行 高音", pd['treble_measures']))
        staff_measures.append((f"第{pi + 1}行 低音", pd['bass_measures']))
        staff_anchors.append(pd.get('timesig_anchors'))
        staff_anchors.append(pd.get('timesig_anchors'))
    report = format_confidence_report(staff_measures, accidentals_map,
                                      beats_per_measure=CFG.duration.beats_per_measure,
                                      dy=dy,
                                      staff_anchors=staff_anchors)
    with open("output_confidence.txt", "w", encoding="utf-8") as f:
        f.write(report)
    print("   Saved confidence: output_confidence.txt")


# ============================================================
# Visualization
# ============================================================

def _generate_annotated_image(image_path, pair_data, accidentals_map,
                              treble_notes, bass_notes, grand_staff_pairs,
                              barlines_per_pair, dy):
    """Generate original staff with jianpu annotation rows below each grand staff."""
    img_orig = cv2.imread(image_path)
    if img_orig is None:
        print("   Error: Could not read image for annotation!")
        return

    orig_h, orig_w = img_orig.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    annotation_height = int(dy * 3.5)
    num_pairs = len(pair_data)

    new_h = orig_h + num_pairs * annotation_height
    img_out = np.ones((new_h, orig_w, 3), dtype=np.uint8) * 255

    y_offset = 0
    prev_copy_end = 0

    for pi, pd in enumerate(pair_data):
        bass_sys = pd['bass_sys']
        barlines = pd['barlines']

        cut_y = bass_sys[4] + int(dy * 2)
        if pi + 1 < num_pairs:
            next_treble_top = pair_data[pi + 1]['treble_sys'][0]
            cut_y = min(cut_y, next_treble_top - int(dy * 1))
        else:
            cut_y = min(cut_y, orig_h)

        strip = img_orig[prev_copy_end:cut_y, :]
        img_out[y_offset:y_offset + strip.shape[0], :strip.shape[1]] = strip
        y_offset += strip.shape[0]

        ann_y1 = y_offset
        ann_y2 = y_offset + annotation_height
        img_out[ann_y1:ann_y2, :] = (245, 245, 255)

        cv2.line(img_out, (0, ann_y1), (orig_w, ann_y1), (180, 180, 180), 1)
        cv2.line(img_out, (0, ann_y2 - 1), (orig_w, ann_y2 - 1), (180, 180, 180), 1)

        treble_measures = pd['treble_measures']
        bass_measures = pd['bass_measures']
        boundaries = [0] + list(barlines) + [orig_w]

        treble_text_y = ann_y1 + int(annotation_height * 0.35)
        bass_text_y = ann_y1 + int(annotation_height * 0.75)

        cv2.putText(img_out, "T:", (5, treble_text_y), font, 0.35,
                    (180, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img_out, "B:", (5, bass_text_y), font, 0.35,
                    (0, 130, 0), 1, cv2.LINE_AA)

        font_scale = 0.38
        thickness = 1

        for mi in range(max(len(treble_measures), len(bass_measures))):
            if mi >= len(boundaries) - 1:
                break
            left = boundaries[mi]
            right = boundaries[mi + 1]
            region_w = right - left

            if mi > 0:
                cv2.line(img_out, (left, ann_y1 + 3), (left, ann_y2 - 3), (150, 150, 150), 1)

            for measures, text_y, color in [
                (treble_measures, treble_text_y, (200, 0, 0)),
                (bass_measures, bass_text_y, (0, 140, 0)),
            ]:
                if mi < len(measures):
                    text = format_measure(measures[mi], accidentals_map,
                                          measure_idx=mi, dy=dy)
                    if text:
                        scale = font_scale
                        (tw, _), _ = cv2.getTextSize(text, font, scale, thickness)
                        if tw > region_w - 10:
                            scale = max(0.22, scale * (region_w - 10) / tw)
                        (tw, _), _ = cv2.getTextSize(text, font, scale, thickness)
                        tx = left + max(3, (region_w - tw) // 2)
                        cv2.putText(img_out, text, (tx, text_y), font, scale,
                                    color, thickness, cv2.LINE_AA)

        y_offset += annotation_height
        prev_copy_end = cut_y

    if prev_copy_end < orig_h:
        remaining = img_orig[prev_copy_end:, :]
        end = min(y_offset + remaining.shape[0], img_out.shape[0])
        h_to_copy = end - y_offset
        img_out[y_offset:end, :remaining.shape[1]] = remaining[:h_to_copy]

    cv2.imwrite("output_jianpu_on_staff.png", img_out)
    print("   Saved: output_jianpu_on_staff.png")


def _generate_jianpu_only_image(pair_data, accidentals_map, dy):
    """Generate clean jianpu-only image."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        _generate_jianpu_pil(pair_data, accidentals_map, dy)
    except ImportError:
        _generate_jianpu_cv2(pair_data, accidentals_map, dy)


def _generate_jianpu_pil(pair_data, accidentals_map, dy):
    """Generate jianpu-only image using PIL."""
    from PIL import Image, ImageDraw, ImageFont

    pair_formatted = []
    for pd in pair_data:
        t_parts = [format_measure(m, accidentals_map, measure_idx=mi, dy=dy)
                   for mi, m in enumerate(pd['treble_measures'])]
        b_parts = [format_measure(m, accidentals_map, measure_idx=mi, dy=dy)
                   for mi, m in enumerate(pd['bass_measures'])]
        pair_formatted.append((t_parts, b_parts))

    font_size = 22
    title_font_size = 28
    label_font_size = 18

    # Chinese-capable fonts for titles/labels
    cjk_candidates = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
    ]
    # Monospace fonts for notation (ASCII only)
    mono_candidates = [
        "C:/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/cour.ttf",
    ]

    def _load_font(candidates, size):
        for fp in candidates:
            if os.path.exists(fp):
                try:
                    return ImageFont.truetype(fp, size)
                except Exception:
                    continue
        return None

    # Notation font: prefer monospace, fallback to CJK
    font = _load_font(mono_candidates, font_size) or _load_font(cjk_candidates, font_size)
    # Title/label fonts: must support Chinese
    title_font = _load_font(cjk_candidates, title_font_size)
    label_font = _load_font(cjk_candidates, label_font_size)

    if font is None:
        font = ImageFont.load_default()
    if title_font is None:
        title_font = font
    if label_font is None:
        label_font = font

    max_img_w = 1200
    padding = 40
    line_height = font_size + 14
    section_gap = 15
    measure_sep = " | "
    content_w = max_img_w - 2 * padding - 60

    display_lines = [("title", "简谱 (Jianpu)"), ("blank", "")]

    for i, (t_parts, b_parts) in enumerate(pair_formatted):
        display_lines.append(("section", f"── 第{i + 1}行 ──"))

        def split_measures(parts, label):
            lines_out = []
            current_parts = []
            for part in parts:
                test_text = label + "| " + measure_sep.join(current_parts + [part]) + " |"
                bbox = font.getbbox(test_text) if hasattr(font, 'getbbox') else (0, 0, len(test_text) * font_size * 0.55, font_size)
                text_w = bbox[2] - bbox[0]
                if text_w > content_w and current_parts:
                    lines_out.append(label + "| " + measure_sep.join(current_parts) + " |")
                    current_parts = [part]
                else:
                    current_parts.append(part)
            if current_parts:
                prefix = label if not lines_out else " " * len(label)
                lines_out.append(prefix + "| " + measure_sep.join(current_parts) + " |")
            return lines_out

        for tl in split_measures(t_parts, "高音: "):
            display_lines.append(("treble", tl))
        for bl in split_measures(b_parts, "低音: "):
            display_lines.append(("bass", bl))
        display_lines.append(("blank", ""))

    max_width = 200
    for kind, text in display_lines:
        if text:
            f = title_font if kind == "title" else font
            bbox = f.getbbox(text) if hasattr(f, 'getbbox') else (0, 0, len(text) * font_size * 0.6, font_size)
            max_width = max(max_width, bbox[2] - bbox[0])

    img_w = int(max_width + 2 * padding)
    img_h = int(len(display_lines) * line_height + 2 * padding + 40)

    img = Image.new('RGB', (img_w, img_h), 'white')
    draw = ImageDraw.Draw(img)

    def _draw_mixed(draw, x, y, text, color, cjk_font, mono_font):
        """Draw text with CJK font for Chinese chars, mono font for ASCII."""
        # Split into Chinese prefix and ASCII notation
        # Patterns: "高音: |...|" or "低音: |...|" or "      |...|"
        split_idx = 0
        for i, ch in enumerate(text):
            if ch == '|' or (ch == ' ' and i > 0 and text[:i].strip() and
                             not any('\u4e00' <= c <= '\u9fff' for c in text[i:])):
                split_idx = i
                break
            if ord(ch) > 127:
                continue
            if i > 0 and all(ord(c) <= 127 for c in text[i:]):
                split_idx = i
                break

        if split_idx > 0 and any('\u4e00' <= c <= '\u9fff' for c in text[:split_idx]):
            prefix = text[:split_idx]
            suffix = text[split_idx:]
            draw.text((x, y), prefix, fill=color, font=cjk_font)
            bbox = cjk_font.getbbox(prefix) if hasattr(cjk_font, 'getbbox') else (0, 0, len(prefix) * 14, 22)
            prefix_w = bbox[2] - bbox[0]
            draw.text((x + prefix_w, y), suffix, fill=color, font=mono_font)
        else:
            draw.text((x, y), text, fill=color, font=mono_font)

    y = padding
    for kind, text in display_lines:
        if kind == "title":
            draw.text((padding, y), text, fill=(0, 0, 0), font=title_font)
            y += title_font_size + 15
        elif kind == "section":
            draw.line([(padding, y), (img_w - padding, y)], fill=(200, 200, 200), width=1)
            y += 5
            draw.text((padding, y), text, fill=(100, 100, 100), font=label_font)
            y += label_font_size + 8
        elif kind == "treble":
            _draw_mixed(draw, padding, y, text, (180, 0, 0), label_font, font)
            y += line_height
        elif kind == "bass":
            _draw_mixed(draw, padding, y, text, (0, 120, 0), label_font, font)
            y += line_height
        elif kind == "blank":
            y += section_gap

    img = img.crop((0, 0, img_w, min(y + padding, img_h)))
    img.save("output_jianpu_clean.png")
    print("   Saved: output_jianpu_clean.png")


def _generate_jianpu_cv2(pair_data, accidentals_map, dy):
    """Fallback: generate jianpu image using OpenCV."""
    lines = []
    for i, pd in enumerate(pair_data):
        t_parts = [format_measure(m, accidentals_map, measure_idx=mi, dy=dy)
                   for mi, m in enumerate(pd['treble_measures'])]
        b_parts = [format_measure(m, accidentals_map, measure_idx=mi, dy=dy)
                   for mi, m in enumerate(pd['bass_measures'])]

        t_line = "| " + " | ".join(t_parts) + " |"
        b_line = "| " + " | ".join(b_parts) + " |"

        lines.append((f"System {i + 1}", (100, 100, 100)))
        lines.append((f"  T: {t_line}", (180, 0, 0)))
        lines.append((f"  B: {b_line}", (0, 140, 0)))
        lines.append(("", (0, 0, 0)))

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 1
    line_height = 22
    pad = 20

    max_w = max((cv2.getTextSize(t, font, font_scale, thickness)[0][0]
                 for t, _ in lines if t), default=100)

    img_w = max_w + 2 * pad
    img_h = len(lines) * line_height + 2 * pad + 40

    img = np.ones((img_h, img_w, 3), dtype=np.uint8) * 255
    cv2.putText(img, "Jianpu Output", (pad, pad + 15),
                font, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

    y = pad + 40
    for text, color in lines:
        if text:
            cv2.putText(img, text, (pad, y), font, font_scale,
                        (color[2], color[1], color[0]), thickness, cv2.LINE_AA)
        y += line_height

    cv2.imwrite("output_jianpu_clean.png", img)
    print("   Saved: output_jianpu_clean.png")


if __name__ == "__main__":
    import sys
    img_path = sys.argv[1] if len(sys.argv) > 1 else "../input_page1.png"
    # Optional: --bpm N to override beats_per_measure
    if '--bpm' in sys.argv:
        idx = sys.argv.index('--bpm')
        if idx + 1 < len(sys.argv):
            from config import CFG
            CFG.duration.beats_per_measure = float(sys.argv[idx + 1])
            print(f"Override beats_per_measure = {CFG.duration.beats_per_measure}")
    main(img_path)
