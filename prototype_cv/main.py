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

from staff_removal import extract_staff_lines
from pitch_detection import get_staff_systems, pair_grand_staves
from template_matching import find_noteheads
from symbol_detection import (
    detect_barlines,
    detect_accidentals_global,
    assign_accidentals_to_notes,
    detect_rests,
)
from note_assignment import assign_notes_to_staves, filter_false_positive_notes
from stem_tracking import track_stem
from note_unit import build_note_units, segment_into_measures, merge_overlapping_note_units
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

    grand_staff_pairs = pair_grand_staves(systems)
    print(f"   Paired into {len(grand_staff_pairs)} grand staff systems (treble+bass)")

    if not grand_staff_pairs:
        print("Error: Could not pair staves into grand staff systems!")
        return

    # Calculate average dy (staff line spacing)
    dy = np.mean([(sys[4] - sys[0]) / 4.0 for sys in systems])
    print(f"   Average staff line spacing (dy): {dy:.1f}px")

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
    clef_area_x_first = int(img_w * 0.17)

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
    global_accs = detect_accidentals_global(binary, systems, dy)
    all_detected_notes = treble_notes + bass_notes
    accidentals_map = assign_accidentals_to_notes(global_accs, all_detected_notes, dy)
    n_sharps = sum(1 for v in accidentals_map.values() if v == '#')
    n_flats = sum(1 for v in accidentals_map.values() if v == 'b')
    print(f"   Total accidentals: {len(accidentals_map)} ({n_sharps} sharps, {n_flats} flats)")

    # ── 7. Detect Rests ──
    print("7. Detecting rests...")
    all_rests = detect_rests(binary, systems, dy)
    all_rests = [r for r in all_rests if r['x'] > int(img_w * 0.10)]
    all_rests = _filter_rests(all_rests, barlines_per_pair, treble_notes + bass_notes, dy)
    print(f"   Found {len(all_rests)} rests (after filtering)")

    treble_rests, bass_rests = _split_rests_by_clef(all_rests, grand_staff_pairs)

    # ── 8. Track Stems ──
    print("8. Tracking stems...")
    for note in treble_notes + bass_notes:
        note['stem'] = track_stem(music_symbols, note, dy)
    print(f"   Tracked stems for {len(treble_notes) + len(bass_notes)} notes")

    # ── 9. Group into Chords, Segment Measures, Estimate Durations ──
    print("9. Grouping notes and formatting output...")
    pair_data = []

    for pair_idx, (treble_sys, bass_sys) in enumerate(grand_staff_pairs):
        pair_treble = [n for n in treble_notes if n.get('pair_idx') == pair_idx]
        pair_bass = [n for n in bass_notes if n.get('pair_idx') == pair_idx]
        pair_t_rests = [r for r in treble_rests if r.get('pair_idx') == pair_idx]
        pair_b_rests = [r for r in bass_rests if r.get('pair_idx') == pair_idx]

        barlines = barlines_per_pair[pair_idx] if pair_idx < len(barlines_per_pair) else []

        treble_units = build_note_units(pair_treble, music_symbols, binary, dy)
        bass_units = build_note_units(pair_bass, music_symbols, binary, dy)

        # TODO: Merge notes whose durations overlap (two-voice alignment)
        # Currently disabled — beam/flag detection returns unreliable per-note
        # durations (1 beam instead of 2 for sixteenths, stem_dir=None for some
        # notes). Enable once beam detection is fixed.
        # treble_units = merge_overlapping_note_units(treble_units, beats_per_measure=2.0, dy=dy)
        # bass_units = merge_overlapping_note_units(bass_units, beats_per_measure=2.0, dy=dy)

        is_first = (pair_idx == 0)
        treble_measures = segment_into_measures(treble_units, pair_t_rests, barlines, dy,
                                                beats_per_measure=2.0,
                                                is_first_system=is_first)
        bass_measures = segment_into_measures(bass_units, pair_b_rests, barlines, dy,
                                              beats_per_measure=2.0,
                                              is_first_system=is_first)

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
# Helper functions
# ============================================================

def _merge_barlines(treble_bl, bass_bl, dy, binary=None,
                    treble_sys=None, bass_sys=None):
    """Merge and verify barlines from treble and bass staves.

    Strategy:
    1. Match treble/bass barlines pairwise within tolerance
    2. Use average position for matched pairs
    3. Verify candidates with _is_real_barline (with ±5px search)
    4. Scan for missed barlines with minimum spacing constraint
    5. Fallback to averaged candidates if verification fails
    """
    if not treble_bl and not bass_bl:
        return []

    # Step 1: Match treble and bass barlines pairwise
    match_tolerance = dy * 8  # ~170px
    t_sorted = sorted(treble_bl)
    b_sorted = sorted(bass_bl)
    t_used = [False] * len(t_sorted)
    b_used = [False] * len(b_sorted)
    matched = []
    unmatched = []

    for ti, tx in enumerate(t_sorted):
        best_bi = -1
        best_dist = match_tolerance
        for bi, bx in enumerate(b_sorted):
            if b_used[bi]:
                continue
            dist = abs(tx - bx)
            if dist < best_dist:
                best_dist = dist
                best_bi = bi
        if best_bi >= 0:
            avg_x = int((tx + b_sorted[best_bi]) / 2)
            matched.append(avg_x)
            t_used[ti] = True
            b_used[best_bi] = True
        else:
            unmatched.append(int(tx))
            t_used[ti] = True

    for bi, bx in enumerate(b_sorted):
        if not b_used[bi]:
            unmatched.append(int(bx))

    # Matched pairs are high confidence; unmatched are low confidence
    all_candidates = sorted(matched + unmatched)

    # Deduplicate with dy tolerance
    deduped = []
    for bx in all_candidates:
        if not deduped or abs(bx - deduped[-1]) > dy:
            deduped.append(bx)
        else:
            deduped[-1] = (deduped[-1] + bx) // 2

    if binary is None or treble_sys is None or bass_sys is None:
        return deduped

    # Step 2: Verify candidates — search in wider range around each
    # The averaged position may be far from the actual barline (e.g., treble at 1154,
    # bass at 1297, average 1225 — actual barline is at neither position).
    # Search in a range proportional to the match tolerance.
    search_radius = int(dy * 4)  # ~85px
    verified = set()
    for bx in deduped:
        best_x = None
        best_density = 0
        for test_x in range(int(bx) - search_radius, int(bx) + search_radius + 1):
            if test_x < 2 or test_x >= binary.shape[1] - 2:
                continue
            col_t = binary[treble_sys[0]:treble_sys[4], test_x-1:test_x+2]
            col_b = binary[bass_sys[0]:bass_sys[4], test_x-1:test_x+2]
            d_t = np.max(np.mean(col_t, axis=0)) / 255 if col_t.size > 0 else 0
            d_b = np.max(np.mean(col_b, axis=0)) / 255 if col_b.size > 0 else 0
            if d_t > 0.55 and d_b > 0.55:
                total = d_t + d_b
                if total > best_density:
                    best_density = total
                    best_x = test_x
        if best_x is not None:
            verified.add(best_x)

    # Step 3: Scan for missed barlines with minimum spacing
    clef_end = int(binary.shape[1] * 0.25)
    # Minimum spacing: use individual system spacings as reference
    all_spacings = []
    for bl_list in [t_sorted, b_sorted]:
        if len(bl_list) >= 2:
            all_spacings.extend([bl_list[i+1]-bl_list[i] for i in range(len(bl_list)-1)])
    min_gap = np.median(all_spacings) * 0.5 if all_spacings else dy * 15

    for bx in range(clef_end, binary.shape[1], 3):
        if _is_real_barline(binary, bx, treble_sys, bass_sys):
            if all(abs(bx - v) > min_gap for v in verified):
                verified.add(bx)

    # Step 4: Apply minimum spacing filter to remove false positives
    if verified:
        sorted_v = sorted(verified)
        filtered = [sorted_v[0]]
        for bx in sorted_v[1:]:
            if bx - filtered[-1] > min_gap:
                filtered.append(bx)
        # Remove barlines too close to left edge or right edge
        filtered = [bx for bx in filtered
                    if clef_end < bx < binary.shape[1] * 0.96]
        if len(filtered) >= 2:
            return filtered

    # Fallback: use averaged candidates with spacing filter
    filtered = [deduped[0]] if deduped else []
    for bx in deduped[1:]:
        if bx - filtered[-1] > min_gap:
            filtered.append(bx)
    filtered = [bx for bx in filtered
                if clef_end < bx < binary.shape[1] * 0.96]
    return filtered


def _is_real_barline(binary, bx, treble_sys, bass_sys):
    """Check if position bx has a real barline present on both staves.

    In piano grand staff notation, regular measure barlines span each
    staff individually but do NOT always cross the gap between staves.
    Only system barlines cross the gap. So we require high density on
    BOTH staves but not necessarily in the gap.
    """
    if bx < 2 or bx >= binary.shape[1] - 2:
        return False
    # Check a 5-pixel-wide column (±2) for more robust detection
    col_t = binary[treble_sys[0]:treble_sys[4], bx-2:bx+3]
    col_b = binary[bass_sys[0]:bass_sys[4], bx-2:bx+3]
    # Use max across columns (barline might be only 1-2px wide)
    d_t = np.max(np.mean(col_t, axis=0)) / 255 if col_t.size > 0 else 0
    d_b = np.max(np.mean(col_b, axis=0)) / 255 if col_b.size > 0 else 0
    return d_t > 0.55 and d_b > 0.55


def _filter_rests(all_rests, barlines_per_pair, all_notes, dy):
    """Filter rests near barlines and overlapping with notes."""
    # Remove rests near barlines
    filtered = []
    for rest in all_rests:
        rest_pair_idx = rest['system_idx'] // 2
        if rest_pair_idx < len(barlines_per_pair):
            barlines = barlines_per_pair[rest_pair_idx]
            near_barline = any(abs(rest['x'] - bx) < dy * 3.0 for bx in barlines)
            if barlines and rest['x'] < barlines[0]:
                near_barline = True
            if near_barline:
                continue
        filtered.append(rest)

    # Remove rests overlapping with notes
    result = []
    for rest in filtered:
        rest_pair_idx = rest['system_idx'] // 2
        has_nearby = any(
            note.get('pair_idx', -1) == rest_pair_idx
            and abs(rest['x'] - note['x']) < dy * 1.5
            and abs(rest['y_center'] - note['y_center']) < dy * 2.0
            for note in all_notes
        )
        if not has_nearby:
            result.append(rest)

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
        barline_xs = pd['barlines']
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
        barline_xs = pd['barlines']
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
    main(img_path)
