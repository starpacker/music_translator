import cv2
import numpy as np
import os
import glob

# --- Path to the provided symbol templates ---
PICTURE_DIR = os.path.join(os.path.dirname(__file__),
    "..", "repo", "translate-staff-to-simple-musical-notation-master",
    "score_recognition_v4", "picture")
PICTURE_EXPAND_DIR = os.path.join(os.path.dirname(__file__),
    "..", "repo", "translate-staff-to-simple-musical-notation-master",
    "score_recognition_v4", "picture_expand")
TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "..", "template")

# Names of non-note symbols that should be EXCLUDED
EXCLUSION_TEMPLATES = [
    "treble_clef.jpg",
    "bass_clef.jpg",
    "sharp_1.jpg",
    "flat_1.jpg",
    "natural_1.jpg",
    "ornament.jpg",
]


def _load_exclusion_templates():
    """Load all clef/accidental template images for exclusion matching."""
    templates = []
    for name in EXCLUSION_TEMPLATES:
        path = os.path.join(PICTURE_DIR, name)
        if os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # Binarize: make symbol white on black background
                _, bimg = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
                templates.append((name, bimg))
    return templates


def find_exclusion_zones(binary_img, staff_systems, dy):
    """
    Use multi-scale template matching with clef/accidental templates to find
    rectangular regions that should NOT be treated as noteheads.
    
    Returns a list of (x1, y1, x2, y2) exclusion rectangles.
    """
    exclusion_zones = []
    raw_templates = _load_exclusion_templates()
    if not raw_templates:
        return exclusion_zones

    img_h, img_w = binary_img.shape

    for system in staff_systems:
        staff_height = system[4] - system[0]
        
        # For each staff system, define the search region (staff + some margin)
        margin = int(dy * 2)
        search_y1 = max(0, system[0] - margin)
        search_y2 = min(img_h, system[4] + margin)
        search_roi = binary_img[search_y1:search_y2, :]
        
        # Only search the left portion of each staff (clefs/key signatures are at the start)
        # This dramatically reduces search area and false positives
        search_x_end = min(img_w, int(img_w * 0.18))
        search_roi = search_roi[:, :search_x_end]
        
        roi_h, roi_w = search_roi.shape
        if roi_h < 10 or roi_w < 10:
            continue

        for tname, timg in raw_templates:
            th_orig, tw_orig = timg.shape
            
            # Calculate the ideal scale: template should match the staff height
            ideal_scale = staff_height / float(th_orig) if th_orig > 0 else 0.5
            
            # Try a few scales around the ideal
            for scale_factor in [ideal_scale * 0.7, ideal_scale * 0.85, ideal_scale, ideal_scale * 1.15, ideal_scale * 1.3]:
                new_h = int(th_orig * scale_factor)
                new_w = int(tw_orig * scale_factor)
                
                if new_h < 10 or new_w < 10:
                    continue
                if new_h >= roi_h or new_w >= roi_w:
                    continue
                
                resized = cv2.resize(timg, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
                res = cv2.matchTemplate(search_roi, resized, cv2.TM_CCOEFF_NORMED)
                
                # Use a relatively high threshold to avoid false exclusions
                threshold = 0.55
                loc = np.where(res >= threshold)
                
                for pt in zip(*loc[::-1]):
                    x1 = pt[0]
                    y1 = search_y1 + pt[1]
                    x2 = x1 + new_w
                    y2 = y1 + new_h
                    score = res[pt[1], pt[0]]
                    exclusion_zones.append((x1, y1, x2, y2, score, tname))
    
    # NMS on exclusion zones to remove duplicates
    if exclusion_zones:
        exclusion_zones = _nms_exclusion_zones(exclusion_zones, 0.3)
    
    return exclusion_zones


def _nms_exclusion_zones(zones, overlap_thresh):
    """NMS for exclusion zones."""
    if not zones:
        return []
    
    boxes = np.array([(z[0], z[1], z[2], z[3], z[4]) for z in zones])
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = boxes[:, 4]
    
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores)[::-1]
    
    pick = []
    while len(idxs) > 0:
        i = idxs[0]
        pick.append(i)
        
        xx1 = np.maximum(x1[i], x1[idxs[1:]])
        yy1 = np.maximum(y1[i], y1[idxs[1:]])
        xx2 = np.minimum(x2[i], x2[idxs[1:]])
        yy2 = np.minimum(y2[i], y2[idxs[1:]])
        
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        
        overlap = (w * h) / np.minimum(area[i], area[idxs[1:]])
        
        idxs = np.delete(idxs, np.concatenate(([0], np.where(overlap > overlap_thresh)[0] + 1)))
    
    return [zones[i] for i in pick]


def is_in_exclusion_zone(cx, cy, exclusion_zones, margin=0):
    """Check if a point (cx, cy) falls inside any exclusion zone."""
    for (x1, y1, x2, y2, *_rest) in exclusion_zones:
        if x1 - margin <= cx <= x2 + margin and y1 - margin <= cy <= y2 + margin:
            return True
    return False


def create_notehead_template(dy):
    """
    Synthesize a solid notehead template based on the staff line gap (dy).
    """
    w = int(dy * 1.3)
    h = int(dy * 0.85)
    if w < 2: w = 2
    if h < 2: h = 2
    size = int(dy * 2)
    if size < 5: size = 5
    template = np.zeros((size, size), dtype=np.uint8)
    center = (size // 2, size // 2)
    cv2.ellipse(template, center, (w // 2, h // 2), -20, 0, 360, 255, -1)
    x, y, bw, bh = cv2.boundingRect(template)
    return template[y:y+bh, x:x+bw]


def create_hollow_notehead_template(dy):
    """
    Synthesize a hollow (open) notehead template for whole/half notes.
    An unfilled ellipse with thicker outline.
    """
    w = int(dy * 1.4)
    h = int(dy * 0.9)
    if w < 3: w = 3
    if h < 3: h = 3
    size = int(dy * 2.5)
    if size < 7: size = 7
    template = np.zeros((size, size), dtype=np.uint8)
    center = (size // 2, size // 2)
    thickness = max(2, int(dy * 0.15))
    cv2.ellipse(template, center, (w // 2, h // 2), -20, 0, 360, 255, thickness)
    x, y, bw, bh = cv2.boundingRect(template)
    return template[y:y+bh, x:x+bw]


def create_ledger_notehead_templates(dy):
    """
    Synthesize multiple notehead templates with ledger lines.
    """
    ew = int(dy * 1.3)
    eh = int(dy * 0.85)
    line_thickness = max(1, int(dy * 0.12))
    line_w = int(dy * 1.8)
    step = int(dy)
    half_step = int(dy / 2.0)
    
    templates = []
    
    def make_template(line_offsets_from_center):
        min_off = min(line_offsets_from_center) if line_offsets_from_center else 0
        max_off = max(line_offsets_from_center) if line_offsets_from_center else 0
        pad_top = max(0, -min_off) + eh
        pad_bot = max(0, max_off) + eh
        canvas_h = pad_top + pad_bot + eh * 2
        canvas_w = line_w + 10
        template = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        cx = canvas_w // 2
        cy = pad_top + eh // 2
        cv2.ellipse(template, (cx, cy), (ew // 2, eh // 2), -20, 0, 360, 255, -1)
        for off in line_offsets_from_center:
            ly = cy + off
            cv2.line(template, (cx - line_w // 2, ly), (cx + line_w // 2, ly), 255, line_thickness)
        x, y, bw, bh = cv2.boundingRect(template)
        if bw > 0 and bh > 0:
            return template[y:y+bh, x:x+bw]
        return None
    
    for offsets in [
        [0], [-half_step], [half_step],
        [0, -step], [0, step],
        [-half_step, -half_step - step],
        [half_step, half_step + step],
        [0, -step, -2*step],
    ]:
        t = make_template(offsets)
        if t is not None:
            templates.append(t)
    
    return templates


def non_max_suppression(boxes, overlapThresh):
    """Standard NMS to remove overlapping bounding boxes. boxes: list of (x, y, w, h, score)"""
    if len(boxes) == 0:
        return []
    boxes_np = np.array(boxes)
    x1 = boxes_np[:, 0]
    y1 = boxes_np[:, 1]
    x2 = boxes_np[:, 0] + boxes_np[:, 2]
    y2 = boxes_np[:, 1] + boxes_np[:, 3]
    scores = boxes_np[:, 4]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores)[::-1]
    pick = []
    while len(idxs) > 0:
        i = idxs[0]
        pick.append(i)
        xx1 = np.maximum(x1[i], x1[idxs[1:]])
        yy1 = np.maximum(y1[i], y1[idxs[1:]])
        xx2 = np.minimum(x2[i], x2[idxs[1:]])
        yy2 = np.minimum(y2[i], y2[idxs[1:]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[1:]]
        idxs = np.delete(idxs, np.concatenate(([0], np.where(overlap > overlapThresh)[0] + 1)))
    return [boxes[i] for i in pick]


def _deduplicate_by_center(boxes, min_dist):
    """
    Remove duplicate detections that are too close to each other (center-distance based).
    Keep the one with the higher score.
    """
    if not boxes:
        return []
    # Sort by score descending
    sorted_boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
    kept = []
    kept_centers = []
    for box in sorted_boxes:
        cx = box[0] + box[2] / 2.0
        cy = box[1] + box[3] / 2.0
        too_close = False
        for (pcx, pcy) in kept_centers:
            if abs(cx - pcx) < min_dist and abs(cy - pcy) < min_dist:
                too_close = True
                break
        if not too_close:
            kept.append(box)
            kept_centers.append((cx, cy))
    return kept


def find_noteheads(binary_img, dy, threshold=0.55, staff_systems=None, music_symbols=None,
                   detect_hollow=False):
    """
    Find solid noteheads using a hybrid approach:
    1. Find exclusion zones (clefs, accidentals) using provided templates.
    2. Morphological opening to find solid blobs (on-staff notes).
    3. Template matching in ledger-line zones AND with real notehead templates.
    4. Filter out detections that fall in exclusion zones.
    5. Deduplicate close detections (chord handling).
    
    music_symbols: binary image with staff lines removed (not used for centroid - too aggressive)
    """
    nh = int(dy * 0.85)
    nw = int(dy * 1.3)
    
    # --- Stage 0: Find exclusion zones (clefs, accidentals) ---
    exclusion_zones = []
    if staff_systems:
        print("   Finding exclusion zones (clefs, accidentals)...")
        exclusion_zones = find_exclusion_zones(binary_img, staff_systems, dy)
        print(f"   Found {len(exclusion_zones)} exclusion zones")
    
    # --- Stage 1: Morphological detection ---
    r = int(dy * 0.65)
    if r < 1: r = 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r, r))
    solid_blobs = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel, iterations=1)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(solid_blobs, connectivity=8)
    
    all_boxes = []
    
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        
        # Use actual centroid from connected components (more accurate than bbox center)
        cx = int(centroids[i][0])
        cy_blob = int(centroids[i][1])
        
        # Skip if in exclusion zone
        if is_in_exclusion_zone(cx, cy_blob, exclusion_zones, margin=int(dy * 0.3)):
            continue
        
        # Filter by size: exclude tiny noise and large objects
        if area < (dy * dy * 0.2) or area > (dy * dy * 6.0):
            continue
        
        aspect = float(w) / h if h > 0 else 0
        solidity = float(area) / (w * h) if (w * h) > 0 else 0
        
        # For tall blobs (potential chords), try to split them
        if h > 1.6 * dy and w < 2.5 * dy:
            # Use original binary image within expanded blob region for better centroid accuracy
            roi = solid_blobs[y:y+h, x:x+w]
            proj = np.sum(roi, axis=1).astype(float)
            
            kernel_smooth = np.ones(3) / 3.0
            proj_smooth = np.convolve(proj, kernel_smooth, mode='same')
            
            # Find peaks with improved clustering
            # First, find all local maxima
            raw_peaks = []
            for row in range(2, h - 2):
                local_max = max(proj_smooth[max(0, row-2):min(h, row+3)])
                if proj_smooth[row] >= local_max * 0.95 and proj_smooth[row] > np.max(proj_smooth) * 0.25:
                    raw_peaks.append(row)
            
            # Cluster raw peaks into groups (peaks within dy*0.3 of each other)
            peak_groups = []
            if raw_peaks:
                current_group = [raw_peaks[0]]
                for p in raw_peaks[1:]:
                    if p - current_group[-1] < dy * 0.3:
                        current_group.append(p)
                    else:
                        peak_groups.append(current_group)
                        current_group = [p]
                peak_groups.append(current_group)
            
            # For each group, find the row with maximum projection value
            group_peaks = []
            max_proj = np.max(proj_smooth)
            for group in peak_groups:
                best_row = max(group, key=lambda r: proj_smooth[r])
                # Only keep peaks that are at least 65% of the maximum projection value
                if proj_smooth[best_row] >= max_proj * 0.65:
                    group_peaks.append(best_row)
            
            # Final cleanup: if a peak is sandwiched between two higher peaks
            # and is much lower than both, it's likely a stem artifact
            clean_peaks = []
            for pi, peak in enumerate(group_peaks):
                peak_val = proj_smooth[peak]
                # Check if this peak is significantly lower than its neighbors
                prev_val = proj_smooth[group_peaks[pi-1]] if pi > 0 else 0
                next_val = proj_smooth[group_peaks[pi+1]] if pi < len(group_peaks)-1 else 0
                
                # If both neighbors exist and are much higher, skip this peak
                if prev_val > 0 and next_val > 0:
                    if peak_val < prev_val * 0.80 and peak_val < next_val * 0.80:
                        continue  # Skip this weak peak between two strong peaks
                
                clean_peaks.append(peak)
            
            if len(clean_peaks) >= 2:
                for pi_cp, p in enumerate(clean_peaks):
                    # Use weighted centroid from morphological projection around the peak
                    half_win = max(3, int(dy * 0.45))
                    p_start = max(0, p - half_win)
                    p_end = min(h, p + half_win + 1)
                    local_proj = proj_smooth[p_start:p_end]
                    if np.sum(local_proj) > 0:
                        local_center = np.average(np.arange(p_start, p_end), weights=local_proj)
                        note_cy = y + int(round(local_center))
                    else:
                        note_cy = y + p
                    all_boxes.append((x, note_cy - nh//2, nw, nh, 0.95))
            elif len(clean_peaks) == 1:
                p = clean_peaks[0]
                half_win = max(3, int(dy * 0.45))
                p_start = max(0, p - half_win)
                p_end = min(h, p + half_win + 1)
                local_proj = proj_smooth[p_start:p_end]
                if np.sum(local_proj) > 0:
                    local_center = np.average(np.arange(p_start, p_end), weights=local_proj)
                    note_cy = y + int(round(local_center))
                else:
                    note_cy = y + p
                all_boxes.append((x, note_cy - nh//2, nw, nh, 0.95))
            
        elif 0.5 < aspect < 2.2 and solidity > 0.45:
            all_boxes.append((x, cy_blob - nh//2, nw, nh, 1.0))
    
    # --- Stage 2: Template matching with ledger-line templates ---
    # Also search on-staff areas for notes that morphological detection might miss
    if staff_systems:
        ledger_templates = create_ledger_notehead_templates(dy)
        clean_template = create_notehead_template(dy)
        all_templates = [clean_template] + ledger_templates
        
        for system in staff_systems:
            top_line = system[0]
            bot_line = system[4]
            zone_height = int(dy * 5)
            
            # Search above and below the staff for ledger line notes
            above_y1 = max(0, top_line - zone_height)
            above_y2 = top_line
            below_y1 = bot_line
            below_y2 = min(binary_img.shape[0], bot_line + zone_height)
            
            # Also search the full staff area (for notes missed by morphological detection)
            # Use only the clean template for on-staff search to avoid false positives
            on_staff_y1 = max(0, top_line - int(dy * 1))
            on_staff_y2 = min(binary_img.shape[0], bot_line + int(dy * 1))
            
            # Ledger zone search (all templates)
            for (zy1, zy2) in [(above_y1, above_y2), (below_y1, below_y2)]:
                if zy2 - zy1 < nh:
                    continue
                roi = binary_img[zy1:zy2, :]
                
                for template in all_templates:
                    tw, th = template.shape[1], template.shape[0]
                    if tw >= roi.shape[1] or th >= roi.shape[0]:
                        continue
                    
                    res = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
                    loc = np.where(res >= threshold)
                    
                    for pt in zip(*loc[::-1]):
                        score = res[pt[1], pt[0]]
                        abs_x = pt[0] + tw // 2
                        abs_y = zy1 + pt[1] + th // 2
                        
                        if is_in_exclusion_zone(abs_x, abs_y, exclusion_zones, margin=int(dy * 0.3)):
                            continue
                        
                        all_boxes.append((abs_x - nw//2, abs_y - nh//2, nw, nh, score))
            
    # Record count of regular detections before hollow stage
    pre_hollow_count = len(all_boxes)

    # --- Stage 2b: Hollow notehead detection (whole/half notes) ---
    # Uses template matching on the binary image for accurate positioning.
    # The hollow notehead template is matched at multiple scales, then
    # center fill is verified (hollow = low fill).
    if detect_hollow and staff_systems:
        hollow_template = create_hollow_notehead_template(dy)
        ht_h, ht_w = hollow_template.shape

        for system in staff_systems:
            top_line = system[0]
            bot_line = system[4]
            search_y1 = max(0, top_line - int(dy * 4))
            search_y2 = min(binary_img.shape[0], bot_line + int(dy * 4))
            roi = binary_img[search_y1:search_y2, :]

            if roi.shape[0] < ht_h or roi.shape[1] < ht_w:
                continue

            for scale in [0.85, 1.0, 1.15]:
                sh = max(5, int(ht_h * scale))
                sw = max(5, int(ht_w * scale))
                if sh >= roi.shape[0] or sw >= roi.shape[1]:
                    continue
                tmpl = cv2.resize(hollow_template, (sw, sh),
                                  interpolation=cv2.INTER_AREA)
                res = cv2.matchTemplate(roi, tmpl, cv2.TM_CCOEFF_NORMED)
                loc = np.where(res >= 0.35)
                for py, px in zip(*loc):
                    cx_abs = px + sw // 2
                    cy_abs = search_y1 + py + sh // 2
                    match_score = float(res[py, px])

                    # Low-score detections (0.35-0.42) are only accepted
                    # at ledger line positions (above/below staff), where
                    # the ledger line interference lowers the template score.
                    # Within the staff, require score >= 0.42 to avoid
                    # false positives from stems and beam artifacts.
                    s_dy = (system[4] - system[0]) / 4.0
                    dist_above = system[0] - cy_abs
                    dist_below = cy_abs - system[4]
                    # Notes in spaces just above/below staff (0.5-1.0*dy) are
                    # valid positions but get lower template scores due to
                    # nearby staff line interference. Use graduated threshold:
                    # - On staff (dist <= 0): require 0.42
                    # - Near staff (dist 0-0.5*dy): require 0.42
                    # - Above/below staff (dist > 0.5*dy): allow 0.35+
                    outside_staff = dist_above > s_dy * 0.5 or dist_below > s_dy * 0.5
                    if match_score < 0.42 and not outside_staff:
                        continue
                    # For low-score ledger detections, verify ink presence
                    # on music_symbols — a real notehead has significant ink.
                    if match_score < 0.42 and outside_staff and music_symbols is not None:
                        ink_r = max(5, int(s_dy * 0.5))
                        iy1 = max(0, cy_abs - ink_r)
                        iy2 = min(music_symbols.shape[0], cy_abs + ink_r)
                        ix1 = max(0, cx_abs - ink_r)
                        ix2 = min(music_symbols.shape[1], cx_abs + ink_r)
                        ink_region = music_symbols[iy1:iy2, ix1:ix2]
                        ink_fill = np.mean(ink_region > 127) if ink_region.size else 0
                        if ink_fill < 0.20:
                            continue

                    if is_in_exclusion_zone(cx_abs, cy_abs, exclusion_zones,
                                            margin=int(dy * 0.3)):
                        continue

                    # Verify: center must be hollow (low ink)
                    r = max(2, int(dy * 0.15))
                    ry1 = max(0, cy_abs - r)
                    ry2 = min(binary_img.shape[0], cy_abs + r)
                    rx1 = max(0, cx_abs - r)
                    rx2 = min(binary_img.shape[1], cx_abs + r)
                    center_roi = binary_img[ry1:ry2, rx1:rx2]
                    if center_roi.size == 0:
                        continue
                    # On binary image, staff lines pass through center so
                    # check a vertical stripe just off-center for fill
                    vr = max(2, int(dy * 0.2))
                    vx1 = max(0, cx_abs - vr)
                    vx2 = min(binary_img.shape[1], cx_abs + vr)
                    vy1 = max(0, cy_abs - int(dy * 0.3))
                    vy2 = min(binary_img.shape[0], cy_abs + int(dy * 0.3))
                    center_block = binary_img[vy1:vy2, vx1:vx2]
                    fill = np.mean(center_block > 127) if center_block.size else 1.0

                    # Hollow notehead + staff lines: fill should be moderate
                    # (staff lines contribute some ink, but not as much as filled)
                    # Filled notehead fill: >0.7; hollow+staff lines: 0.2-0.5
                    if fill < 0.55:
                        # Refine y position: find centroid of ink on
                        # music_symbols near the detection. The contour
                        # fragments approximate the hollow oval center.
                        # Refine y: weighted centroid of ink on
                        # music_symbols within ±1.0*dy of template match.
                        # Uses ink count per row as weight → robust center.
                        refined_cy = cy_abs
                        if music_symbols is not None:
                            s_dy = (system[4] - system[0]) / 4.0
                            search_r = int(s_dy * 1.8)
                            hw = int(s_dy * 0.9)
                            ry1r = max(0, cy_abs - search_r)
                            ry2r = min(music_symbols.shape[0], cy_abs + search_r)
                            rx1r = max(0, cx_abs - hw)
                            rx2r = min(music_symbols.shape[1], cx_abs + hw)
                            patch = music_symbols[ry1r:ry2r, rx1r:rx2r]
                            if patch.size > 0:
                                row_ink = np.sum(patch > 127, axis=1).astype(float)
                                # Find ink runs and merge those separated by
                                # small gaps (staff/ledger line removal creates
                                # gaps of ~2-4px in hollow notehead outlines).
                                has_ink = (row_ink > 4).astype(np.uint8)
                                runs = np.diff(np.concatenate([[0], has_ink, [0]]))
                                starts = np.where(runs == 1)[0]
                                ends = np.where(runs == -1)[0]
                                if len(starts) > 0:
                                    # Merge runs with gap < 0.25*dy
                                    max_gap = max(3, int(s_dy * 0.25))
                                    merged_starts = [starts[0]]
                                    merged_ends = [ends[0]]
                                    for ri in range(1, len(starts)):
                                        gap = starts[ri] - merged_ends[-1]
                                        if gap <= max_gap:
                                            merged_ends[-1] = ends[ri]
                                        else:
                                            merged_starts.append(starts[ri])
                                            merged_ends.append(ends[ri])
                                    # Pick the longest merged run
                                    merged_lengths = [e - s for s, e in zip(merged_starts, merged_ends)]
                                    best_ri = int(np.argmax(merged_lengths))
                                    rs = merged_starts[best_ri]
                                    re = merged_ends[best_ri]
                                    # Use ink from all rows (including gap
                                    # rows which have 0 ink — this naturally
                                    # weights the centroid toward the center)
                                    seg_ink = row_ink[rs:re]
                                    seg_total = np.sum(seg_ink)
                                    if seg_total > 10:
                                        ys = np.arange(len(seg_ink))
                                        refined_cy = ry1r + rs + int(np.sum(ys * seg_ink) / seg_total)

                        score = float(res[py, px])
                        all_boxes.append((cx_abs - nw//2, refined_cy - nh//2,
                                          nw, nh, max(0.80, score)))

    # --- Stage 2c: Hollow notehead deduplication ---
    # Hollow detections (score ≤ 0.85) may produce multiple hits from
    # different template scales. Dedup with a wider radius (dy*1.0)
    # before final NMS to prevent duplicates from corrupting chord grouping.
    if detect_hollow:
        regular_boxes = list(all_boxes[:pre_hollow_count])
        hollow_boxes = all_boxes[pre_hollow_count:]
        hollow_boxes = _deduplicate_by_center(hollow_boxes, dy * 1.3)
        # When a hollow detection is near a regular detection, the hollow
        # one has better y-position (from y-refinement on music_symbols).
        # Replace the regular detection with the hollow one.
        new_hollow = []
        for hx, hy, hw, hh, hs in hollow_boxes:
            hcx = hx + hw / 2.0
            hcy = hy + hh / 2.0
            replaced = False
            for ri, (fx, fy, fw, fh, fs) in enumerate(regular_boxes):
                fcx = fx + fw / 2.0
                fcy = fy + fh / 2.0
                if abs(hcx - fcx) < dy * 1.5 and abs(hcy - fcy) < dy * 1.5:
                    # Replace regular with hollow (better y)
                    regular_boxes[ri] = (hx, hy, hw, hh, hs)
                    replaced = True
                    break
            if not replaced:
                new_hollow.append((hx, hy, hw, hh, hs))
        all_boxes = regular_boxes + new_hollow
        # Final pass: dedup any pairs created by hollow replacements
        all_boxes = _deduplicate_by_center(all_boxes, dy * 1.0)

    # --- Stage 3: NMS + center-distance deduplication ---
    picked_boxes = non_max_suppression(all_boxes, 0.3)
    picked_boxes = _deduplicate_by_center(picked_boxes, dy * 0.4)
    
    return picked_boxes, create_notehead_template(dy), exclusion_zones


def _load_real_notehead_templates(dy):
    """
    Load real notehead templates from the template/ folder (1_1.jpg, 1_2.jpg, etc.)
    and scale them to match the current staff spacing.
    """
    templates = []
    if not os.path.isdir(TEMPLATE_DIR):
        return templates
    
    for ext in ['*.png', '*.jpg']:
        for path in glob.glob(os.path.join(TEMPLATE_DIR, ext)):
            fname = os.path.basename(path)
            name = os.path.splitext(fname)[0]
            # Only load filled notehead templates (1_x)
            if name.startswith('1_'):
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    _, bimg = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
                    # Scale to match dy
                    th, tw = bimg.shape
                    ideal_h = int(dy * 0.85)
                    if th > 0:
                        scale = ideal_h / float(th)
                        new_w = max(3, int(tw * scale))
                        new_h = max(3, int(th * scale))
                        resized = cv2.resize(bimg, (new_w, new_h), interpolation=cv2.INTER_AREA)
                        templates.append(resized)
    
    return templates
