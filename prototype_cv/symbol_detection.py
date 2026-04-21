"""
symbol_detection.py
Detects barlines, accidentals (sharps/flats), rests, and note durations
from a binarized music score image.

Updated to use new templates from ../template/ folder with multiple
sharp/flat/rest variants for better detection accuracy.
"""
import cv2
import numpy as np
import os
import glob

from config import TEMPLATE_DIR, PICTURE_DIR, PICTURE_EXPAND_DIR


def _load_template(name, directory=None):
    """Load and binarize a template image."""
    if directory is None:
        directory = TEMPLATE_DIR
    path = os.path.join(directory, name)
    if not os.path.exists(path):
        return None
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    _, bimg = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return bimg


def _load_all_templates_by_prefix(prefix, directory=None):
    """Load all templates matching a prefix from the template directory."""
    if directory is None:
        directory = TEMPLATE_DIR
    templates = []
    if not os.path.isdir(directory):
        return templates
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        pattern = os.path.join(directory, ext)
        for path in glob.glob(pattern):
            fname = os.path.basename(path)
            name_no_ext = os.path.splitext(fname)[0]
            if name_no_ext.startswith(prefix):
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    _, bimg = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
                    templates.append((fname, bimg))
    return templates


# ============================================================
# 0. SLUR / TIE ARC DETECTION
# ============================================================
def detect_slur_arcs(music_symbols, staff_systems, dy):
    """Detect slur and tie arcs in the staff-removed image.

    Slur/tie arcs are thin curved lines connecting noteheads.  They cause
    false positives in hollow-notehead and rest detection.  This function
    finds them via contour analysis and returns a binary mask of arc pixels
    so they can be subtracted from ``music_symbols`` before downstream
    detection.

    Returns
    -------
    arc_mask : np.ndarray (same shape as music_symbols, dtype uint8)
        255 where a slur/tie arc pixel was detected, 0 elsewhere.
    arc_regions : list[dict]
        Each dict has keys x, y, w, h, system_idx describing a detected arc.
    """
    h, w = music_symbols.shape
    arc_mask = np.zeros((h, w), dtype=np.uint8)
    arc_regions = []

    min_arc_width = 3.0 * dy      # arcs span at least 3 staff-spacings
    max_arc_height = 1.5 * dy     # arcs are thin vertically
    max_fill = 0.22               # arcs are sparse (thin curve in bbox)

    for si, sys in enumerate(staff_systems):
        y_top = max(0, int(sys[0] - 2.5 * dy))
        y_bot = min(h, int(sys[4] + 2.5 * dy))
        roi = music_symbols[y_top:y_bot, :]

        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x_c, y_c, w_c, h_c = cv2.boundingRect(c)
            if w_c < min_arc_width or h_c > max_arc_height:
                continue
            area = cv2.contourArea(c)
            bbox_area = w_c * h_c
            if bbox_area == 0:
                continue
            fill = area / bbox_area
            if fill >= max_fill:
                continue

            # Additional guard: reject components whose arc length is too
            # short relative to width (noise fragments).  A real slur arc
            # has perimeter ≈ 2*width (top and bottom of thin curve).
            peri = cv2.arcLength(c, closed=True)
            if peri < w_c * 1.2:
                continue

            # Draw the contour (filled) onto the mask
            shifted = c.copy()
            shifted[:, :, 1] += y_top
            cv2.drawContours(arc_mask, [shifted], -1, 255, thickness=cv2.FILLED)
            # Also draw a thin border to capture anti-aliased edge pixels
            cv2.drawContours(arc_mask, [shifted], -1, 255, thickness=2)

            arc_regions.append({
                'x': x_c, 'y': y_c + y_top,
                'w': w_c, 'h': h_c,
                'system_idx': si,
            })

    return arc_mask, arc_regions


# ============================================================
# 1. BARLINE DETECTION
# ============================================================
def detect_barlines(binary_img, staff_systems, dy, min_spacing_dy=18.0):
    """
    Detect vertical barlines using template matching with the bar template.
    Returns list of barline x-positions for each staff system.

    min_spacing_dy: minimum spacing between barlines in multiples of dy.
                    Default 18.0 for grand staff; use ~8.0 for single-staff scores.
    """
    img_h, img_w = binary_img.shape
    
    bar_template = _load_template("bar.jpg", PICTURE_EXPAND_DIR)
    
    all_barlines = []
    
    for sys_idx, system in enumerate(staff_systems):
        y_top = system[0]
        y_bot = system[4]
        staff_height = y_bot - y_top
        margin = int(dy * 0.3)
        
        search_y1 = max(0, y_top - margin)
        search_y2 = min(img_h, y_bot + margin)
        roi = binary_img[search_y1:search_y2, :]
        roi_h, roi_w = roi.shape
        
        barline_xs = []
        
        if bar_template is not None:
            th_orig, tw_orig = bar_template.shape
            ideal_scale = staff_height / float(th_orig) if th_orig > 0 else 1.0
            
            for scale_factor in [ideal_scale * 0.85, ideal_scale, ideal_scale * 1.15]:
                new_h = int(th_orig * scale_factor)
                new_w = max(2, int(tw_orig * scale_factor))
                
                if new_h < 5 or new_h >= roi_h or new_w >= roi_w:
                    continue
                
                resized = cv2.resize(bar_template, (new_w, new_h), interpolation=cv2.INTER_AREA)
                res = cv2.matchTemplate(roi, resized, cv2.TM_CCOEFF_NORMED)
                
                threshold = 0.50
                loc = np.where(res >= threshold)
                
                for pt in zip(*loc[::-1]):
                    bx = pt[0] + new_w // 2
                    score = res[pt[1], pt[0]]
                    barline_xs.append((bx, score))
        
        # Fallback: vertical morphology
        vert_kernel_len = max(int(staff_height * 0.8), 15)
        vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_kernel_len))
        vertical_lines = cv2.morphologyEx(roi, cv2.MORPH_OPEN, vert_kernel, iterations=1)
        
        horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, int(dy * 0.3)), 1))
        thick_vert = cv2.morphologyEx(vertical_lines, cv2.MORPH_OPEN, horiz_kernel, iterations=1)
        thin_vert = cv2.subtract(vertical_lines, thick_vert)
        
        proj = np.sum(thin_vert, axis=0).astype(float)
        threshold_proj = staff_height * 255 * 0.5
        
        in_peak = False
        peak_start = 0
        for x in range(len(proj)):
            if proj[x] > threshold_proj:
                if not in_peak:
                    peak_start = x
                    in_peak = True
            else:
                if in_peak:
                    peak_center = (peak_start + x) // 2
                    barline_xs.append((peak_center, 0.6))
                    in_peak = False
        
        barline_xs.sort(key=lambda b: b[0])
        deduped = []
        for bx, score in barline_xs:
            if not deduped or abs(bx - deduped[-1][0]) > dy * 3:
                deduped.append((bx, score))
            else:
                if score > deduped[-1][1]:
                    deduped[-1] = (bx, score)
        
        # Adaptive spacing: find natural threshold between stem gaps and
        # barline gaps using the largest jump in the sorted gap distribution.
        if len(deduped) >= 3 and min_spacing_dy <= 0:
            cand_xs = [b[0] for b in deduped]
            raw_gaps = sorted([cand_xs[i+1] - cand_xs[i]
                               for i in range(len(cand_xs) - 1)])
            # Find largest jump between consecutive sorted gaps
            best_jump = 0
            best_threshold = dy * 8  # fallback
            for gi in range(len(raw_gaps) - 1):
                jump = raw_gaps[gi + 1] - raw_gaps[gi]
                if jump > best_jump:
                    best_jump = jump
                    best_threshold = (raw_gaps[gi] + raw_gaps[gi + 1]) / 2
            # Ensure minimum threshold to avoid merging very close stems
            adaptive_spacing = max(dy * 5, best_threshold)
        else:
            adaptive_spacing = dy * min_spacing_dy

        filtered = []
        for bx, score in deduped:
            if not filtered or (bx - filtered[-1]) > adaptive_spacing:
                filtered.append(bx)
        
        while filtered and filtered[0] < int(img_w * 0.20):
            filtered = filtered[1:]
        
        if filtered and filtered[-1] > int(img_w * 0.96):
            filtered = filtered[:-1]
            
        all_barlines.append(filtered)
    
    return all_barlines


# ============================================================
# 2. ACCIDENTAL DETECTION - GLOBAL APPROACH
# ============================================================
def detect_accidentals_global(binary_img, staff_systems, dy, clef_boundaries=None,
                              music_symbols=None):
    """Detect accidentals globally in each staff region using multi-scale template matching.
    Uses ALL available sharp/flat/natural templates from the new template/ folder.

    If music_symbols is provided, also searches it for naturals (catches signs
    that overlap staff lines and are invisible on the binary image).

    Returns list of dicts: {'x': x, 'y': y, 'type': '#'/'b'/'n', 'score': score, 'system_idx': idx}
    Natural signs ('n') are used to cancel accidental persistence within a measure.
    """
    from config import CFG

    # Load all sharp, flat, and natural templates from new template folder
    sharp_templates = _load_all_templates_by_prefix("sharp_", TEMPLATE_DIR)
    flat_templates = _load_all_templates_by_prefix("flat_", TEMPLATE_DIR)
    natural_templates = _load_all_templates_by_prefix("natural_", TEMPLATE_DIR)
    natural_templates += _load_all_templates_by_prefix("nature_", TEMPLATE_DIR)

    # Also load from legacy folder
    for name in ["sharp_1.jpg"]:
        t = _load_template(name, PICTURE_DIR)
        if t is not None:
            sharp_templates.append((name + "_legacy", t))
    for name in ["flat_1.jpg"]:
        t = _load_template(name, PICTURE_DIR)
        if t is not None:
            flat_templates.append((name + "_legacy", t))
    for name in ["natural_1.jpg"]:
        t = _load_template(name, PICTURE_DIR)
        if t is not None:
            natural_templates.append((name + "_legacy", t))

    print(f"   Loaded {len(sharp_templates)} sharp, {len(flat_templates)} flat, {len(natural_templates)} natural templates")
    
    img_h, img_w = binary_img.shape
    all_accidentals = []
    
    for sys_idx, system in enumerate(staff_systems):
        y_top = system[0]
        y_bot = system[4]
        margin = int(dy * 5)  # Extended to cover ledger line notes
        search_y1 = max(0, y_top - margin)
        search_y2 = min(img_h, y_bot + margin)
        
        # Skip clef area: use per-system boundary if available, else 18% fallback
        if clef_boundaries and sys_idx in clef_boundaries:
            search_x1 = max(0, clef_boundaries[sys_idx] - int(dy * 2))
        else:
            search_x1 = int(img_w * 0.18)
        
        roi = binary_img[search_y1:search_y2, search_x1:]
        roi_h, roi_w = roi.shape
        if roi_h < 10 or roi_w < 10:
            continue
        
        for templates, acc_char in [(sharp_templates, '#'), (flat_templates, 'b'), (natural_templates, 'n')]:
            for tname, template in templates:
                th_orig, tw_orig = template.shape
                if th_orig < 3 or tw_orig < 3:
                    continue
                
                # Accidentals are about 1.5-2.5 dy tall
                ideal_scale = (dy * 2.0) / float(th_orig)
                
                for scale_factor in np.linspace(ideal_scale * 0.7, ideal_scale * 1.35, 7):
                    new_h = int(th_orig * scale_factor)
                    new_w = int(tw_orig * scale_factor)
                    
                    if new_h < 5 or new_w < 3 or new_h >= roi_h or new_w >= roi_w:
                        continue
                    
                    resized = cv2.resize(template, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    res = cv2.matchTemplate(roi, resized, cv2.TM_CCOEFF_NORMED)

                    ac = CFG.accidental
                    threshold = ac.match_threshold_sharp if acc_char == '#' else ac.match_threshold_global
                    loc = np.where(res >= threshold)
                    
                    for pt in zip(*loc[::-1]):
                        score = float(res[pt[1], pt[0]])
                        abs_x = search_x1 + pt[0] + new_w // 2
                        abs_y = search_y1 + pt[1] + new_h // 2
                        
                        all_accidentals.append({
                            'x': abs_x,
                            'y': abs_y,
                            'type': acc_char,
                            'score': score,
                            'system_idx': sys_idx,
                            'w': new_w,
                            'h': new_h,
                        })
    
    # Supplemental pass: search music_symbols for naturals that overlap staff
    # lines. Staff removal destroys the middle of the sign, but the vertical
    # bars survive above/below the line, giving a weaker but valid match.
    if music_symbols is not None and natural_templates:
        ms_threshold = CFG.accidental.match_threshold_global - 0.05  # slightly lower
        for sys_idx, system in enumerate(staff_systems):
            y_top = system[0]
            y_bot = system[4]
            margin = int(dy * 5)
            search_y1 = max(0, y_top - margin)
            search_y2 = min(img_h, y_bot + margin)
            if clef_boundaries and sys_idx in clef_boundaries:
                search_x1 = max(0, clef_boundaries[sys_idx] - int(dy * 2))
            else:
                search_x1 = int(img_w * 0.18)
            roi = music_symbols[search_y1:search_y2, search_x1:]
            roi_h, roi_w = roi.shape
            if roi_h < 10 or roi_w < 10:
                continue
            for tname, template in natural_templates:
                th_orig, tw_orig = template.shape
                if th_orig < 3 or tw_orig < 3:
                    continue
                ideal_scale = (dy * 2.0) / float(th_orig)
                for scale_factor in np.linspace(ideal_scale * 0.7, ideal_scale * 1.35, 7):
                    new_h = int(th_orig * scale_factor)
                    new_w = int(tw_orig * scale_factor)
                    if new_h < 5 or new_w < 3 or new_h >= roi_h or new_w >= roi_w:
                        continue
                    resized = cv2.resize(template, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    res = cv2.matchTemplate(roi, resized, cv2.TM_CCOEFF_NORMED)
                    loc = np.where(res >= ms_threshold)
                    for pt in zip(*loc[::-1]):
                        score = float(res[pt[1], pt[0]])
                        abs_x = search_x1 + pt[0] + new_w // 2
                        abs_y = search_y1 + pt[1] + new_h // 2
                        all_accidentals.append({
                            'x': abs_x, 'y': abs_y, 'type': 'n',
                            'score': score, 'system_idx': sys_idx,
                            'w': new_w, 'h': new_h,
                        })

    # NMS: remove duplicates (keep highest score within dy*0.8 radius)
    if all_accidentals:
        all_accidentals.sort(key=lambda a: a['score'], reverse=True)
        kept = []
        for acc in all_accidentals:
            is_dup = False
            for k in kept:
                if abs(acc['x'] - k['x']) < dy * 0.8 and abs(acc['y'] - k['y']) < dy * 0.8:
                    is_dup = True
                    break
            if not is_dup:
                kept.append(acc)
        all_accidentals = kept

    print(f"   Global accidental detection: {len(all_accidentals)} found "
          f"({sum(1 for a in all_accidentals if a['type']=='#')} sharps, "
          f"{sum(1 for a in all_accidentals if a['type']=='b')} flats, "
          f"{sum(1 for a in all_accidentals if a['type']=='n')} naturals)")
    for a in all_accidentals:
        print(f"     {a['type']} at ({a['x']}, {a['y']}) score={a['score']:.3f} sys={a['system_idx']}")

    return all_accidentals


def assign_accidentals_to_notes(global_accidentals, noteheads, dy):
    """
    Assign globally detected accidentals to their nearest noteheads.
    
    Rules:
    - Accidental must be to the LEFT of the note (or very slightly overlapping)
    - Accidental y must be close to note y (within 1.2 * dy)
    - Accidental x must be within 3.5 * dy of the note
    - Each accidental is assigned to exactly one note (the closest valid one)
    
    Returns dict mapping (note_cx, note_cy) -> '#' or 'b'
    """
    accidentals_map = {}
    # Sort accidentals by score descending so high-confidence ones get assigned first
    sorted_accs = sorted(enumerate(global_accidentals), key=lambda x: x[1]['score'], reverse=True)
    
    for acc_idx, acc in sorted_accs:
        best_note = None
        best_dist = float('inf')
        best_key = None
        acc_sys_idx = acc.get('system_idx', -1)
        
        for note in noteheads:
            note_cx = note['x'] + note['w'] // 2
            note_cy = note['y_center']
            
            # Accidental and note must be on the same staff system (or adjacent)
            note_system = note.get('system')
            if note_system is not None and acc_sys_idx >= 0:
                # Check if the accidental's staff system matches the note's staff
                note_sys_top = note_system[0]
                note_sys_bot = note_system[4]
                # Accidental y should be within the note's staff region (with margin)
                if abs(acc['y'] - (note_sys_top + note_sys_bot) / 2.0) > (note_sys_bot - note_sys_top) * 1.5:
                    continue
            
            # Accidental must be to the left of the note
            dx = note_cx - acc['x']
            if dx < -dy * 0.5:  # allow slight overlap
                continue
            if dx > dy * 3.5:
                continue
            
            dy_dist = abs(note_cy - acc['y'])
            if dy_dist > dy * 1.2:
                continue
            
            dist = np.sqrt(dx**2 + dy_dist**2)
            key = (note_cx, note_cy)
            
            if dist < best_dist:
                best_dist = dist
                best_note = note
                best_key = key
        
        if best_note is not None and best_key is not None:
            # Only assign if this note doesn't already have a higher-score accidental
            if best_key not in accidentals_map:
                accidentals_map[best_key] = (acc['type'], acc['score'])
    
    result = {k: v[0] for k, v in accidentals_map.items()}
    print(f"   Assigned {len(result)} accidentals to notes "
          f"({sum(1 for v in result.values() if v == '#')} sharps, "
          f"{sum(1 for v in result.values() if v == 'b')} flats, "
          f"{sum(1 for v in result.values() if v == 'n')} naturals)")
    return result




def _match_accidental_template(roi, template, dy, roi_h, roi_w):
    """Match a single accidental template against an ROI using multi-scale matching."""
    th_orig, tw_orig = template.shape
    if th_orig < 3 or tw_orig < 3:
        return 0.0
    
    ideal_scale = (dy * 2.0) / float(th_orig)
    best_score = 0.0
    
    for scale_factor in np.linspace(ideal_scale * 0.7, ideal_scale * 1.35, 7):
        new_h = int(th_orig * scale_factor)
        new_w = int(tw_orig * scale_factor)
        
        if new_h < 5 or new_w < 3 or new_h >= roi_h or new_w >= roi_w:
            continue
        
        resized = cv2.resize(template, (new_w, new_h), interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(roi, resized, cv2.TM_CCOEFF_NORMED)
        
        _, max_val, _, _ = cv2.minMaxLoc(res)
        
        if max_val > best_score:
            best_score = max_val
    
    return best_score


# ============================================================
# 3. REST DETECTION
# ============================================================
def detect_rests(binary_img, staff_systems, dy):
    """Detect rest symbols using templates."""
    from config import CFG
    rc = CFG.rest

    rests = []
    img_h, img_w = binary_img.shape

    # Per-template config: (filename, duration, height_ratio, scale_factors, threshold, dirs)
    rest_specs = [
        ('stop_4.jpg', 1.0, rc.quarter_height_ratio, rc.quarter_scale_factors,
         rc.quarter_threshold, [TEMPLATE_DIR, PICTURE_EXPAND_DIR, PICTURE_DIR]),
        ('stop_8.jpg', 0.5, rc.eighth_height_ratio, rc.eighth_scale_factors,
         rc.eighth_threshold, [TEMPLATE_DIR, PICTURE_EXPAND_DIR]),
        ('stop_2.jpg', 2.0, 1.0, rc.half_whole_scale_factors,
         rc.half_threshold, [TEMPLATE_DIR, PICTURE_EXPAND_DIR, PICTURE_DIR]),
        ('stop_1.jpg', 4.0, 1.0, rc.half_whole_scale_factors,
         rc.whole_threshold, [TEMPLATE_DIR, PICTURE_EXPAND_DIR, PICTURE_DIR]),
    ]

    for sys_idx, system in enumerate(staff_systems):
        staff_height = system[4] - system[0]
        margin = int(dy * 0.5)
        search_y1 = max(0, system[0] - margin)
        search_y2 = min(img_h, system[4] + margin)
        roi = binary_img[search_y1:search_y2, :]
        roi_h, roi_w = roi.shape

        for tname, duration, height_ratio, scale_factors, threshold, dirs in rest_specs:
            template = None
            for d in dirs:
                template = _load_template(tname, d)
                if template is not None:
                    break
            if template is None:
                continue

            th_orig, tw_orig = template.shape
            ideal_scale = (staff_height * height_ratio) / float(th_orig) if th_orig > 0 else 0.5

            for sf in scale_factors:
                new_h = int(th_orig * ideal_scale * sf)
                new_w = int(tw_orig * ideal_scale * sf)
                if new_h < 5 or new_w < 5 or new_h >= roi_h or new_w >= roi_w:
                    continue

                resized = cv2.resize(template, (new_w, new_h), interpolation=cv2.INTER_AREA)
                res = cv2.matchTemplate(roi, resized, cv2.TM_CCOEFF_NORMED)
                loc = np.where(res >= threshold)

                for pt in zip(*loc[::-1]):
                    rests.append({
                        'x': pt[0] + new_w // 2,
                        'y_center': search_y1 + pt[1] + new_h // 2,
                        'system_idx': sys_idx,
                        'duration': duration,
                        'score': float(res[pt[1], pt[0]]),
                        'type': tname,
                    })

    if rests:
        rests = _nms_rests(rests, dy * rc.nms_distance_dy)

    return rests


def _nms_rests(rests, min_dist_x, min_dist_y=None):
    """Remove duplicate rest detections.

    stop_1/stop_2 templates include staff-line context, inflating their
    correlation scores. Give priority to stop_4/stop_8 (block-only
    templates): in type-priority tier first, then by score.
    """
    if min_dist_y is None:
        min_dist_y = min_dist_x * 4

    def _priority(r):
        t = r.get('type', '')
        # stop_4 has the most distinctive shape and fires reliably on
        # quarter rests. stop_8 over-fires on slurs/chords and on
        # quarter rests themselves (where it competes with stop_4).
        # When the two overlap, prefer the quarter-rest interpretation.
        if t == 'stop_4.jpg':
            return 0
        if t == 'stop_8.jpg':
            return 1
        return 2  # stop_1/stop_2 — lowest priority due to staff-line inflation

    sorted_rests = sorted(rests, key=lambda r: (_priority(r), -r['score']))
    kept = []
    for r in sorted_rests:
        too_close = False
        for kr in kept:
            if abs(r['x'] - kr['x']) < min_dist_x and abs(r['y_center'] - kr['y_center']) < min_dist_y:
                too_close = True
                break
        if not too_close:
            kept.append(r)
    return kept


# ============================================================
# 5. TUPLET MARKER DETECTION
# ============================================================
def detect_tuplet_markers(grayscale_img, grand_staff_pairs, dy, clef_boundaries=None):
    """Detect tuplet number markers (3, 6) using template matching.

    Tuplet markers are small digits placed below the staff (in the gap
    between treble and bass, or below the bass staff) indicating that a
    group of notes forms a tuplet.

    Parameters
    ----------
    grayscale_img : ndarray, grayscale original image
    grand_staff_pairs : list of (treble_sys, bass_sys) tuples
    dy : float, staff line spacing
    clef_boundaries : dict mapping system index to clef x boundary

    Returns
    -------
    list of dicts: {'x', 'y', 'n' (tuplet number), 'pair_idx', 'clef'}
    """
    # Load tuplet digit templates
    templates = []
    for digit in [3, 6]:
        t = _load_template(f"tuplet_{digit}.png")
        if t is not None:
            templates.append((digit, t))
    if not templates:
        return []

    img_h, img_w = grayscale_img.shape[:2]
    _, img_bin = cv2.threshold(grayscale_img, 128, 255, cv2.THRESH_BINARY_INV)

    default_clef_x = int(img_w * 0.15)
    raw_hits = []

    for pair_idx, (treble_sys, bass_sys) in enumerate(grand_staff_pairs):
        clef_x = default_clef_x
        if clef_boundaries:
            sys_idx = pair_idx * 2
            clef_x = clef_boundaries.get(sys_idx, default_clef_x)

        # Search regions: between staves, below bass staff
        regions = [
            ('treble', treble_sys[4], bass_sys[0]),          # between staves
            ('bass', bass_sys[4], min(img_h, bass_sys[4] + int(dy * 5))),  # below bass
        ]

        for clef, ry1, ry2 in regions:
            if ry2 <= ry1:
                continue
            roi = img_bin[ry1:ry2, :]

            for digit, template in templates:
                th, tw = template.shape
                for scale in [0.7, 0.85, 1.0, 1.15, 1.3]:
                    new_h = max(5, int(th * scale))
                    new_w = max(5, int(tw * scale))
                    if new_h >= roi.shape[0] or new_w >= roi.shape[1]:
                        continue
                    resized = cv2.resize(template, (new_w, new_h))
                    res = cv2.matchTemplate(roi, resized, cv2.TM_CCOEFF_NORMED)
                    loc = np.where(res >= 0.65)
                    for py, px in zip(*loc):
                        abs_x = px + new_w // 2
                        abs_y = ry1 + py + new_h // 2
                        # Skip clef area
                        if abs_x < clef_x:
                            continue
                        score = res[py, px]
                        raw_hits.append({
                            'x': abs_x, 'y': abs_y, 'n': digit,
                            'pair_idx': pair_idx, 'clef': clef,
                            'score': score,
                        })

    # NMS: keep highest score within 15px radius
    raw_hits.sort(key=lambda h: -h['score'])
    kept = []
    for h in raw_hits:
        too_close = any(
            abs(h['x'] - k['x']) < 15 and abs(h['y'] - k['y']) < 15
            for k in kept
        )
        if not too_close:
            kept.append(h)

    return kept


_TIMESIG_TEMPLATE_CACHE = None


def _load_timesig_templates():
    """Load and cache binarized time-signature templates keyed by (num, den)."""
    global _TIMESIG_TEMPLATE_CACHE
    if _TIMESIG_TEMPLATE_CACHE is not None:
        return _TIMESIG_TEMPLATE_CACHE

    templates = {}
    for ext in ['*.jpg', '*.png']:
        for path in glob.glob(os.path.join(TEMPLATE_DIR, ext)):
            fname = os.path.splitext(os.path.basename(path))[0]
            if not fname.startswith('number_'):
                continue
            parts = fname.split('_')
            if len(parts) >= 3:
                try:
                    num, den = int(parts[1]), int(parts[2])
                except ValueError:
                    continue
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    _, bimg = cv2.threshold(
                        img, 128, 255,
                        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
                    templates[(num, den)] = bimg
    _TIMESIG_TEMPLATE_CACHE = templates
    return templates


def _match_timesig_in_roi(roi, staff_height, min_score=0.50):
    """Match timesig templates against ROI.

    Returns ``((num, den), score, match_x_in_roi, match_w)`` or ``None``.
    ``match_x_in_roi`` is the center x of the best match relative to the
    ROI's left edge; callers add the ROI offset to recover absolute x.
    ``match_w`` is the resized template width for that match.
    """
    if roi is None or roi.size == 0:
        return None
    templates = _load_timesig_templates()
    if not templates:
        return None

    best_match = None
    best_score = min_score
    best_x = None
    best_w = None

    for (num, den), template in templates.items():
        th, tw = template.shape
        ideal_scale = staff_height / th if th > 0 else 1.0
        for f in [0.8, 0.9, 1.0, 1.1, 1.2]:
            new_h = max(5, int(th * ideal_scale * f))
            new_w = max(5, int(tw * ideal_scale * f))
            if new_h >= roi.shape[0] or new_w >= roi.shape[1]:
                continue
            resized = cv2.resize(template, (new_w, new_h),
                                 interpolation=cv2.INTER_AREA)
            res = cv2.matchTemplate(roi, resized, cv2.TM_CCOEFF_NORMED)
            if res.size == 0:
                continue
            max_val = float(np.max(res))
            if max_val > best_score:
                best_score = max_val
                best_match = (num, den)
                py, px = np.unravel_index(int(np.argmax(res)), res.shape)
                best_x = int(px) + new_w // 2
                best_w = new_w

    if best_match is None:
        return None
    return (best_match, best_score, best_x, best_w)


def _scan_timesigs_full_line(binary_img, staff_system, dy, x1, x2,
                             min_score=0.70, nms_dy=1.5):
    """Scan a horizontal range of one staff for time-signature matches.

    Returns list of dicts [{'x': abs_x_of_match_left, 'num': n, 'den': d,
    'score': s}, ...] sorted by x, with NMS applied so nearby hits are
    collapsed to the single best.

    Unlike the per-ROI matcher, this one reports the x-position of each
    match so callers can snap to the nearest barline.
    """
    templates = _load_timesig_templates()
    if not templates:
        return []

    staff_height = staff_system[4] - staff_system[0]
    y1 = max(0, staff_system[0] - int(dy * 0.5))
    y2 = min(binary_img.shape[0], staff_system[4] + int(dy * 0.5))
    x1 = max(0, int(x1))
    x2 = min(binary_img.shape[1], int(x2))
    if x2 - x1 < 10:
        return []
    roi = binary_img[y1:y2, x1:x2]
    roi_h, roi_w = roi.shape

    raw = []  # (abs_x, num, den, score)
    for (num, den), template in templates.items():
        th, tw = template.shape
        ideal_scale = staff_height / th if th > 0 else 1.0
        for f in [0.8, 0.9, 1.0, 1.1, 1.2]:
            new_h = max(5, int(th * ideal_scale * f))
            new_w = max(5, int(tw * ideal_scale * f))
            if new_h >= roi_h or new_w >= roi_w:
                continue
            resized = cv2.resize(template, (new_w, new_h),
                                 interpolation=cv2.INTER_AREA)
            res = cv2.matchTemplate(roi, resized, cv2.TM_CCOEFF_NORMED)
            if res.size == 0:
                continue
            locs = np.where(res >= min_score)
            for (py, px) in zip(locs[0], locs[1]):
                raw.append((x1 + int(px) + new_w // 2,
                            num, den, float(res[py, px])))

    if not raw:
        return []

    # NMS: sort by score desc, greedily keep non-overlapping
    raw.sort(key=lambda t: -t[3])
    kept = []
    for cand in raw:
        cx, cn, cd, cs = cand
        if any(abs(cx - k[0]) < dy * nms_dy for k in kept):
            continue
        kept.append(cand)

    kept.sort(key=lambda t: t[0])
    return [{'x': float(cx), 'num': cn, 'den': cd, 'score': cs}
            for cx, cn, cd, cs in kept]


# ============================================================
# KEY SIGNATURE DETECTION
# ============================================================

# Circle of fifths: fixed order of sharps/flats in key signatures.
# Jianpu note numbers: 1=C, 2=D, 3=E, 4=F, 5=G, 6=A, 7=B
_SHARPS_ORDER_JIANPU = [4, 1, 5, 2, 6, 3, 7]  # F C G D A E B
_FLATS_ORDER_JIANPU = [7, 3, 6, 2, 5, 1, 4]   # B E A D G C F


def detect_key_signature(binary_img, staff_system, dy, time_sig_x=None):
    """Detect the key signature from the clef area of a staff system.

    Searches the region between the clef symbol and the time signature
    (or the standard clef-area right edge if no time sig found) for
    sharp and flat accidentals.

    Parameters
    ----------
    binary_img : ndarray
        Binarized image (white-on-black notation).
    staff_system : list
        5 Y-coordinates of the staff lines.
    dy : float
        Staff-line spacing.
    time_sig_x : int or None
        X-coordinate of the detected time signature's left edge.
        If provided, limits the key-sig search to end before it.

    Returns
    -------
    dict or None
        {'type': '#' or 'b', 'count': int, 'notes': list[int]}
        where 'notes' are the jianpu note numbers affected.
        Returns None if no key signature detected (C major / A minor).
    """
    img_h, img_w = binary_img.shape
    staff_height = staff_system[4] - staff_system[0]

    # Key signature sits between the clef symbol and time signature.
    # Treble/bass clef symbols are about 3.5-4 dy wide from the left margin.
    # Start searching AFTER the clef to avoid false positives on clef curves.
    roi_x1 = max(int(img_w * 0.05), int(dy * 4))
    if time_sig_x is not None:
        roi_x2 = max(roi_x1 + 10, time_sig_x - int(dy * 0.5))
    else:
        roi_x2 = int(img_w * 0.12)

    # Vertical range: staff lines ± 1.5dy to catch accidentals on ledger positions
    margin_y = int(dy * 1.5)
    roi_y1 = max(0, staff_system[0] - margin_y)
    roi_y2 = min(img_h, staff_system[4] + margin_y)

    roi = binary_img[roi_y1:roi_y2, roi_x1:roi_x2]
    roi_h, roi_w = roi.shape
    if roi_h < 10 or roi_w < 10:
        return None

    # Load sharp and flat templates
    sharp_templates = _load_all_templates_by_prefix("sharp_", TEMPLATE_DIR)
    flat_templates = _load_all_templates_by_prefix("flat_", TEMPLATE_DIR)

    # Also load legacy templates
    for name in ["sharp_1.jpg"]:
        t = _load_template(name, PICTURE_DIR)
        if t is not None:
            sharp_templates.append((name + "_legacy", t))
    for name in ["flat_1.jpg"]:
        t = _load_template(name, PICTURE_DIR)
        if t is not None:
            flat_templates.append((name + "_legacy", t))

    if not sharp_templates and not flat_templates:
        return None

    # Detect accidentals in the key-sig ROI
    # Key sig accidentals are spaced ~0.8-1.2 dy apart horizontally
    # and occupy the full staff height vertically.
    min_score = 0.55

    sharp_hits = []
    flat_hits = []

    for templates, hits_list in [(sharp_templates, sharp_hits),
                                  (flat_templates, flat_hits)]:
        for tname, template in templates:
            th_orig, tw_orig = template.shape
            if th_orig < 3 or tw_orig < 3:
                continue
            # Key-sig accidentals are about 2-2.5 dy tall
            ideal_scale = (dy * 2.2) / float(th_orig)
            for scale_factor in np.linspace(ideal_scale * 0.75, ideal_scale * 1.3, 6):
                new_h = int(th_orig * scale_factor)
                new_w = int(tw_orig * scale_factor)
                if new_h < 5 or new_w < 3 or new_h >= roi_h or new_w >= roi_w:
                    continue
                resized = cv2.resize(template, (new_w, new_h),
                                     interpolation=cv2.INTER_AREA)
                res = cv2.matchTemplate(roi, resized, cv2.TM_CCOEFF_NORMED)
                if res.size == 0:
                    continue
                locs = np.where(res >= min_score)
                for (py, px) in zip(locs[0], locs[1]):
                    hits_list.append({
                        'x': roi_x1 + int(px) + new_w // 2,
                        'y': roi_y1 + int(py) + new_h // 2,
                        'score': float(res[py, px]),
                        'w': new_w, 'h': new_h,
                    })

    # NMS within each type: collapse hits within 0.7*dy distance
    def _nms(hits, radius):
        if not hits:
            return []
        hits_sorted = sorted(hits, key=lambda h: h['score'], reverse=True)
        kept = []
        for h in hits_sorted:
            dup = False
            for k in kept:
                if abs(h['x'] - k['x']) < radius and abs(h['y'] - k['y']) < radius:
                    dup = True
                    break
            if not dup:
                kept.append(h)
        return kept

    nms_radius = dy * 0.7
    sharp_hits = _nms(sharp_hits, nms_radius)
    flat_hits = _nms(flat_hits, nms_radius)

    n_sharps = len(sharp_hits)
    n_flats = len(flat_hits)

    # Key signatures are exclusively sharps OR flats, never mixed.
    # Pick the type with more detections; if tied or both 0, no key sig.
    if n_sharps == 0 and n_flats == 0:
        return None

    # Validation: key sig accidentals should be clustered in a narrow x-band
    # (roughly within 5*dy width). Reject if hits are too spread out.
    def _validate_cluster(hits):
        if len(hits) <= 1:
            return hits
        xs = [h['x'] for h in hits]
        x_span = max(xs) - min(xs)
        # Key sig typically spans at most 4-5 dy wide
        if x_span > dy * 6:
            # Remove outliers: keep only hits within the densest cluster
            xs_sorted = sorted(xs)
            best_count = 0
            best_start = 0
            window = dy * 5
            for i, x0 in enumerate(xs_sorted):
                count = sum(1 for x in xs_sorted if x0 <= x <= x0 + window)
                if count > best_count:
                    best_count = count
                    best_start = x0
            hits = [h for h in hits if best_start <= h['x'] <= best_start + window]
        return hits

    sharp_hits = _validate_cluster(sharp_hits)
    flat_hits = _validate_cluster(flat_hits)
    n_sharps = len(sharp_hits)
    n_flats = len(flat_hits)

    # Decide: sharps or flats?
    if n_sharps > n_flats:
        acc_type = '#'
        count = min(n_sharps, 7)  # max 7 sharps
        notes = _SHARPS_ORDER_JIANPU[:count]
    elif n_flats > n_sharps:
        acc_type = 'b'
        count = min(n_flats, 7)  # max 7 flats
        notes = _FLATS_ORDER_JIANPU[:count]
    else:
        # Tied: use average score to break tie
        avg_sharp = sum(h['score'] for h in sharp_hits) / n_sharps if n_sharps else 0
        avg_flat = sum(h['score'] for h in flat_hits) / n_flats if n_flats else 0
        if avg_sharp >= avg_flat:
            acc_type = '#'
            count = min(n_sharps, 7)
            notes = _SHARPS_ORDER_JIANPU[:count]
        else:
            acc_type = 'b'
            count = min(n_flats, 7)
            notes = _FLATS_ORDER_JIANPU[:count]

    # Final sanity checks
    chosen_hits = sharp_hits if acc_type == '#' else flat_hits
    avg_score = sum(h['score'] for h in chosen_hits) / len(chosen_hits)
    # Single accidental detections are prone to false positives (clef
    # residue, artifacts) — require higher confidence.
    min_avg = 0.65 if count == 1 else 0.55
    if avg_score < min_avg:
        return None

    print(f"   [key_sig] Detected: {count}{acc_type} "
          f"(sharps_found={n_sharps}, flats_found={n_flats}, avg_score={avg_score:.3f})")
    print(f"   [key_sig] Affected jianpu notes: {notes}")

    return {'type': acc_type, 'count': count, 'notes': notes}


def detect_time_signature(binary_img, staff_system, dy):
    """Auto-detect time signature from a staff system's clef area.

    Matches time-signature templates (number_N_D format). Returns
    (numerator, denominator) or None if not detected.
    """
    staff_height = staff_system[4] - staff_system[0]
    img_w = binary_img.shape[1]

    roi_x1 = int(img_w * 0.08)
    roi_x2 = int(img_w * 0.22)
    roi = binary_img[staff_system[0]:staff_system[4], roi_x1:roi_x2]
    match = _match_timesig_in_roi(roi, staff_height, min_score=0.45)
    if match is None:
        return None
    return match[0]


def detect_time_signatures_along_system(binary_img, staff_system, barlines, dy,
                                        clef_roi_x1=None, clef_roi_x2=None):
    """Scan a staff system for time signatures.

    Two-phase approach:
    1. Scan the clef area for an initial time sig (keyed x=0, source='clef').
    2. Scan the full staff width beyond the clef for additional matches,
       apply NMS, then snap each match to the nearest barline and emit one
       anchor per snapped position (source='barline').

    Why full-line scan instead of per-barline ROI: barline positions are
    noisy and time-sig digits can sit slightly before or after the barline
    visually. A full-line scan finds every match robustly; snapping to the
    nearest barline then yields clean per-measure anchors.

    Returns
    -------
    list of dicts [{'x': x_pos, 'num': N, 'den': D, 'score': s,
                    'source': 'clef'|'barline'}]
        where x_pos is the anchor x at which this time signature takes
        effect. Sorted by x. At most one anchor per (barline x).
    """
    img_w = binary_img.shape[1]

    if clef_roi_x1 is None:
        clef_roi_x1 = int(img_w * 0.08)
    if clef_roi_x2 is None:
        clef_roi_x2 = int(img_w * 0.22)

    results = []

    # --- Phase 1: clef area ---
    staff_height = staff_system[4] - staff_system[0]
    y1 = max(0, staff_system[0] - int(dy * 0.5))
    y2 = min(binary_img.shape[0], staff_system[4] + int(dy * 0.5))
    clef_roi = binary_img[y1:y2, clef_roi_x1:clef_roi_x2]
    clef_match = _match_timesig_in_roi(clef_roi, staff_height, min_score=0.55)
    if clef_match is not None:
        (num, den), score, mx_in_roi, mw = clef_match
        # Absolute x of the time-sig digit center in the source image.
        # Used by the hollow-notehead time-sig filter to suppress false
        # positives on the digit shapes ("5" of 5/4, etc.).
        abs_match_x = clef_roi_x1 + mx_in_roi if mx_in_roi is not None else None
        results.append({'x': 0.0, 'num': num, 'den': den,
                        'score': score, 'source': 'clef',
                        'match_x': abs_match_x, 'match_w': mw})

    # --- Phase 2: full-line scan for mid-line changes ---
    # Start just after clef_roi_x2; scan all the way to the right edge.
    # A mid-line time sig change is rare but always announced with clear
    # stacked digits; threshold 0.70 keeps false positives from notes low.
    mid_matches = _scan_timesigs_full_line(
        binary_img, staff_system, dy,
        x1=clef_roi_x2, x2=img_w,
        min_score=0.70, nms_dy=2.0)

    # Snap each mid-line match to the nearest barline. A match with no
    # barline within 3*dy is dropped (likely a false positive).
    sorted_barlines = sorted(barlines) if barlines else []
    snap_tol = dy * 3.5
    snapped_by_bl = {}  # barline_x -> best match dict
    for m in mid_matches:
        mx = m['x']
        if not sorted_barlines:
            continue
        nearest = min(sorted_barlines, key=lambda b: abs(b - mx))
        if abs(nearest - mx) > snap_tol:
            continue
        prior = snapped_by_bl.get(nearest)
        if prior is None or m['score'] > prior['score']:
            snapped_by_bl[nearest] = {
                'x': float(nearest), 'num': m['num'], 'den': m['den'],
                'score': m['score'], 'source': 'barline'}

    results.extend(snapped_by_bl.values())
    results.sort(key=lambda r: r['x'])
    return results


_DIGIT_TEMPLATE_CACHE = None


def _load_digit_templates():
    """Load and cache binarized single-digit templates (digit_<N>.png).

    Returns dict {digit_int: binarized_template_array}.
    Currently we ship templates for 2 and 4 (sufficient for qudi page 1
    multi-rests "2", "2", "4", "42"). Adding more digits is just a matter
    of dropping digit_<N>.png into the template directory.
    """
    global _DIGIT_TEMPLATE_CACHE
    if _DIGIT_TEMPLATE_CACHE is not None:
        return _DIGIT_TEMPLATE_CACHE

    templates = {}
    for ext in ['*.png', '*.jpg']:
        for path in glob.glob(os.path.join(TEMPLATE_DIR, ext)):
            fname = os.path.splitext(os.path.basename(path))[0]
            if not fname.startswith('digit_'):
                continue
            parts = fname.split('_')
            if len(parts) != 2:
                continue
            try:
                d = int(parts[1])
            except ValueError:
                continue
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            _, bimg = cv2.threshold(img, 128, 255,
                                    cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            templates[d] = bimg
    _DIGIT_TEMPLATE_CACHE = templates
    return templates


def detect_multi_rest_count(binary_img, staff_system, seg_x1, seg_x2, dy):
    """Detect a multi-measure-rest numeral above a staff segment.

    A multi-measure rest in this engraving style is drawn as a thick
    horizontal bar centered on the middle staff line, with an Arabic
    numeral (typically 2-9, sometimes multi-digit) directly ABOVE the
    staff. The numeral indicates how many empty measures the bar
    represents.

    Parameters
    ----------
    binary_img : ndarray, full-page inverted binary image
        (notation = 255, background = 0)
    staff_system : tuple, the 5 staff-line y-coordinates for this staff
    seg_x1, seg_x2 : int, x-bounds of the segment to inspect
    dy : float, staff line spacing

    Returns
    -------
    int or None: number of measures the bar represents (>= 2),
        or None if no multi-measure rest is detected.
    """
    # 1. Confirm a thick horizontal bar on the middle staff line.
    sys = staff_system
    img_h, img_w = binary_img.shape
    mid_y = int((sys[1] + sys[3]) / 2)
    bar_y1 = max(0, mid_y - max(3, int(dy * 0.5)))
    bar_y2 = min(img_h, mid_y + max(3, int(dy * 0.5)))
    seg_x1 = max(0, int(seg_x1))
    seg_x2 = min(img_w, int(seg_x2))
    if seg_x2 - seg_x1 < dy * 3:
        return None
    bar_region = binary_img[bar_y1:bar_y2, seg_x1:seg_x2]
    if bar_region.size == 0:
        return None
    row_fills = np.mean(bar_region > 127, axis=1)
    thick_rows = int(np.sum(row_fills > 0.5))
    if thick_rows < max(4, int(dy * 0.3)):
        return None  # no multi-rest bar in this segment

    # 2. Find connected components above the staff (where the numeral sits).
    digit_y1 = max(0, sys[0] - int(dy * 4))
    digit_y2 = max(0, sys[0] - int(dy * 0.2))
    if digit_y2 <= digit_y1:
        return None
    pad = int(dy * 1.0)
    digit_x1 = max(0, seg_x1 - pad)
    digit_x2 = min(img_w, seg_x2 + pad)
    crop = binary_img[digit_y1:digit_y2, digit_x1:digit_x2]
    if crop.size == 0:
        return None

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        crop, connectivity=8)
    comps = []
    min_area = (dy * 0.5) ** 2
    min_h = dy * 0.8
    for li in range(1, num_labels):
        x, y, w, h, area = stats[li]
        if area < min_area or h < min_h:
            continue
        comps.append((x, y, w, h))
    if not comps:
        return None
    comps.sort(key=lambda c: c[0])  # left-to-right

    # 3. Match each component against single-digit templates.
    templates = _load_digit_templates()
    if not templates:
        return None

    digits = []
    for cx, cy, cw, ch in comps:
        comp_crop = crop[cy:cy + ch, cx:cx + cw]
        best_d = None
        best_score = 0.55  # floor
        for d, tmpl in templates.items():
            th, tw = tmpl.shape
            # Resize template to match component height
            scale = ch / th if th > 0 else 1.0
            new_h = max(5, int(th * scale))
            new_w = max(5, int(tw * scale))
            if new_h > comp_crop.shape[0] or new_w > comp_crop.shape[1]:
                # template larger than component crop after rescale: use
                # single-pixel match by resizing template down
                new_h = min(new_h, comp_crop.shape[0])
                new_w = min(new_w, comp_crop.shape[1])
                if new_h < 5 or new_w < 5:
                    continue
            resized = cv2.resize(tmpl, (new_w, new_h),
                                 interpolation=cv2.INTER_AREA)
            res = cv2.matchTemplate(comp_crop, resized, cv2.TM_CCOEFF_NORMED)
            if res.size == 0:
                continue
            mv = float(np.max(res))
            if mv > best_score:
                best_score = mv
                best_d = d
        if best_d is None:
            return None  # unrecognized component → bail
        digits.append(best_d)

    # 4. Combine digits left-to-right into integer
    n = 0
    for d in digits:
        n = n * 10 + d
    if n < 2:
        return None  # multi-rest must represent >= 2 measures
    return n


