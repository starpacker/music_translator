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

# --- Template paths ---
# New comprehensive template folder
TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "..", "template")
# Legacy template folders (fallback)
PICTURE_DIR = os.path.join(os.path.dirname(__file__),
    "..", "repo", "translate-staff-to-simple-musical-notation-master",
    "score_recognition_v4", "picture")
PICTURE_EXPAND_DIR = os.path.join(os.path.dirname(__file__),
    "..", "repo", "translate-staff-to-simple-musical-notation-master",
    "score_recognition_v4", "picture_expand")


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
# 1. BARLINE DETECTION
# ============================================================
def detect_barlines(binary_img, staff_systems, dy):
    """
    Detect vertical barlines using template matching with the bar template.
    Returns list of barline x-positions for each staff system.
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
        
        filtered = []
        for bx, score in deduped:
            if not filtered or (bx - filtered[-1]) > dy * 18:
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
def detect_accidentals_global(binary_img, staff_systems, dy):
    """
    Detect accidentals globally in each staff region using multi-scale template matching.
    Uses ALL available sharp/flat/natural templates from the new template/ folder.
    
    Returns list of dicts: {'x': x, 'y': y, 'type': '#'/'b'/'n', 'score': score, 'system_idx': idx}
    Natural signs ('n') are used to cancel accidental persistence within a measure.
    """
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
        
        # Skip clef area (first ~18% of width)
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
                    
                    threshold = 0.60
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
    used_accidentals = set()
    
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
                used_accidentals.add(acc_idx)
    
    result = {k: v[0] for k, v in accidentals_map.items()}
    print(f"   Assigned {len(result)} accidentals to notes "
          f"({sum(1 for v in result.values() if v == '#')} sharps, "
          f"{sum(1 for v in result.values() if v == 'b')} flats, "
          f"{sum(1 for v in result.values() if v == 'n')} naturals)")
    return result


# Legacy per-note approach (kept as fallback)
def detect_accidentals(binary_img, staff_systems, dy, noteheads):
    """
    Detect sharp (#) and flat (b) symbols near noteheads using per-note search.
    Uses multiple templates from the new template/ folder.
    """
    sharp_templates = _load_all_templates_by_prefix("sharp_", TEMPLATE_DIR)
    flat_templates = _load_all_templates_by_prefix("flat_", TEMPLATE_DIR)
    
    # Also from legacy
    for name in ["sharp_1.jpg"]:
        t = _load_template(name, PICTURE_DIR)
        if t is not None:
            sharp_templates.append((name + "_legacy", t))
    for name in ["flat_1.jpg"]:
        t = _load_template(name, PICTURE_DIR)
        if t is not None:
            flat_templates.append((name + "_legacy", t))
    
    print(f"   Per-note: {len(sharp_templates)} sharp templates, {len(flat_templates)} flat templates")
    
    accidentals = {}
    img_h, img_w = binary_img.shape
    
    for note in noteheads:
        nx = note['x']
        ny = note['y_center']
        
        search_x1 = max(0, nx - int(dy * 3.5))
        search_x2 = nx + int(dy * 0.3)
        search_y1 = max(0, ny - int(dy * 2.0))
        search_y2 = min(img_h, ny + int(dy * 2.0))
        
        roi = binary_img[search_y1:search_y2, search_x1:search_x2]
        roi_h, roi_w = roi.shape
        if roi_h < 5 or roi_w < 5:
            continue
        
        best_score = 0.0
        best_acc = None
        best_threshold = 0.45
        
        for tname, template in sharp_templates:
            score = _match_accidental_template(roi, template, dy, roi_h, roi_w)
            if score > best_score and score > best_threshold:
                best_score = score
                best_acc = '#'
        
        for tname, template in flat_templates:
            score = _match_accidental_template(roi, template, dy, roi_h, roi_w)
            if score > best_score and score > best_threshold:
                best_score = score
                best_acc = 'b'
        
        if best_acc is not None:
            key = (note['x'] + note['w'] // 2, note['y_center'])
            if key not in accidentals or accidentals[key][1] < best_score:
                accidentals[key] = (best_acc, best_score)
    
    result = {k: v[0] for k, v in accidentals.items()}
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
    rest_templates = {
        'stop_4.jpg': 1.0,   # quarter rest = 1 beat
    }
    
    rests = []
    img_h, img_w = binary_img.shape
    
    for sys_idx, system in enumerate(staff_systems):
        staff_height = system[4] - system[0]
        margin = int(dy * 0.5)
        search_y1 = max(0, system[0] - margin)
        search_y2 = min(img_h, system[4] + margin)
        
        for tname, duration in rest_templates.items():
            # Try from multiple directories
            template = None
            for d in [TEMPLATE_DIR, PICTURE_EXPAND_DIR, PICTURE_DIR]:
                template = _load_template(tname, d)
                if template is not None:
                    break
            if template is None:
                continue
            
            th_orig, tw_orig = template.shape
            ideal_scale = (staff_height * 0.5) / float(th_orig) if th_orig > 0 else 0.5
            
            for scale_factor in [ideal_scale * 0.9, ideal_scale * 1.1]:
                new_h = int(th_orig * scale_factor)
                new_w = int(tw_orig * scale_factor)
                
                if new_h < 5 or new_w < 5:
                    continue
                
                roi = binary_img[search_y1:search_y2, :]
                roi_h, roi_w = roi.shape
                
                if new_h >= roi_h or new_w >= roi_w:
                    continue
                
                resized = cv2.resize(template, (new_w, new_h), interpolation=cv2.INTER_AREA)
                res = cv2.matchTemplate(roi, resized, cv2.TM_CCOEFF_NORMED)
                
                threshold = 0.55  # quarter rest threshold
                loc = np.where(res >= threshold)
                
                for pt in zip(*loc[::-1]):
                    rx = pt[0] + new_w // 2
                    ry = search_y1 + pt[1] + new_h // 2
                    score = res[pt[1], pt[0]]
                    
                    rests.append({
                        'x': rx,
                        'y_center': ry,
                        'system_idx': sys_idx,
                        'duration': duration,
                        'score': score,
                        'type': tname,
                    })
    
    # Also try stop_8 (eighth rest = 0.5 beat) for half-beat rests
    for sys_idx, system in enumerate(staff_systems):
        staff_height = system[4] - system[0]
        margin = int(dy * 0.5)
        search_y1 = max(0, system[0] - margin)
        search_y2 = min(img_h, system[4] + margin)
        
        for tname in ['stop_8.jpg']:
            template = None
            for d in [TEMPLATE_DIR, PICTURE_EXPAND_DIR]:
                template = _load_template(tname, d)
                if template is not None:
                    break
            if template is None:
                continue
            
            th_orig, tw_orig = template.shape
            ideal_scale = (staff_height * 0.35) / float(th_orig) if th_orig > 0 else 0.5
            
            for scale_factor in [ideal_scale * 0.85, ideal_scale, ideal_scale * 1.15]:
                new_h = int(th_orig * scale_factor)
                new_w = int(tw_orig * scale_factor)
                
                if new_h < 5 or new_w < 5:
                    continue
                
                roi = binary_img[search_y1:search_y2, :]
                roi_h, roi_w = roi.shape
                
                if new_h >= roi_h or new_w >= roi_w:
                    continue
                
                resized = cv2.resize(template, (new_w, new_h), interpolation=cv2.INTER_AREA)
                res = cv2.matchTemplate(roi, resized, cv2.TM_CCOEFF_NORMED)
                
                threshold = 0.50
                loc = np.where(res >= threshold)
                
                for pt in zip(*loc[::-1]):
                    rx = pt[0] + new_w // 2
                    ry = search_y1 + pt[1] + new_h // 2
                    score = res[pt[1], pt[0]]
                    
                    rests.append({
                        'x': rx,
                        'y_center': ry,
                        'system_idx': sys_idx,
                        'duration': 0.5,
                        'score': score,
                        'type': tname,
                    })
    
    if rests:
        rests = _nms_rests(rests, dy * 2.0)
    
    return rests


def _nms_rests(rests, min_dist_x, min_dist_y=None):
    """Remove duplicate rest detections."""
    if min_dist_y is None:
        min_dist_y = min_dist_x * 4
    sorted_rests = sorted(rests, key=lambda r: r['score'], reverse=True)
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


