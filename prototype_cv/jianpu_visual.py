"""
jianpu_visual.py
Render measures as proper visual Jianpu (简谱) onto a PIL image.

Conventions modeled after
``repo/translate-staff-to-simple-musical-notation-master/score_recognition_v4``:

  - Note      : digit drawn as text (e.g. "5"); chord = stack of digits
  - Octave    : small filled dots above (high) or below (low) the digit
  - Sharp/Flat: "#" / "b" prefix character before the digit
  - Eighth    : single horizontal underline (减时线) under the digit
  - Sixteenth : double underline
  - Dotted    : middle dot · drawn at digit baseline (or as text ".")
  - Half      : trailing "  -"  characters
  - Dotted half: "  -  -"
  - Whole     : "  -  -  -"
  - Rest      : digit "0" with the same rhythm marks
  - Barline   : vertical red line between measures

The renderer runs over each measure once: it lays out events left to
right inside the measure's pixel range, drawing each event with the
duration glyphs appropriate to its rhythm. Chord stacking mirrors the
reference repo (digits stacked vertically; rhythm marks shared).
"""
from PIL import Image, ImageDraw, ImageFont
import os


# ── Font discovery ──────────────────────────────────────────────────
def _load_font(size):
    candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/Arial.ttf",
        "C:/Windows/Fonts/consola.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


# ── Note number conversion ──────────────────────────────────────────
def _split_pitch(pitch):
    """Split pitch like "3'", "5,," into (digit_char, octave_int).

    Returns (digit, octave) where octave is +N for high (' marks) or
    -N for low (, marks). 0 = middle octave.
    """
    digit = pitch[0]
    rest = pitch[1:]
    octave = 0
    for ch in rest:
        if ch == "'":
            octave += 1
        elif ch == ",":
            octave -= 1
    return digit, octave


# ── Rhythm category ─────────────────────────────────────────────────
def _rhythm_marks(duration):
    """Return (underlines, dotted, trailing_dashes) for a duration.

    underlines     : 0 for quarter, 1 for eighth, 2 for sixteenth
    dotted         : True if a · should be drawn after the digit
    trailing_dashes: number of "-" glyphs to draw after the digit
                     (0 for quarter and shorter; 1 for half; 2 for
                      dotted half; 3 for whole)
    """
    # tuplets — render as quarter underline (no precise glyph)
    if abs(duration - 1.0 / 6.0) < 0.05 or abs(duration - 1.0 / 12.0) < 0.05:
        return 2, False, 0
    if abs(duration - 1.0 / 3.0) < 0.05:
        return 1, False, 0
    if abs(duration - 2.0 / 3.0) < 0.05:
        return 0, False, 0

    if abs(duration - 0.25) < 0.05:
        return 2, False, 0
    if abs(duration - 0.5) < 0.05:
        return 1, False, 0
    if abs(duration - 0.75) < 0.05:
        return 1, True, 0
    if abs(duration - 1.0) < 0.05:
        return 0, False, 0
    if abs(duration - 1.5) < 0.05:
        return 0, True, 0
    if abs(duration - 2.0) < 0.05:
        return 0, False, 1
    if abs(duration - 3.0) < 0.05:
        return 0, False, 2
    if abs(duration - 4.0) < 0.05:
        return 0, False, 3
    return 0, False, 0


# ── Event glyph rendering ───────────────────────────────────────────
def _draw_digit_with_marks(draw, x, baseline_y, text, octave, underlines,
                           dotted, font, color, glyph_w, glyph_h):
    """Draw one digit + its octave dots + underlines + dot.

    baseline_y is the y of the digit's baseline (top of font box).
    Returns the width consumed (for advancing x).
    """
    # Digit
    draw.text((x, baseline_y), text, font=font, fill=color)

    # Octave dots
    dot_radius = max(1, glyph_h // 12)
    dot_gap = max(2, glyph_h // 8)
    if octave > 0:
        for i in range(octave):
            cy = baseline_y - dot_gap * (i + 1)
            cx = x + glyph_w // 2
            draw.ellipse((cx - dot_radius, cy - dot_radius,
                          cx + dot_radius, cy + dot_radius), fill=color)
    elif octave < 0:
        for i in range(-octave):
            cy = baseline_y + glyph_h + dot_gap * i + dot_radius
            cx = x + glyph_w // 2
            draw.ellipse((cx - dot_radius, cy - dot_radius,
                          cx + dot_radius, cy + dot_radius), fill=color)

    # 减时线 (underlines below digit)
    if underlines > 0:
        line_gap = max(2, glyph_h // 10)
        line_y0 = baseline_y + glyph_h + 1
        for i in range(underlines):
            ly = line_y0 + i * line_gap
            draw.line((x, ly, x + glyph_w, ly), fill=color, width=1)

    # 附点 (dot after digit)
    if dotted:
        # small filled circle at mid-digit-height, just past the digit
        cy = baseline_y + glyph_h // 2
        cx = x + glyph_w + dot_radius + 1
        draw.ellipse((cx - dot_radius, cy - dot_radius,
                      cx + dot_radius, cy + dot_radius), fill=color)

    return glyph_w + (3 * dot_radius if dotted else 0)


def _draw_event(draw, event, x, baseline_y, accidentals_map,
                font, glyph_w, glyph_h, color, persistent_accs,
                dy_for_acc, key_sig=None):
    """Draw one event (note_unit or rest); return new x cursor."""
    if event['type'] == 'rest':
        underlines, dotted, dashes = _rhythm_marks(event.get('duration', 1.0))
        consumed = _draw_digit_with_marks(draw, x, baseline_y, "0", 0,
                                          underlines, dotted, font, color,
                                          glyph_w, glyph_h)
        x_after = x + consumed
        # Trailing dashes for half/whole rests
        for k in range(dashes):
            x_after += glyph_w // 2
            draw.text((x_after, baseline_y), "-", font=font, fill=color)
            x_after += glyph_w
        return x_after + glyph_w // 3

    if event.get('type') == 'multi_rest_count':
        label = f"×{event['count']}"
        draw.text((x, baseline_y), label, font=font, fill=color)
        return x + glyph_w * len(label)

    unit = event['unit']
    duration = unit['duration']
    underlines, dotted, dashes = _rhythm_marks(duration)
    notes = unit['notes']

    # Accidental: take from the highest-pitched note (top of chord)
    chord_w = glyph_w
    digit_x = x

    # Accidental prefix (look up first note for this chord)
    n0 = notes[0]
    cx = n0['x'] + n0.get('w', 0) // 2
    cy = n0['y_center']
    acc = accidentals_map.get((cx, cy), '')

    # Apply persistence and key signature logic (mirrors jianpu_formatter)
    note_x = n0['x']
    pitch = n0.get('pitch', '')
    base_char = pitch[0] if pitch else ''
    pitch_key = pitch
    max_persist_gap = dy_for_acc * 5.0

    if acc == 'n':
        if persistent_accs is not None:
            persistent_accs[pitch_key] = ('n', note_x)
        acc = 'H'  # Natural rendered as 'H' like the reference repo
    elif acc:
        if persistent_accs is not None:
            persistent_accs[pitch_key] = (acc, note_x)
    else:
        if persistent_accs is not None and pitch_key in persistent_accs:
            stored_acc, stored_x = persistent_accs[pitch_key]
            if abs(note_x - stored_x) <= max_persist_gap:
                if stored_acc == 'n':
                    acc = ''
                else:
                    acc = stored_acc
                persistent_accs[pitch_key] = (stored_acc, note_x)
            else:
                if key_sig and base_char.isdigit() and int(base_char) in key_sig['notes']:
                    acc = key_sig['type']
        else:
            if key_sig and base_char.isdigit() and int(base_char) in key_sig['notes']:
                acc = key_sig['type']

    if acc:
        draw.text((digit_x, baseline_y), acc, font=font, fill=color)
        digit_x += glyph_w

    # Stack chord notes vertically (top = highest pitch).
    # notes are sorted by y_center ascending in NoteUnit; reverse so
    # top of chord is the highest staff position (smallest y).
    sorted_notes = sorted(notes, key=lambda n: n['y_center'])

    # Compute baseline for each chord note. Highest gets the topmost y.
    n_chord = len(sorted_notes)
    for ci, n in enumerate(sorted_notes):
        digit_char, octave = _split_pitch(n['pitch'])
        # ci=0 is highest pitch → top; subsequent stack downward
        slot_y = baseline_y - (n_chord - 1 - ci) * glyph_h \
                 if False else baseline_y + ci * glyph_h
        # Actually highest pitch should be highest y on screen (smaller y).
        # ci=0 (smallest y_center, highest pitch) → topmost slot.
        slot_y = baseline_y + ci * glyph_h
        # Apply rhythm marks only to the BOTTOM digit so underlines
        # don't run through the chord; share dotted on bottom too.
        u = underlines if ci == n_chord - 1 else 0
        d = dotted if ci == n_chord - 1 else False
        _draw_digit_with_marks(draw, digit_x, slot_y, digit_char, octave,
                               u, d, font, color, glyph_w, glyph_h)

    x_after = digit_x + glyph_w + (glyph_w // 2 if dotted else 0)

    # Trailing dashes for half/whole notes — render at bottom digit's y
    for k in range(dashes):
        x_after += glyph_w // 2
        draw.text((x_after, baseline_y + (n_chord - 1) * glyph_h),
                  "-", font=font, fill=color)
        x_after += glyph_w

    return x_after + glyph_w // 3


# ── Public renderer ─────────────────────────────────────────────────
def render_measure_strip(draw, measures, barlines, x_left, x_right,
                          y_top, strip_h, accidentals_map, dy, key_sig=None):
    """Render one staff's worth of measures into the rectangle
    (x_left..x_right, y_top..y_top+strip_h). The strip is assumed
    to be a clear background colour already.
    """
    glyph_h = max(12, int(strip_h * 0.45))
    glyph_w = int(glyph_h * 0.6)
    font = _load_font(glyph_h)
    color = (180, 0, 0)
    barline_color = (160, 160, 160)

    # Compute baseline y so digits sit centred vertically with room for
    # octave dots above and underlines below.
    baseline_y = y_top + (strip_h - glyph_h) // 2 - glyph_h // 4

    # Prepare measure boundary list
    boundaries = [x_left] + [b for b in barlines if x_left < b < x_right] \
                 + [x_right]

    persistent_accs = {}

    for mi in range(len(measures)):
        if mi >= len(boundaries) - 1:
            break
        seg_left = boundaries[mi]
        seg_right = boundaries[mi + 1]
        seg_w = seg_right - seg_left

        # Draw measure barline (skip the leftmost edge)
        if mi > 0:
            draw.line((seg_left, y_top + 2, seg_left, y_top + strip_h - 2),
                      fill=barline_color, width=1)

        # Lay events evenly inside the segment with small left margin
        events = measures[mi]
        if not events:
            continue
        # Reset persistent accidentals per measure
        persistent_accs.clear()

        n_ev = len(events)
        # Left/right inner margin
        inner_margin = max(4, glyph_w // 2)
        avail = max(1, seg_w - 2 * inner_margin)
        # Even spacing, anchored at left edge of each slot
        slot_w = avail / n_ev

        for ei, ev in enumerate(events):
            ex = seg_left + inner_margin + int(ei * slot_w)
            _draw_event(draw, ev, ex, baseline_y, accidentals_map, font,
                        glyph_w, glyph_h, color, persistent_accs, dy,
                        key_sig=key_sig)


def render_full_image(image_path, staff_data, accidentals_map, dy,
                       output_path, key_sig=None):
    """Render the full annotated image: each staff system gets a
    visual jianpu strip drawn beneath it (red text + underlines + dots).
    """
    pil_orig = Image.open(image_path).convert('RGB')
    orig_w, orig_h = pil_orig.size

    strip_h = int(dy * 3.0)
    n_staves = len(staff_data)
    new_h = orig_h + n_staves * strip_h
    out = Image.new('RGB', (orig_w, new_h), (255, 255, 255))

    cursor = 0
    prev_copy = 0
    draw = ImageDraw.Draw(out)

    for si, sd in enumerate(staff_data):
        sys_info = sd['system']
        barlines = sd['barlines']
        measures = sd['measures']

        cut_y = sys_info[4] + int(dy * 1.5)
        if si + 1 < n_staves:
            next_top = staff_data[si + 1]['system'][0]
            cut_y = min(cut_y, next_top - int(dy * 0.5))
        else:
            cut_y = min(cut_y, orig_h)

        strip = pil_orig.crop((0, prev_copy, orig_w, cut_y))
        out.paste(strip, (0, cursor))
        cursor += strip.height

        # Light blue background for the annotation strip
        draw.rectangle((0, cursor, orig_w, cursor + strip_h),
                       fill=(245, 245, 255))

        render_measure_strip(draw, measures, barlines, 0, orig_w,
                              cursor, strip_h, accidentals_map, dy,
                              key_sig=key_sig)
        cursor += strip_h
        prev_copy = cut_y

    # Tail
    if prev_copy < orig_h:
        tail = pil_orig.crop((0, prev_copy, orig_w, orig_h))
        if cursor + tail.height <= new_h:
            out.paste(tail, (0, cursor))

    out.save(output_path)
