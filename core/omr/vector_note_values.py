"""Limited, font-verified evidence about noteheads and connected beams.

No duration is inferred from an absent beam: flags, partial beams and other
fonts may be invisible to these checks. The source score is never modified.
"""
from __future__ import annotations

import pymupdf
from .vector_dynamics import _ANONYMOUS, _is_engraver

_HEAD_KIND = {0xCF: 'filled', 0xFA: 'half', 0x77: 'whole'}
_FILLED_TYPES = {'quarter', 'eighth', '16th', '32nd', '64th', '128th', '256th', '512th', '1024th'}
_BEAMS_BY_TYPE = {'quarter': 0, 'eighth': 1, '16th': 2, '32nd': 3, '64th': 4,
                  '128th': 5, '256th': 6, '512th': 7, '1024th': 8}


def verified_notehead_glyphs(doc, page):
    """Return glyphs only when codepoint and embedded music-font GID agree."""
    fonts, seen = [], set()
    for entry in page.get_fonts(full=True):
        xref, base = entry[0], entry[3]
        if xref in seen or not (_ANONYMOUS.fullmatch(base or '')
                                or 'engraverfontset' in (base or '').lower()):
            continue
        seen.add(xref)
        try:
            data = doc.extract_font(xref)[3]
            if not data:
                return []
            fonts.append(pymupdf.Font(fontbuffer=data))
        except Exception:
            return []
    if not any(_is_engraver(font) for font in fonts):
        return []
    result = []
    for span in page.get_texttrace():
        name = (span.get('font') or '').lower()
        opacity = span.get('opacity')
        if ((name and not name.startswith('engraverfontset'))
                or (opacity is not None and opacity < .9)):
            continue
        for cp, glyph, (x, y), bbox in span['chars']:
            code = cp - 0xF000 if 0xF000 <= cp <= 0xF0FF else cp
            if code not in _HEAD_KIND:
                continue
            matching = [font for font in fonts if glyph and font.has_glyph(0xF000 + code) == glyph]
            if len(matching) == 1 and _is_engraver(matching[0]):
                result.append({'x': float(x), 'y': float(y), 'size': float(span['size']),
                               'bbox': tuple(float(value) for value in bbox), 'kind': _HEAD_KIND[code]})
    return result


def _beam_vertical_range(drawing, x):
    rect = drawing['rect']
    x = min(max(x, rect.x0), rect.x1)
    ys = []
    for _, a, b in drawing['items']:
        if abs(a.x - b.x) < .001:
            if abs(x - a.x) < .45:
                ys.extend((a.y, b.y))
        elif min(a.x, b.x) - .001 <= x <= max(a.x, b.x) + .001:
            ys.append(a.y + (b.y - a.y) * (x - a.x) / (b.x - a.x))
    return (min(ys), max(ys)) if ys else None


def _is_beam_polygon(drawing, staff, left, right):
    rect, items = drawing['rect'], drawing['items']
    if (drawing['type'] != 'f' or len(items) != 4 or not all(item[0] == 'l' for item in items)
            or rect.width < 3 or rect.height > 2 * staff.line_spacing
            or rect.x0 < left - .5 or rect.x1 > right + .5
            or not drawing.get('fill') or max(drawing['fill']) >= .15
            or (drawing.get('fill_opacity') is not None and drawing['fill_opacity'] < .9)):
        return False
    # A filled beam is a closed thin parallelogram, not an arbitrary black
    # polygon that happens to touch two stems.
    if any(abs(items[i][2] - items[(i + 1) % 4][1]) > .2 for i in range(4)):
        return False
    long_edges = [(a, b) for _, a, b in items if abs(a.x - b.x) >= .8 * rect.width]
    if len(long_edges) != 2:
        return False
    slopes = [(b.y - a.y) / (b.x - a.x) for a, b in long_edges]
    ends = [(a, b) for _, a, b in items if abs(a.x - b.x) < .8 * rect.width]
    return (abs(slopes[0] - slopes[1]) <= .15 and len(ends) == 2
            and all(.3 <= abs(a - b) <= staff.line_spacing * .8 for a, b in ends))


def _beam_meets_stem(drawing, stem, spacing):
    x, low, high, note_y, up = stem
    rect = drawing['rect']
    if not rect.x0 - .4 <= x <= rect.x1 + .4:
        return False
    bounds = _beam_vertical_range(drawing, x)
    if bounds is None:
        return False
    minimum, maximum = bounds
    if maximum < low - .8 or minimum > high + .8:
        return False
    tip = low if up else high
    if up:
        return minimum >= tip - 1.5 and maximum <= tip + 1.8 * spacing and maximum <= note_y - .5 * spacing
    return maximum <= tip + 1.5 and minimum >= tip - 1.8 * spacing and minimum >= note_y + .5 * spacing


def compare_note_values(measure, staff, left, right, matched, glyphs, drawings):
    """Compare supported heads and count beams joining two verified stems."""
    events = list(measure.recurse().notes)
    if matched is None or len(events) != len(matched):
        return {'status': 'unavailable', 'reason': 'notehead_positions_not_matched'}
    heads = [g for g in glyphs if left <= g['x'] < right
             and staff.top - 6 * staff.line_spacing <= g['y'] <= staff.bottom + 6 * staff.line_spacing
             and 2.4 * staff.line_spacing <= g['size'] <= 4.6 * staff.line_spacing]
    stems = [(d['rect'].x0, d['rect'].y0, d['rect'].y1) for d in drawings
             if d['type'] == 's' and len(d['items']) == 1 and d['items'][0][0] == 'l'
             and d['rect'].width < .2 and d['rect'].height > 4 and (d.get('width') or 0) < 1.4
             and (d.get('stroke_opacity') is None or d['stroke_opacity'] >= .9)
             and left < d['rect'].x0 < right]
    beams, seen_beams = [], set()
    for drawing in drawings:
        if not _is_beam_polygon(drawing, staff, left, right):
            continue
        signature = tuple(round(v, 2) for v in drawing['rect'])
        if signature not in seen_beams:
            beams.append(drawing)
            seen_beams.add(signature)
    shape_agree = shape_unknown = 0
    shape_conflicts, grace_size_reviews = [], []
    grace_size_consistent = 0
    stem_by_event = {}
    for event_index, (event, (xs, offset, grace)) in enumerate(zip(events, matched)):
        selected = [g for g in heads if any(abs(g['x'] - x) < .04 for x in xs)]
        if len(selected) != len(xs):
            shape_unknown += len(xs)
            continue
        typ = event.duration.type
        for glyph in selected:
            allowed = (_FILLED_TYPES if glyph['kind'] == 'filled'
                       else {'half'} if glyph['kind'] == 'half' else {'whole'})
            if typ in allowed:
                shape_agree += 1
            elif typ in _FILLED_TYPES | {'whole', 'half'}:
                shape_conflicts.append({'event_index': event_index + 1, 'offset_quarters': str(offset),
                                        'musicxml_type': typ, 'pdf_head_kind': glyph['kind'],
                                        'x': round(glyph['x'], 2), 'y': round(glyph['y'], 2)})
            else:
                shape_unknown += 1
            ratio = glyph['size'] / staff.line_spacing
            # A cue-sized note is not necessarily a grace note. Use size only
            # to identify review locations; it is never used to rewrite time.
            small = ratio <= 2.75
            regular = ratio >= 3.5
            if (small and not grace) or (regular and grace):
                grace_size_reviews.append({'event_index': event_index + 1, 'offset_quarters': str(offset),
                                           'musicxml_is_grace': bool(grace),
                                           'font_size_in_staff_spacings': round(ratio, 2),
                                           'x': round(glyph['x'], 2), 'y': round(glyph['y'], 2)})
            elif (small and grace) or (regular and not grace):
                grace_size_consistent += 1
        if len(xs) != 1 or glyph['kind'] != 'filled':
            continue
        glyph = selected[0]
        bx0, _, bx1, _ = glyph['bbox']
        found = [(sx, low, high) for sx, low, high in stems
                 if min(abs(sx - (bx0 + .24)), abs(sx - (bx1 - .24))) < .6
                 and min(abs(low - glyph['y']), abs(high - glyph['y'])) < .8
                 and high - low > glyph['size'] * .35
                 and min(abs(sx - left), abs(sx - right)) > .8]
        if len(found) == 1:
            sx, low, high = found[0]
            up = abs(high - glyph['y']) < abs(low - glyph['y'])
            stem_by_event[event_index] = (sx, low, high, glyph['y'], up)
    beam_counts = {i: 0 for i in stem_by_event}
    for beam in beams:
        rect = beam['rect']
        left_stems = [i for i, stem in stem_by_event.items() if abs(stem[0] - rect.x0) <= .7
                      and _beam_meets_stem(beam, stem, staff.line_spacing)]
        right_stems = [i for i, stem in stem_by_event.items() if abs(stem[0] - rect.x1) <= .7
                       and _beam_meets_stem(beam, stem, staff.line_spacing)]
        if len(left_stems) != 1 or len(right_stems) != 1 or left_stems[0] == right_stems[0]:
            continue
        for i, stem in stem_by_event.items():
            if _beam_meets_stem(beam, stem, staff.line_spacing):
                beam_counts[i] += 1
    beam_evidence = []
    for i, count in beam_counts.items():
        if not count:
            continue
        event = events[i]
        expected = _BEAMS_BY_TYPE.get(event.duration.type)
        if expected is None:
            continue
        beam_evidence.append({'event_index': i + 1, 'offset_quarters': str(matched[i][1]),
                              'musicxml_type': event.duration.type, 'connected_beams_at_least': count,
                              'stem_x': round(stem_by_event[i][0], 2),
                              'note_y': round(stem_by_event[i][3], 2), 'is_grace': bool(matched[i][2]),
                              'status': ('more_beams_than_musicxml_review' if count > expected else
                                         'fewer_connected_beams_review' if count < expected else
                                         'connected_beam_lower_bound_consistent')})
    return {'status': 'partial_evidence', 'head_shapes_agree': shape_agree,
            'head_shapes_unverified': shape_unknown, 'head_shape_conflicts': shape_conflicts,
            'grace_size_consistent': grace_size_consistent, 'grace_size_reviews': grace_size_reviews,
            'beam_evidence': beam_evidence}
