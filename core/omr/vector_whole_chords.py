"""Recover unambiguous whole-note chords from embedded Maestro PDF glyphs.

This deliberately handles only treble staves without a key signature and bars
containing one whole-note onset. All other bars remain OMR results. Positions
come from PDF barlines, never from OMR measure durations or note-count pairing.
"""
from pathlib import Path
import os
import xml.etree.ElementTree as ET

from .multirest_collapse import page_system_slot_kinds, _read_state

WHOLE = ''
OTHER_NOTES = {'', '', ''}
ACCIDENTALS = {'': -1, '': 1, '': 0}


def whole_chords_on_page(page):
    geometry = []
    page_system_slot_kinds(page, _geometry=geometry)
    glyphs = [(c['c'], *c['origin'])
              for b in page.get_text('rawdict')['blocks']
              for line in b.get('lines', []) for span in line['spans']
              if span['font'] == 'Maestro'
              for c in span['chars']]
    result = {}
    offset = 0
    for lines, slots in geometry:
        spacing = (lines[-1] - lines[0]) / 4
        # Maestro treble G clef is anchored on the second line from bottom.
        staff = [(c, x, y) for c, x, y in glyphs
                 if lines[0] - 3 * spacing < y < lines[-1] + 3 * spacing]
        if not any(c == '' for c, x, y in staff):
            offset += len(slots)
            continue
        heads = [(c, x, y) for c, x, y in staff if c in OTHER_NOTES | {WHOLE}]
        first_x = min((x for c, x, y in heads), default=1e9)
        # Key signatures precede the notes; inline accidentals are at most
        # 22pt left of this score's first onset. Otherwise defer to OMR.
        key_signature = any(c in ACCIDENTALS and x < first_x - 22
                            for c, x, y in staff)
        for i, (left, right) in enumerate(slots):
            notes = [(c, x, y) for c, x, y in heads if left <= x < right]
            if key_signature or not notes or any(c != WHOLE for c, x, y in notes):
                continue
            # Adjacent seconds have horizontally displaced noteheads.
            if max(x for c, x, y in notes) - min(x for c, x, y in notes) > 9:
                continue
            # Dots, rests or additional rhythmic symbols make duration uncertain.
            if any(c in {'', '', '', ''}
                   and left <= x < right for c, x, y in staff):
                continue
            pitches = []
            used = set()
            for c, x, y in notes:
                step = round((lines[-1] - y) / (spacing / 2))
                if abs(lines[-1] - y - step * spacing / 2) > .5:
                    break
                # Bottom treble line is E4 (diatonic index 4*7+2).
                octave, degree = divmod(30 + step, 7)
                accidental = [(a, ax, ay) for a, ax, ay in staff
                              if a in ACCIDENTALS and x - 22 <= ax < x
                              and x - ax <= 22 and abs(ay - y) < .6]
                if len(accidental) > 1:
                    break
                alter = ACCIDENTALS[accidental[0][0]] if accidental else 0
                if accidental:
                    used.add(accidental[0])
                pitches.append(('CDEFGAB'[degree], alter, octave))
            else:
                all_acc = {(a, x, y) for a, x, y in staff
                           if a in ACCIDENTALS and min(nx for _, nx, _ in notes) - 22 <= x
                           < max(nx for _, nx, _ in notes)}
                if all_acc == used and len(set(pitches)) == len(pitches):
                    result[offset + i] = sorted(pitches, key=lambda p: (p[2], 'CDEFGAB'.index(p[0])))
        offset += len(slots)
    return result, offset


def recover_vector_whole_chords(pdf_path, mxl_path, page_index=0, log=None):
    if Path(pdf_path).suffix.lower() != '.pdf':
        return 0
    import pymupdf
    with pymupdf.open(pdf_path) as doc:
        chords, count = whole_chords_on_page(doc[page_index])
    tree = ET.parse(mxl_path)
    parts = tree.findall('./part')
    if len(parts) != 1:
        return 0
    measures = parts[0].findall('measure')
    # Existing collapsed caches have one measure per source slot. On fresh
    # mismatched OMR output, refuse positional replacement rather than shift.
    if len(measures) != count:
        return 0
    divisions, beats, beat_type = 1, 4, 4
    changed = 0
    for index, measure in enumerate(measures):
        state = _read_state(measure)
        divisions, beats, beat_type = (a if a is not None else b
            for a, b in zip(state, (divisions, beats, beat_type)))
        if index not in chords or (beats, beat_type) != (4, 4):
            continue
        expected = chords[index]
        actual = [(n.findtext('pitch/step'), int(n.findtext('pitch/alter', '0')),
                   int(n.findtext('pitch/octave', '0'))) for n in measure.findall('note')]
        if actual == expected and all(n.findtext('duration') == str(4 * divisions)
                                     for n in measure.findall('note')) and all(
                (n.find('chord') is not None) == bool(j)
                for j, n in enumerate(measure.findall('note'))) and not any(
                c.tag in {'forward', 'backup'} for c in measure):
            continue
        # Keep metadata and barlines; notes and time-position directives are
        # replaced as a unit so phantom forward/backup cannot add extra beats.
        for child in list(measure):
            if child.tag in {'note', 'forward', 'backup'}:
                measure.remove(child)
        insertion = next((i for i, c in enumerate(measure) if c.tag == 'barline'), len(measure))
        for j, (step, alter, octave) in enumerate(expected):
            n = ET.Element('note')
            if j:
                ET.SubElement(n, 'chord')
            pitch = ET.SubElement(n, 'pitch')
            ET.SubElement(pitch, 'step').text = step
            if alter:
                ET.SubElement(pitch, 'alter').text = str(alter)
            ET.SubElement(pitch, 'octave').text = str(octave)
            ET.SubElement(n, 'duration').text = str(4 * divisions)
            ET.SubElement(n, 'voice').text = '1'
            ET.SubElement(n, 'type').text = 'whole'
            measure.insert(insertion + j, n)
        changed += 1
    if changed:
        tmp = Path(str(mxl_path) + '.vector.tmp')
        tree.write(tmp, encoding='UTF-8', xml_declaration=True)
        os.replace(tmp, mxl_path)
        if log:
            log(f'[vector] Recovered {changed} whole-note chord bars on page {page_index + 1}')
    return changed
