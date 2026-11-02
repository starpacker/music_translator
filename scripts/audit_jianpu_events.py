#!/usr/bin/env python3
"""Compare extracted Jianpu pitch events with MusicXML, measure by measure.

This compares the conversion stage only. It does not establish that MusicXML
matches the original engraving. Ties, multiple voices and tuplet rounding can
produce differences that require review.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from music21 import articulations, chord, converter, expressions, note, spanner
from core.notation.jianpu import _get_score_key_tonic, extract_jianpu_measures


_SUPPORTED_MARKS = {
    articulations.Accent: 'accent', articulations.StrongAccent: 'marcato',
    articulations.Staccato: 'staccato', articulations.Staccatissimo: 'staccatissimo',
    articulations.Tenuto: 'tenuto', expressions.Fermata: 'fermata',
    expressions.Trill: 'trill', expressions.Mordent: 'mordent',
    expressions.InvertedMordent: 'prall', expressions.Turn: 'turn',
    expressions.InvertedTurn: 'reverseturn',
}


def audit(path: Path) -> dict:
    score = converter.parse(str(path))
    if not score.parts:
        raise ValueError('MusicXML 没有独立声部，无法逐小节比较')
    part = score.parts[0]
    original = list(part.getElementsByClass('Measure'))
    tonic, _ = _get_score_key_tonic(score)
    extracted, _ = extract_jianpu_measures(score, tonic, _part=part, follow_key_changes=True)
    part_note_ids = {id(event) for event in part.recurse().notes}
    comparable_slurs = []
    for slur in score.recurse().getElementsByClass(spanner.Slur):
        ends = slur.getSpannedElements()
        if (len(ends) >= 2 and all(id(event) in part_note_ids for event in (ends[0], ends[-1]))
                and not any(getattr(event.duration, 'isGrace', False) for event in (ends[0], ends[-1]))):
            comparable_slurs.append(slur)
    mismatches = []
    source_mark_counts, jianpu_mark_counts, unsupported_marks = Counter(), Counter(), Counter()
    previous_pitches = None
    reference = tonic
    for index in range(max(len(original), len(extracted))):
        if index >= len(original) or index >= len(extracted):
            mismatches.append({'index': index + 1, 'reason': 'measure_count'})
            continue
        source = []
        if index in getattr(extracted, 'key_changes', {}):
            reference = extracted.key_changes[index][0]
        canonical_measure = original[index]
        for event in canonical_measure.recurse().notes:
            pitches = (sorted(p.midi for p in event.pitches)
                       if isinstance(event, chord.Chord) else [event.pitch.midi])
            source.append({'pitches': pitches, 'duration': round(float(event.duration.quarterLength), 6),
                           'grace': bool(event.duration.isGrace),
                           'continuation': bool(event.tie and event.tie.type in ('stop', 'continue')),
                           'tuplets': [[t.numberNotesActual, t.numberNotesNormal]
                                       for t in event.duration.tuplets]})
        target = []
        symbol_mismatches = []
        for event in extracted[index]:
            if event.is_rest:
                previous_pitches = None
                continue
            continuation = event.symbol == '-'
            if not continuation:
                fields = [(event.accidental, event.symbol, event.upper_dots, event.lower_dots, event.midi),
                          *event.chord_notes]
                for accidental, symbol, upper, lower, expected in fields:
                    degrees = {'1': 0, '2': 2, '3': 4, '4': 5, '5': 7, '6': 9, '7': 11}
                    alters = {'': 0, '#': 1, 'b': -1}
                    rendered = (60 + reference + degrees[symbol] + alters[accidental] + 12 * (upper - lower)
                                if symbol in degrees and accidental in alters else None)
                    if rendered != expected:
                        symbol_mismatches.append({'symbol': symbol, 'accidental': accidental,
                                                  'upper_dots': upper, 'lower_dots': lower,
                                                  'reference': reference, 'rendered_midi': rendered,
                                                  'source_midi': expected})
            pitches = previous_pitches if continuation else [event.midi, *[p[4] for p in event.chord_notes]]
            target.append({'pitches': pitches,
                           'duration': round(float(event.duration), 6), 'grace': event.is_grace,
                           'continuation': continuation,
                           'tuplets': ([[event.tuplet_actual, event.tuplet_normal]]
                                       if event.tuplet_actual else [])})
            if not event.is_grace and not continuation:
                previous_pitches = pitches
        source_rests = [round(float(event.duration.quarterLength), 6)
                        for event in canonical_measure.recurse().notesAndRests
                        if isinstance(event, note.Rest)]
        target_rests = [round(float(event.duration), 6) for event in extracted[index] if event.is_rest]
        source_rest_tuplets = [[[t.numberNotesActual, t.numberNotesNormal] for t in event.duration.tuplets]
                               for event in canonical_measure.recurse().notesAndRests
                               if isinstance(event, note.Rest)]
        target_rest_tuplets = [([[event.tuplet_actual, event.tuplet_normal]] if event.tuplet_actual else [])
                               for event in extracted[index] if event.is_rest]
        source_order = ['rest' if isinstance(event, note.Rest) else 'note'
                        for event in canonical_measure.recurse().notesAndRests]
        target_order = ['rest' if event.is_rest else 'note' for event in extracted[index]]
        source_marks = []
        for event in canonical_measure.recurse().notesAndRests:
            marks = []
            for mark in (*event.articulations, *event.expressions):
                name = _SUPPORTED_MARKS.get(type(mark))
                if name is None:
                    unsupported_marks[type(mark).__name__] += 1
                else:
                    marks.append(name)
                    source_mark_counts[name] += 1
            source_marks.append(sorted(marks))
        target_marks = []
        for event in extracted[index]:
            marks = sorted(mark.lstrip('^_').removeprefix('\\') for mark in event.notations)
            target_marks.append(marks)
            jianpu_mark_counts.update(marks)
        # Empty per-event mark lists can differ merely because an unmarked
        # note or rest is missing; the event comparison already reports that.
        if source_marks != target_marks and (any(source_marks) or any(target_marks)):
            mismatches.append({'index': index + 1, 'measure_number': str(original[index].number),
                               'reason': 'note_mark_placement', 'source_marks': source_marks,
                               'jianpu_marks': target_marks})
        if (source != target or source_rests != target_rests or source_order != target_order
                or source_rest_tuplets != target_rest_tuplets):
            mismatches.append({'index': index + 1, 'measure_number': str(original[index].number),
                               'source': source, 'jianpu': target,
                               'source_rests': source_rests, 'jianpu_rests': target_rests,
                               'source_rest_tuplets': source_rest_tuplets,
                               'jianpu_rest_tuplets': target_rest_tuplets,
                               'source_order': source_order, 'jianpu_order': target_order})
        if symbol_mismatches:
            mismatches.append({'index': index + 1, 'measure_number': str(original[index].number),
                               'reason': 'jianpu_symbol_pitch', 'events': symbol_mismatches})
    starts = sum(event.slur_starts for measure in extracted for event in measure)
    stops = sum(event.slur_stops for measure in extracted for event in measure)
    if starts != len(comparable_slurs) or stops != len(comparable_slurs):
        mismatches.append({'reason': 'slur_boundary_count', 'source_slurs': len(comparable_slurs),
                           'jianpu_starts': starts, 'jianpu_stops': stops})
    return {'input': str(path), 'scope': 'first_part_musicxml_to_jianpu_events',
            'source_measures': len(original), 'jianpu_measures': len(extracted),
            'comparable_melodic_slurs': len(comparable_slurs),
            'jianpu_slur_starts': starts, 'jianpu_slur_stops': stops,
            'source_tie_continuations': sum(1 for event in part.recurse().notes
                                            if event.tie and event.tie.type in ('stop', 'continue')),
            'jianpu_tie_continuations': sum(1 for measure in extracted for event in measure
                                            if event.symbol == '-'),
            'source_supported_note_marks': dict(source_mark_counts),
            'jianpu_supported_note_marks': dict(jianpu_mark_counts),
            'source_unsupported_note_marks': dict(unsupported_marks),
            'mismatch_count': len(mismatches), 'mismatches': mismatches}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('musicxml', type=Path)
    args = parser.parse_args()
    result = audit(args.musicxml)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return int(result['mismatch_count'] > 0)


if __name__ == '__main__':
    raise SystemExit(main())
