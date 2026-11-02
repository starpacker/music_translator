#!/usr/bin/env python3
"""Compare extracted Jianpu pitch events with MusicXML, measure by measure.

This compares the conversion stage only. It does not establish that MusicXML
matches the original engraving. Ties, multiple voices and tuplet rounding can
produce differences that require review.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from music21 import chord, converter, note
from core.notation.jianpu import _get_score_key_tonic, extract_jianpu_measures


def audit(path: Path) -> dict:
    score = converter.parse(str(path))
    if not score.parts:
        raise ValueError('MusicXML 没有独立声部，无法逐小节比较')
    part = score.parts[0]
    original = list(part.getElementsByClass('Measure'))
    tonic, _ = _get_score_key_tonic(score)
    extracted, _ = extract_jianpu_measures(score, tonic, _part=part)
    mismatches = []
    for index in range(max(len(original), len(extracted))):
        if index >= len(original) or index >= len(extracted):
            mismatches.append({'index': index + 1, 'reason': 'measure_count'})
            continue
        source = []
        for event in original[index].recurse().notes:
            pitches = (sorted(p.midi for p in event.pitches)
                       if isinstance(event, chord.Chord) else [event.pitch.midi])
            source.append({'pitches': pitches, 'duration': round(float(event.duration.quarterLength), 6),
                           'grace': bool(event.duration.isGrace)})
        target = []
        for event in extracted[index]:
            if event.is_rest or event.symbol == '-':
                continue
            target.append({'pitches': [event.midi, *[p[4] for p in event.chord_notes]],
                           'duration': round(float(event.duration), 6), 'grace': event.is_grace})
        source_rests = [round(float(event.duration.quarterLength), 6)
                        for event in original[index].recurse().notesAndRests
                        if isinstance(event, note.Rest)]
        target_rests = [round(float(event.duration), 6) for event in extracted[index] if event.is_rest]
        source_order = ['rest' if isinstance(event, note.Rest) else 'note'
                        for event in original[index].recurse().notesAndRests]
        target_order = ['rest' if event.is_rest else 'note' for event in extracted[index]
                        if event.symbol != '-']
        if source != target or source_rests != target_rests or source_order != target_order:
            mismatches.append({'index': index + 1, 'measure_number': str(original[index].number),
                               'source': source, 'jianpu': target,
                               'source_rests': source_rests, 'jianpu_rests': target_rests,
                               'source_order': source_order, 'jianpu_order': target_order})
    return {'input': str(path), 'scope': 'first_part_musicxml_to_jianpu_events',
            'source_measures': len(original), 'jianpu_measures': len(extracted),
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
