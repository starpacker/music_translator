"""Partial ties must not turn newly attacked chord pitches into dashes."""
from pathlib import Path
import json
import shutil
import tempfile
import unittest
from unittest import mock

import pymupdf
from music21 import bar, chord, dynamics, key, meter, midi, note, spanner, stream, tempo, tie

from core.notation.jianpu import extract_jianpu_measures, jianpu_note_token
from scripts.audit_jianpu_events import audit


def score_with_ties(first_pitches, first_ties, second_pitches, second_ties):
    score, part = stream.Score(), stream.Part()
    for index, (pitches, kinds) in enumerate(((first_pitches, first_ties),
                                               (second_pitches, second_ties)), 1):
        measure = stream.Measure(number=index)
        if index == 1:
            measure.insert(0, key.Key('C'))
            measure.insert(0, meter.TimeSignature('4/4'))
        event = chord.Chord(pitches, quarterLength=4)
        for component, kind in zip(event.notes, kinds):
            if kind:
                component.tie = tie.Tie(kind)
        measure.insert(0, event)
        part.append(measure)
    score.insert(0, part)
    return score


def midi_intervals(path):
    midi_file = midi.MidiFile()
    midi_file.open(str(path))
    try:
        midi_file.read()
    finally:
        midi_file.close()
    intervals = []
    for track in midi_file.tracks:
        time, active = 0, {}
        for event in track.events:
            if isinstance(event, midi.DeltaTime):
                time += event.time
            elif event.type == 144 and event.velocity:
                active.setdefault((event.channel, event.pitch), []).append(time)
            elif event.type == 128 or (event.type == 144 and not event.velocity):
                starts = active.get((event.channel, event.pitch), [])
                if starts:
                    start = starts.pop(0)
                    intervals.append((event.pitch, start / midi_file.ticksPerQuarterNote,
                                      time / midi_file.ticksPerQuarterNote))
        if any(active.values()):
            raise AssertionError('MIDI has an unmatched note-on')
    return sorted(intervals)


class PartialChordTieTest(unittest.TestCase):
    def test_midi_keeps_partial_chord_tie_without_changing_source_score(self):
        from core.render.renderer import render_midi_from_score
        score = score_with_ties(['C4', 'E4'], ['start', None],
                                ['C4', 'G4'], ['stop', None])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'out.mid'
            self.assertTrue(render_midi_from_score(score, path))
            self.assertEqual(midi_intervals(path), [(60, 0, 8), (64, 0, 4), (67, 4, 8)])
        chords = list(score.parts[0].recurse().getElementsByClass(chord.Chord))
        self.assertEqual([[n.tie.type if n.tie else None for n in c.notes] for c in chords],
                         [['start', None], ['stop', None]])
        self.assertFalse(list(score.parts[0].recurse().getElementsByClass('Instrument')))

    def test_midi_keeps_partial_tie_across_three_chords(self):
        from core.render.renderer import render_midi_from_score
        score = score_with_ties(['C4', 'E4'], ['start', None],
                                ['C4', 'G4'], ['continue', None])
        third = stream.Measure(number=3)
        event = chord.Chord(['C4', 'A4'], quarterLength=4)
        event.notes[0].tie = tie.Tie('stop')
        third.insert(0, event)
        score.parts[0].append(third)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'three.mid'
            self.assertTrue(render_midi_from_score(score, path))
            self.assertEqual(midi_intervals(path),
                             [(60, 0, 12), (64, 0, 4), (67, 4, 8), (69, 8, 12)])

    def test_midi_preserves_both_fully_tied_and_released_chord_tones(self):
        from core.render.renderer import render_midi_from_score
        cases = [
            (score_with_ties(['C4', 'E4'], ['start', 'start'],
                             ['C4', 'E4'], ['stop', 'stop']),
             [(60, 0, 8), (64, 0, 8)]),
            (score_with_ties(['C4', 'E4', 'G4'], ['start', 'start', 'start'],
                             ['C4', 'E4'], ['stop', 'stop']),
             [(60, 0, 8), (64, 0, 8), (67, 0, 4)]),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            for index, (score, expected) in enumerate(cases):
                path = Path(tmp) / f'{index}.mid'
                self.assertTrue(render_midi_from_score(score, path))
                self.assertEqual(midi_intervals(path), expected)

    def test_midi_ties_remain_separate_between_voices(self):
        from core.render.renderer import render_midi_from_score
        score, part = stream.Score(), stream.Part()
        for number in (1, 2):
            measure = stream.Measure(number=number)
            if number == 1:
                measure.insert(0, meter.TimeSignature('4/4'))
            for voice_id, pitch in (('1', 'C4'), ('2', 'E4')):
                voice = stream.Voice(id=voice_id)
                event = note.Note(pitch, quarterLength=4)
                event.tie = tie.Tie('start' if number == 1 else 'stop')
                voice.insert(0, event)
                measure.insert(0, voice)
            part.append(measure)
        score.insert(0, part)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'voices.mid'
            self.assertTrue(render_midi_from_score(score, path))
            self.assertEqual(midi_intervals(path), [(60, 0, 8), (64, 0, 8)])

    def test_midi_partial_tie_survives_repeat_and_slur(self):
        from core.render.renderer import render_midi_from_score
        score = score_with_ties(['C4', 'E4'], ['start', None],
                                ['C4', 'G4'], ['stop', None])
        measures = list(score.parts[0].getElementsByClass(stream.Measure))
        measures[0].leftBarline = bar.Repeat(direction='start')
        measures[1].rightBarline = bar.Repeat(direction='end')
        notes = [next(m.recurse().getElementsByClass(chord.Chord)) for m in measures]
        score.insert(0, spanner.Slur(*notes))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'repeated.mid'
            self.assertTrue(render_midi_from_score(score, path))
            self.assertEqual(midi_intervals(path), [(60, 0, 8), (60, 8, 16),
                                                    (64, 0, 4), (64, 8, 12),
                                                    (67, 4, 8), (67, 12, 16)])

    def test_midi_unbalanced_repeat_still_keeps_tied_note(self):
        from core.render import renderer
        score, part = stream.Score(), stream.Part()
        measures = []
        for number in (1, 2, 3):
            measure = stream.Measure(number=number)
            event = note.Note('C4', quarterLength=4)
            if number == 1:
                event.tie = tie.Tie('start')
            elif number == 2:
                event.tie = tie.Tie('stop')
                measure.rightBarline = bar.Repeat(direction='end')
            measure.insert(0, event)
            part.append(measure)
            measures.append(measure)
        part.insert(0, spanner.RepeatBracket(measures[2], number=1))
        score.insert(0, part)
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(renderer, 'log_message') as log:
            path = Path(tmp) / 'broken-repeat.mid'
            self.assertTrue(renderer.render_midi_from_score(score, path))
            self.assertEqual(midi_intervals(path), [(60, 0, 8), (60, 8, 12)])
            self.assertTrue(any('反复记号不成对' in str(call) for call in log.call_args_list))

    def test_midi_musicxml_parser_accepts_wedge_continuation(self):
        from core.render.renderer import render_midi_from_musicxml
        score = score_with_ties(['C4', 'E4'], ['start', None],
                                ['C4', 'G4'], ['stop', None])
        with tempfile.TemporaryDirectory() as tmp:
            xml, path = Path(tmp) / 'score.musicxml', Path(tmp) / 'score.mid'
            score.write('musicxml', fp=xml)
            text = xml.read_text().replace(
                '</attributes>',
                '</attributes><direction><direction-type><wedge type="continue"/>'
                '</direction-type></direction>', 1)
            xml.write_text(text)
            self.assertTrue(render_midi_from_musicxml(xml, path))
            self.assertEqual(midi_intervals(path), [(60, 0, 8), (64, 0, 4), (67, 4, 8)])

    def test_midi_keeps_tempo_and_dynamic_changes(self):
        from core.render.renderer import render_midi_from_score
        score = score_with_ties(['C4', 'E4'], ['start', None],
                                ['C4', 'G4'], ['stop', None])
        measures = list(score.parts[0].getElementsByClass(stream.Measure))
        measures[0].insert(0, dynamics.Dynamic('f'))
        measures[1].insert(0, dynamics.Dynamic('p'))
        score.insert(0, tempo.MetronomeMark(number=120))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'dynamics.mid'
            self.assertTrue(render_midi_from_score(score, path))
            mf = midi.MidiFile()
            mf.open(str(path))
            try:
                mf.read()
            finally:
                mf.close()
            velocities, tempos = {}, []
            for track in mf.tracks:
                for event in track.events:
                    if event.type == 144 and event.velocity:
                        velocities[event.pitch] = event.velocity
                    elif event.type == 81:
                        tempos.append(int.from_bytes(event.data, 'big'))
            self.assertIn(500000, tempos)  # 120 quarter notes per minute
            self.assertGreater(velocities[64], velocities[67])

    def test_unmatched_midi_tie_is_kept_and_reported(self):
        from core.render import renderer
        score, part, measure = stream.Score(), stream.Part(), stream.Measure(number=1)
        event = note.Note('C4', quarterLength=4)
        event.tie = tie.Tie('stop')
        measure.insert(0, event)
        part.append(measure)
        score.insert(0, part)
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(renderer, 'log_message') as log:
            path = Path(tmp) / 'unmatched.mid'
            self.assertTrue(renderer.render_midi_from_score(score, path))
            self.assertEqual(midi_intervals(path), [(60, 0, 4)])
            self.assertTrue(any('延音不完整或有歧义' in str(call) for call in log.call_args_list))

    def test_partial_tie_keeps_both_chords_and_is_reported(self):
        score = score_with_ties(['C4', 'E4'], ['start', None],
                                ['C4', 'G4'], ['stop', None])
        with mock.patch('core.notation.jianpu.extract.log_message') as log:
            measures, _ = extract_jianpu_measures(score, 0, follow_key_changes=True)
        self.assertEqual([jianpu_note_token(m[0]) for m in measures], ['13 - - -', '15 - - -'])
        self.assertTrue(any('部分延音' in str(call) for call in log.call_args_list))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'score.musicxml'
            score.write('musicxml', fp=path)
            result = audit(path)
            partial = [m for m in result['mismatches'] if m.get('reason') == 'partial_chord_tie_unrepresented']
            self.assertEqual([m['index'] for m in partial], [1, 2])
            self.assertEqual(partial[1]['pitches'], [60, 67])

    def test_entire_identical_chord_can_still_continue_as_dashes(self):
        score = score_with_ties(['C4', 'E4'], ['start', 'start'],
                                ['C4', 'E4'], ['stop', 'stop'])
        measures, _ = extract_jianpu_measures(score, 0, follow_key_changes=True)
        self.assertEqual([jianpu_note_token(m[0]) for m in measures], ['13 - - -', '- - - -'])
        self.assertFalse(any(event.tie_review for bar in measures for event in bar))
        with tempfile.TemporaryDirectory() as tmp:
            xml = Path(tmp) / 'score.musicxml'
            score.write('musicxml', fp=xml)
            self.assertEqual(audit(xml)['mismatch_count'], 0)

    def test_subset_of_previous_chord_does_not_repeat_a_released_pitch(self):
        score = score_with_ties(['C4', 'E4', 'G4'], ['start', 'start', 'start'],
                                ['C4', 'E4'], ['stop', 'stop'])
        with mock.patch('core.notation.jianpu.extract.log_message') as log:
            measures, _ = extract_jianpu_measures(score, 0, follow_key_changes=True)
        self.assertEqual([jianpu_note_token(m[0]) for m in measures], ['135 - - -', '13 - - -'])
        self.assertEqual(measures[1][0].tie_review, 'unmatched_tie')
        self.assertTrue(any('不匹配延音' in str(call) for call in log.call_args_list))

    def test_full_continuation_with_partial_outgoing_ties_is_flagged(self):
        score = score_with_ties(['C4', 'E4'], ['start', 'start'],
                                ['C4', 'E4'], ['continue', 'stop'])
        third = stream.Measure(number=3)
        event = note.Note('C4', quarterLength=4)
        event.tie = tie.Tie('stop')
        third.insert(0, event)
        score.parts[0].append(third)
        measures, _ = extract_jianpu_measures(score, 0, follow_key_changes=True)
        self.assertEqual([jianpu_note_token(m[0]) for m in measures],
                         ['13 - - -', '- - - -', '1 - - -'])
        self.assertEqual(measures[1][0].tie_review, 'partial_chord_tie')
        self.assertEqual(measures[2][0].tie_review, 'unmatched_tie')

    def test_orphan_stop_is_printed_as_a_note(self):
        score, part, measure = stream.Score(), stream.Part(), stream.Measure(number=1)
        measure.insert(0, key.Key('C'))
        event = note.Note('C4', quarterLength=4)
        event.tie = tie.Tie('stop')
        measure.insert(0, event)
        part.append(measure)
        score.insert(0, part)
        with mock.patch('core.notation.jianpu.extract.log_message') as log:
            measures, _ = extract_jianpu_measures(score, 0, follow_key_changes=True)
        self.assertEqual(jianpu_note_token(measures[0][0]), '1 - - -')
        self.assertTrue(any('不匹配延音' in str(call) for call in log.call_args_list))

    @unittest.skipUnless(shutil.which('lilypond'), 'LilyPond is required')
    def test_both_pdf_paths_keep_new_chord_pitch(self):
        from core.render import renderer
        from insitu import render_insitu
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            score = score_with_ties(['C4', 'E4'], ['start', None],
                                    ['C4', 'G4'], ['stop', None])
            xml, staff = root / 'score.musicxml', root / 'staff.pdf'
            score.write('musicxml', fp=xml)
            with pymupdf.open() as doc:
                page = doc.new_page(width=600, height=240)
                for y in (100, 105, 110, 115, 120):
                    page.draw_line((40, y), (540, y), width=.5)
                for x in (290, 540):
                    page.draw_line((x, 100), (x, 120), width=.5)
                doc.save(staff)
            workspace = root / 'workspace'
            self.assertTrue(renderer.generate_jianpu_pdf_from_mxl(
                xml, root / 'jianpu.pdf', root, editor_workspace_dir=workspace,
                midi_output_path=root / 'score.mid'))
            self.assertEqual(midi_intervals(root / 'score.mid'), [(60, 0, 8), (64, 0, 4), (67, 4, 8)])
            text = next(workspace.glob('*.jianpu.txt')).read_text()
            self.assertIn('15 - - -', text)
            with mock.patch.object(render_insitu, '_run_jianpu_ly',
                                   wraps=render_insitu._run_jianpu_ly) as render:
                render_insitu.build(staff, xml, root / 'insitu.pdf', expand=True)
            self.assertIn('15 - - -', render.call_args_list[0].args[0])
            self.assertTrue((root / 'insitu.pdf').is_file())
            report = json.loads((root / 'insitu.report.json').read_text())
            self.assertEqual(report['status'], 'needs_review')
            self.assertEqual(report['tie_review_measure_count'], 2)
            self.assertEqual(report['rows'][0]['voices'][0]['tie_review_measures'],
                             [{'musicxml_index': 1, 'reasons': ['partial_chord_tie']},
                              {'musicxml_index': 2, 'reasons': ['partial_chord_tie']}])


if __name__ == '__main__':
    unittest.main()
