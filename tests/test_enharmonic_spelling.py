"""Written accidentals should retain the source note's diatonic identity."""
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from music21 import chord, converter, key, meter, note, stream

from core.notation.jianpu import (_get_score_key_tonic, extract_jianpu_measures,
                                  jianpu_note_token)
from core.notation.jianpu.primitives import note_to_jianpu
from scripts.audit_jianpu_events import audit


class EnharmonicSpellingTest(unittest.TestCase):
    def test_c_major_spelling_and_octave_boundaries(self):
        expected = [('B-4', 'b7'), ('A#4', '#6'), ('F-4', 'b4'),
                    ('E#4', '#3'), ('C-4', 'b1'), ('B#4', '#7')]
        for pitch, token in expected:
            with self.subTest(pitch=pitch):
                converted = note_to_jianpu(note.Note(pitch), 0, tonic_step='C')
                self.assertEqual(jianpu_note_token(converted), token)

    def test_extraction_tracks_spelling_key_changes_chords_and_graces(self):
        score, part = stream.Score(), stream.Part()
        first = stream.Measure(number=1)
        first.insert(0, key.Key('C'))
        first.insert(0, meter.TimeSignature('4/4'))
        first.insert(0, note.Note('B-4', quarterLength=1))
        first.insert(1, note.Note('A#4').getGrace())
        first.insert(1, chord.Chord(['E#4', 'B-4'], quarterLength=1))
        first.insert(2, note.Note('C-4', quarterLength=1))
        first.insert(3, note.Note('B#4', quarterLength=1))
        second = stream.Measure(number=2)
        second.insert(0, key.Key('G#'))
        second.insert(0, note.Note('B4', quarterLength=2))
        second.insert(2, note.Note('A#4', quarterLength=2))
        part.append([first, second])
        score.insert(0, part)
        tonic, _ = _get_score_key_tonic(score)
        measures, _ = extract_jianpu_measures(score, tonic, follow_key_changes=True)
        self.assertEqual([jianpu_note_token(n) for n in measures[0]],
                         ['b7', 'g[#6]', '#3b7', 'b1', '#7'])
        self.assertEqual([jianpu_note_token(n) for n in measures[1]], ['b3 -', '2 -'])
        self.assertEqual(measures.key_changes[1], (8, '1=G#'))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'score.musicxml'
            score.write('musicxml', fp=path)
            result = audit(path)
            self.assertEqual(result['mismatch_count'], 0, result['mismatches'])

    def test_enharmonic_key_change_same_pitch_class_changes_degree_spelling(self):
        score, part = stream.Score(), stream.Part()
        for number, signature in enumerate((key.Key('G#'), key.Key('A-')), 1):
            measure = stream.Measure(number=number)
            measure.insert(0, signature)
            measure.insert(0, note.Note('C5', quarterLength=4))
            part.append(measure)
        score.insert(0, part)
        measures, _ = extract_jianpu_measures(score, 8, follow_key_changes=True)
        self.assertEqual([jianpu_note_token(m[0]) for m in measures], ['b4 - - -', '3 - - -'])
        self.assertEqual(measures.key_changes[1], (8, '1=Ab'))

    def test_double_accidentals_preserve_source_spelling_and_octave(self):
        for pitch, token in [('F##4', '##4'), ('B--4', 'bb7'),
                             ('C##4', '##1'), ('E--4', 'bb3'),
                             ('C--4', 'bb1'), ('B##4', '##7')]:
            with self.subTest(pitch=pitch):
                self.assertEqual(jianpu_note_token(note_to_jianpu(note.Note(pitch), 0, tonic_step='C')), token)

    def test_microtone_is_warned_and_cannot_pass_pitch_audit(self):
        score, part, measure = stream.Score(), stream.Part(), stream.Measure(number=1)
        measure.insert(0, key.Key('C'))
        event = note.Note('C4', quarterLength=4)
        event.pitch.microtone = 50
        measure.insert(0, event)
        part.append(measure)
        score.insert(0, part)
        with mock.patch('core.notation.jianpu.primitives._log_message') as log:
            note_to_jianpu(event, 0, tonic_step='C')
        self.assertTrue(any('微分音' in str(call) for call in log.call_args_list))
        with tempfile.TemporaryDirectory() as tmp:
            xml = Path(tmp) / 'microtone.musicxml'
            score.write('musicxml', fp=xml)
            # music21's exporter omits a free-standing .microtone adjustment;
            # represent the same quarter-tone explicitly in MusicXML.
            xml.write_text(xml.read_text().replace('<step>C</step>',
                                                   '<step>C</step><alter>0.5</alter>', 1))
            result = audit(xml)
            self.assertGreater(result['mismatch_count'], 0)
            self.assertEqual(result['mismatches'][0]['source'][0]['pitches'], [60.5])

    @unittest.skipUnless(shutil.which('lilypond'), 'LilyPond is required')
    def test_both_real_pdf_rendering_paths_accept_written_accidentals(self):
        from core.render import renderer
        from insitu.render_insitu import _jianpu_ly_source, _render_ly_tile, _run_jianpu_ly
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            score, part, measure = stream.Score(), stream.Part(), stream.Measure(number=1)
            measure.insert(0, key.Key('C'))
            measure.insert(0, meter.TimeSignature('4/4'))
            for i, pitch in enumerate(['B-4', 'A#4', 'F-4', 'E#4']):
                measure.insert(i, note.Note(pitch, quarterLength=1))
            part.append(measure)
            score.insert(0, part)
            path = root / 'score.musicxml'
            score.write('musicxml', fp=path)
            workspace = root / 'workspace'
            self.assertTrue(renderer.generate_jianpu_pdf_from_mxl(path, root / 'standalone.pdf', root,
                                                                 editor_workspace_dir=workspace))
            text = next(workspace.glob('*.jianpu.txt')).read_text()
            self.assertIn('b7 ', text)
            self.assertIn('#6 b4 #3', text)
            ly = _run_jianpu_ly(_jianpu_ly_source([['b7 #6 b4 #3']], '1=C', '4/4', 'test'), root)
            self.assertTrue(_render_ly_tile(ly, root)[0].is_file())
            midi = converter.parse(str(root / 'line.midi'))
            self.assertEqual([event.pitch.midi for event in midi.recurse().notes],
                             [70, 70, 64, 65])
            ly = _run_jianpu_ly(_jianpu_ly_source([['b1 - #7 -']], '1=C', '4/4', 'test'), root)
            self.assertTrue(_render_ly_tile(ly, root)[0].is_file())
            midi = converter.parse(str(root / 'line.midi'), forceSource=True)
            self.assertEqual([event.pitch.midi for event in midi.recurse().notes], [59, 72])

    @unittest.skipUnless(shutil.which('lilypond'), 'LilyPond is required')
    def test_double_accidentals_in_normal_notes_chords_and_graces(self):
        from core.render import renderer
        from insitu.render_insitu import _jianpu_ly_source, _render_ly_tile, _run_jianpu_ly
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            score, part, measure = stream.Score(), stream.Part(), stream.Measure(number=1)
            measure.insert(0, key.Key('C'))
            measure.insert(0, meter.TimeSignature('4/4'))
            for i, pitch in enumerate(['F##4', 'B--4', 'C##4', 'E--4']):
                measure.insert(i, note.Note(pitch, quarterLength=1))
            part.append(measure)
            score.insert(0, part)
            xml = root / 'score.musicxml'
            score.write('musicxml', fp=xml)
            result = audit(xml)
            self.assertEqual(result['mismatch_count'], 0, result['mismatches'])
            workspace = root / 'workspace'
            with mock.patch.object(renderer, 'render_jianpu_ly', wraps=renderer.render_jianpu_ly) as typeset:
                self.assertTrue(renderer.generate_jianpu_pdf_from_mxl(
                    xml, root / 'standalone.pdf', root, editor_workspace_dir=workspace))
            self.assertEqual(typeset.call_count, 1, 'the normal LilyPond path must succeed')
            text = next(workspace.glob('*.jianpu.txt')).read_text()
            self.assertIn('##4 ', text)
            self.assertIn('bb7 ##1 bb3', text)
            ly = _run_jianpu_ly(_jianpu_ly_source([['##4 bb7 ##1 bb3']], '1=C', '4/4', 'test'), root)
            self.assertTrue(_render_ly_tile(ly, root)[0].is_file())
            midi = converter.parse(str(root / 'line.midi'), forceSource=True)
            self.assertEqual([event.pitch.midi for event in midi.recurse().notes], [67, 69, 62, 62])
            ly = _run_jianpu_ly(_jianpu_ly_source([['g[##1bb5] ##1bb5 - - -']], '1=C', '4/4', 'test'), root)
            self.assertIn("cisis'16 geses'16", ly)
            self.assertIn("cisis' geses'", ly)
            self.assertTrue(_render_ly_tile(ly, root)[0].is_file())
            midi = converter.parse(str(root / 'line.midi'), forceSource=True)
            self.assertEqual([p.midi for p in list(midi.recurse().notes)[-1].pitches], [62, 65])

    @unittest.skipUnless(shutil.which('lilypond'), 'LilyPond is required')
    def test_editor_regeneration_keeps_sharps_at_start_of_a_line(self):
        from core.app import tui
        from core.render import lilypond_runner
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source.jianpu.txt'
            source.write_text('# instructional comment\n1=C\n4/4\n#4 - - - |\n##4 - - - |\n'
                              '#q4 q4 q4 q4 q4 q4 q4 q4 |\n')
            original = lilypond_runner.render_jianpu_ly
            generated = []
            def capture(txt, ly):
                result = original(txt, ly)
                if result:
                    generated.append(ly.read_text())
                return result
            ui = tui.TUI()
            with mock.patch.object(lilypond_runner, 'render_jianpu_ly', side_effect=capture), \
                 mock.patch.object(tui, '_open_file_default') as opened, mock.patch.object(tui, '_read_single_key'), \
                 mock.patch.object(ui, '_header'), mock.patch.object(ui, '_status_bar'):
                ui._do_regenerate(source, 'test', root)
            opened.assert_called_once_with(root / 'source.jianpu.pdf')
            self.assertTrue((root / 'source.jianpu.pdf').is_file())
            self.assertEqual(len(generated), 1)
            self.assertRegex(generated[0], r"\bfis[',0-9]")
            self.assertRegex(generated[0], r"\bfisis[',0-9]")


if __name__ == '__main__':
    unittest.main()
