"""MusicXML tempo marks must not move to the start or disappear."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from music21 import converter, duration, meter, note, stream, tempo
from core.notation.jianpu import build_jianpu_ly_text


def without_meter_label(text):
    return text.replace(' ^\\markup{\\concat{\\hspace#3"4/4"}}', '')


def make_score():
    score, part = stream.Score(), stream.Part()
    for index in range(2):
        measure = stream.Measure(number=index + 1)
        if index == 0:
            measure.insert(0, meter.TimeSignature('4/4'))
            measure.insert(2, tempo.MetronomeMark(number=70))
        else:
            dotted = duration.Duration(type='quarter', dots=1)
            measure.insert(0, tempo.MetronomeMark(number=92, referent=dotted))
        for beat in range(4):
            measure.insert(beat, note.Note(('C4', 'D4', 'E4', 'F4')[beat], quarterLength=1))
        part.append(measure)
    score.insert(0, part)
    return score


class TempoPreservationTest(unittest.TestCase):
    def test_tempo_changes_keep_measure_offset_and_beat_unit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'score.musicxml'
            make_score().write('musicxml', fp=path)
            text = build_jianpu_ly_text(converter.parse(str(path)), 'test')
            self.assertIn('1 2 4=70 3 4 | 4.=92 1 2 3 4 |', without_meter_label(text))
            self.assertNotIn('\n4=70\n', text)
            script = Path(__file__).resolve().parents[1] / 'scripts/jianpu-ly.py'
            source = Path(tmp) / 'score.txt'
            source.write_text(text)
            parsed = subprocess.run([sys.executable, str(script), str(source)], capture_output=True, text=True)
            self.assertEqual(parsed.returncode, 0, parsed.stderr)
            self.assertIn('\\tempo 4=70', parsed.stdout)
            self.assertIn('\\tempo 4.=92', parsed.stdout)
            if shutil.which('lilypond'):
                ly = Path(tmp) / 'score.ly'
                ly.write_text(parsed.stdout)
                rendered = subprocess.run(['lilypond', '-o', str(Path(tmp) / 'score'), str(ly)],
                                          capture_output=True, text=True, env={**os.environ, 'TMPDIR': tmp})
                self.assertEqual(rendered.returncode, 0, rendered.stderr)

    def test_explicit_initial_tempo_does_not_erase_later_changes(self):
        score = make_score()
        score.parts[0].measure(1).insert(0, tempo.MetronomeMark(number=60))
        text = build_jianpu_ly_text(score, 'test', tempo=80)
        self.assertIn('\n4=80\n', text)
        self.assertNotIn('4=60', text)
        self.assertIn('1 2 4=70 3 4 | 4.=92', without_meter_label(text))

    def test_marks_directly_on_score_and_part_keep_their_offsets(self):
        score = make_score()
        part = score.parts[0]
        score.insert(0, tempo.MetronomeMark(number=50))
        part.insert(4, tempo.MetronomeMark(number=80, referent='eighth'))
        text = build_jianpu_ly_text(score, 'test')
        self.assertIn('4=50 1 2 4=70 3 4 |', without_meter_label(text))
        self.assertIn('8=80', text)

    def test_unmeasured_score_retains_tempo(self):
        score = stream.Score()
        score.insert(0, tempo.MetronomeMark(number=50))
        score.insert(0, note.Note('C4', quarterLength=1))
        self.assertIn('4=50', build_jianpu_ly_text(score, 'test'))

    def test_musicxml_to_pdf_path_does_not_hoist_first_later_tempo(self):
        if not shutil.which('lilypond'):
            self.skipTest('LilyPond is required')
        from core.render.renderer import generate_jianpu_pdf_from_mxl
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, target = root / 'source.musicxml', root / 'result.pdf'
            make_score().write('musicxml', fp=source)
            workspace = root / 'workspace'
            self.assertTrue(generate_jianpu_pdf_from_mxl(source, target, root, editor_workspace_dir=workspace))
            text = next(workspace.glob('*.jianpu.txt')).read_text()
            self.assertIn('1 2 4=70 3 4 | 4.=92 1 2 3 4 |', without_meter_label(text))
            self.assertNotIn('\n4=70\n', text)
            self.assertTrue(target.is_file())

    def test_unplaceable_mark_is_reported_not_moved(self):
        score = make_score()
        score.parts[0].measure(1).insert(1.5, tempo.MetronomeMark(number=88))
        with mock.patch('core.notation.jianpu.log_message') as log:
            text = build_jianpu_ly_text(score, 'test')
        self.assertNotIn('4=88', text)
        self.assertIn('4=70', text)
        self.assertTrue(any('不在可表示的音符边界' in str(call) for call in log.call_args_list))


if __name__ == '__main__':
    unittest.main()
