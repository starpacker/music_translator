"""Recognized MusicXML dynamic directions survive Jianpu rendering."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from fractions import Fraction
from unittest import mock

from music21 import converter, dynamics, meter, note, stream
from core.notation.jianpu import build_jianpu_ly_text


def make_score():
    score, part = stream.Score(), stream.Part()
    first, second = stream.Measure(number=1), stream.Measure(number=2)
    first.insert(0, meter.TimeSignature('4/4'))
    first.insert(0, dynamics.Dynamic('mp'))
    first.insert(0, note.Note('C4', quarterLength=2))
    first.insert(2, dynamics.Dynamic('f'))
    first.insert(2, note.Note('D4', quarterLength=2))
    second.insert(0, dynamics.Dynamic('pp'))
    second.insert(0, note.Note('E4', quarterLength=4))
    part.append([first, second])
    score.insert(0, part)
    return score


class DynamicPreservationTest(unittest.TestCase):
    def test_musicxml_dynamics_attach_to_first_fragment_of_correct_note(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source.musicxml'
            make_score().write('musicxml', fp=source)
            text = build_jianpu_ly_text(converter.parse(str(source)), 'test')
            self.assertIn('1 \\mp - 2 \\f - | 3 \\pp - - - |', text)
            txt = root / 'score.txt'
            txt.write_text(text)
            script = Path(__file__).resolve().parents[1] / 'scripts' / 'jianpu-ly.py'
            parsed = subprocess.run([sys.executable, str(script), str(txt)],
                                    capture_output=True, text=True)
            self.assertEqual(parsed.returncode, 0, parsed.stderr)
            for mark in ('\\mp', '\\f', '\\pp'):
                self.assertIn(mark, parsed.stdout)
            if shutil.which('lilypond'):
                ly = root / 'score.ly'
                ly.write_text(parsed.stdout)
                rendered = subprocess.run(['lilypond', '-o', str(root / 'score'), str(ly)],
                                          capture_output=True, text=True, env={**os.environ, 'TMPDIR': tmp})
                self.assertEqual(rendered.returncode, 0, rendered.stderr)

    def test_non_boundary_dynamic_is_reported(self):
        score = make_score()
        score.parts[0].measure(1).insert(1, dynamics.Dynamic('ff'))
        with mock.patch('core.notation.jianpu.log_message') as log:
            text = build_jianpu_ly_text(score, 'test')
        self.assertNotIn('\\ff', text)
        self.assertIn('\\f', text)
        self.assertTrue(any('力度记号' in str(call) and '不在可表示的音符边界' in str(call)
                            for call in log.call_args_list))

    def test_dynamics_wait_for_principal_note_after_grace(self):
        score = make_score()
        score.parts[0].measure(1).insert(0, note.Note('D4').getGrace())
        text = build_jianpu_ly_text(score, 'test')
        self.assertIn('g[2] 1 \\mp -', text)

    def test_dynamic_attaches_inside_tuplet_after_its_first_note(self):
        score, part, measure = stream.Score(), stream.Part(), stream.Measure(number=1)
        measure.insert(0, dynamics.Dynamic('mf'))
        for i, pitch in enumerate(('C4', 'D4', 'E4')):
            measure.insert(Fraction(i, 3), note.Note(pitch, quarterLength=Fraction(1, 3)))
        measure.insert(1, note.Rest(quarterLength=3))
        part.append(measure)
        score.insert(0, part)
        self.assertIn('3[ q1 \\mf q2 q3 ]', build_jianpu_ly_text(score, 'test'))

    def test_unsupported_dynamic_is_reported(self):
        score = make_score()
        score.parts[0].measure(1).insert(1, dynamics.Dynamic('sffz'))
        with mock.patch('core.notation.jianpu.log_message') as log:
            text = build_jianpu_ly_text(score, 'test')
        self.assertNotIn('\\sffz', text)
        self.assertTrue(any('力度记号' in str(call) and '无法表示' in str(call)
                            for call in log.call_args_list))


if __name__ == '__main__':
    unittest.main()
