"""Grace notes remain visible without adding beats to the measure."""
import os
import subprocess
import sys
import tempfile
import unittest
import shutil
from pathlib import Path
from music21 import meter, note, stream
from core.notation.jianpu import extract_jianpu_measures, jianpu_note_token
from core.notation.jianpu.primitives import build_lyric_lines
from core.notation.jianpu.measure import measure_units


class GraceNoteTest(unittest.TestCase):
    def test_graces_keep_pitch_order_and_zero_duration(self):
        score = stream.Score()
        part = stream.Part()
        measure = stream.Measure(number=1)
        measure.insert(0, meter.TimeSignature('4/4'))
        measure.insert(0, note.Note('D4').getGrace())
        measure.insert(0, note.Note('E4').getGrace())
        measure.insert(0, note.Note('C4', quarterLength=4))
        part.append(measure)
        score.insert(0, part)
        measures, _ = extract_jianpu_measures(score, 0)
        tokens = [jianpu_note_token(n) for n in measures[0]]
        self.assertEqual(tokens, ['g[2]', 'g[3]', '1 - - -'])
        self.assertEqual(measure_units(measures[0]), 64)
        src = '1=C\n4/4\n' + ' '.join(tokens) + ' |\n'
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'score.txt'
            path.write_text(src, encoding='utf-8')
            result = subprocess.run([sys.executable, 'scripts/jianpu-ly.py', str(path)],
                                    capture_output=True, text=True, env={**os.environ, 'TMPDIR': '/tmp'})
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('jianpuGrace', result.stdout)
            if shutil.which('lilypond'):
                ly = Path(tmp) / 'score.ly'
                ly.write_text(result.stdout, encoding='utf-8')
                render = subprocess.run(['lilypond', '-o', str(Path(tmp) / 'score'), str(ly)],
                                        capture_output=True, text=True, env={**os.environ, 'TMPDIR': tmp})
                self.assertEqual(render.returncode, 0, render.stderr)

    def test_graces_do_not_consume_lyric_syllables(self):
        score = stream.Score()
        part = stream.Part()
        measure = stream.Measure(number=1)
        measure.insert(0, note.Note('D4').getGrace())
        main = note.Note('C4', quarterLength=4)
        main.lyric = 'la'
        measure.insert(0, main)
        part.append(measure)
        score.insert(0, part)
        measures, _ = extract_jianpu_measures(score, 0)
        self.assertEqual(build_lyric_lines(measures), ['L: la'])
