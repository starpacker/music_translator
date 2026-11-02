"""MusicXML melodic slurs survive conversion without changing rhythm."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from music21 import meter, note, spanner, stream
from core.notation.jianpu import extract_jianpu_measures, jianpu_note_token
from core.notation.jianpu.measure import measure_units
from scripts.audit_jianpu_events import audit


class SlurPreservationTest(unittest.TestCase):
    def test_cross_line_slur_is_closed_and_reopened_without_mutating_source(self):
        if not shutil.which('lilypond'):
            self.skipTest('LilyPond is required by the in-situ renderer')
        from insitu.render_insitu import _line_slur_continuations
        score, part = stream.Score(), stream.Part()
        first, second = stream.Measure(number=1), stream.Measure(number=2)
        first.insert(0, meter.TimeSignature('4/4'))
        notes = [note.Note(p, quarterLength=1) for p in ('C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5')]
        first.append(notes[:4])
        second.append(notes[4:])
        part.append([first, second])
        score.insert(0, part)
        score.insert(0, spanner.Slur(notes[1], notes[6]))
        measures, _ = extract_jianpu_measures(score, 0)
        left = _line_slur_continuations(measures, 0, 1)
        right = _line_slur_continuations(measures, 1, 2)
        self.assertEqual(left[-1][-1].slur_carry_out, 1)
        self.assertEqual(right[0][0].slur_carry_in, 1)
        self.assertEqual(measures[0][-1].slur_carry_out, 0)
        self.assertEqual(measures[1][0].slur_carry_in, 0)

    def test_melodic_slur_is_retained_across_measures(self):
        score, part = stream.Score(), stream.Part()
        first, second = stream.Measure(number=1), stream.Measure(number=2)
        first.insert(0, meter.TimeSignature('4/4'))
        a, b = note.Note('C4', quarterLength=4), note.Note('D4', quarterLength=4)
        first.append(a)
        second.append(b)
        part.append([first, second])
        score.insert(0, part)
        score.insert(0, spanner.Slur(a, b))
        measures, _ = extract_jianpu_measures(score, 0)
        self.assertEqual([jianpu_note_token(m[0]) for m in measures], ['1 - - - (', '2 - - - )'])
        self.assertEqual([measure_units(m) for m in measures], [64, 64])
        with tempfile.TemporaryDirectory() as tmp:
            mxl = Path(tmp) / 'score.musicxml'
            score.write('musicxml', fp=mxl)
            report = audit(mxl)
            self.assertEqual(report['comparable_melodic_slurs'], 1)
            self.assertEqual(report['jianpu_slur_starts'], 1)
            self.assertEqual(report['jianpu_slur_stops'], 1)
            path = Path(tmp) / 'score.txt'
            path.write_text('1=C\n4/4\n' + ' | '.join(jianpu_note_token(m[0]) for m in measures) + ' |\n')
            script = Path(__file__).resolve().parents[1] / 'scripts/jianpu-ly.py'
            out = subprocess.run([sys.executable, str(script), str(path)], text=True, capture_output=True)
            self.assertEqual(out.returncode, 0, out.stderr)
            if shutil.which('lilypond'):
                ly = Path(tmp) / 'score.ly'
                ly.write_text(out.stdout)
                result = subprocess.run(['lilypond', '-o', str(Path(tmp) / 'score'), str(ly)],
                                        text=True, capture_output=True, env={**os.environ, 'TMPDIR': tmp})
                self.assertEqual(result.returncode, 0, result.stderr)
