"""Tie handling must work with the installed music21 API."""
import unittest
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from music21 import meter, note, stream, tie
from core.notation.jianpu import extract_jianpu_measures, jianpu_note_token
from core.notation.jianpu.measure import measure_units


def tied_note(pitch, duration, kind):
    event = note.Note(pitch, quarterLength=duration)
    event.tie = tie.Tie(kind)
    return event


class TieExtractionTest(unittest.TestCase):
    def test_tie_with_two_and_half_beats_does_not_invent_a_rest(self):
        score, part, measure = stream.Score(), stream.Part(), stream.Measure(number=1)
        measure.insert(0, meter.TimeSignature('4/4'))
        measure.append([tied_note('D5', 2, 'start'), tied_note('D5', .5, 'stop'),
                        note.Note('E4', quarterLength=.5), note.Note('A4', quarterLength=.5),
                        note.Note('B4', quarterLength=.5)])
        part.append(measure)
        score.insert(0, part)
        events = extract_jianpu_measures(score, 0)[0][0]
        self.assertFalse(any(n.is_rest for n in events))
        self.assertEqual([n.duration for n in events], [2, .5, .5, .5, .5])
        self.assertEqual(measure_units(events), 64)
        with tempfile.TemporaryDirectory() as tmp:
            txt = Path(tmp) / 'score.txt'
            txt.write_text('1=C\n4/4\n' + ' '.join(jianpu_note_token(n) for n in events) + ' |\n')
            script = Path(__file__).resolve().parents[1] / 'scripts/jianpu-ly.py'
            result = subprocess.run([sys.executable, str(script), str(txt)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            if shutil.which('lilypond'):
                ly = Path(tmp) / 'score.ly'
                ly.write_text(result.stdout)
                render = subprocess.run(['lilypond', '-o', str(Path(tmp) / 'score'), str(ly)],
                                        capture_output=True, text=True, env={**os.environ, 'TMPDIR': tmp})
                self.assertEqual(render.returncode, 0, render.stderr)

    def test_same_measure_ties_keep_fragments_without_mutating_source(self):
        score, part, measure = stream.Score(), stream.Part(), stream.Measure(number=1)
        measure.insert(0, meter.TimeSignature('4/4'))
        measure.append([tied_note('C4', 2, 'start'), tied_note('C4', 2, 'stop')])
        part.append(measure)
        score.insert(0, part)
        extracted, _ = extract_jianpu_measures(score, 0)
        self.assertEqual([(n.symbol, n.duration) for n in extracted[0]], [('1', 2.0), ('-', 2.0)])
        self.assertEqual([n.duration.quarterLength for n in measure.notes], [2, 2])

    def test_cross_measure_tie_retains_continuation(self):
        score, part = stream.Score(), stream.Part()
        first, second = stream.Measure(number=1), stream.Measure(number=2)
        first.insert(0, meter.TimeSignature('4/4'))
        first.append(tied_note('C4', 4, 'start'))
        second.append(tied_note('C4', 4, 'stop'))
        part.append([first, second])
        score.insert(0, part)
        extracted, _ = extract_jianpu_measures(score, 0)
        self.assertEqual([[(n.symbol, n.duration) for n in m] for m in extracted],
                         [[('1', 4.0)], [('-', 4.0)]])
