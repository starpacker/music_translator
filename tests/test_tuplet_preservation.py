"""Common tuplets retain their written rhythm and actual duration."""
from fractions import Fraction
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from music21 import duration, meter, note, stream
from core.notation.jianpu import extract_jianpu_measures, jianpu_note_token
from core.notation.jianpu.measure import measure_units
from core.notation.jianpu.extract import _mark_tuplet_groups
from core.config import JianpuNote
from scripts.audit_jianpu_events import audit


class TupletPreservationTest(unittest.TestCase):
    def test_explicit_groups_split_at_a_new_start(self):
        events = [JianpuNote('1', '', 0, 0, 1/3, 0, 60, False,
                             tuplet_actual=3, tuplet_normal=2, tuplet_type=kind)
                  for kind in ('start', None, 'start', 'stop')]
        _mark_tuplet_groups(events)
        self.assertEqual([event.tuplet_start for event in events], [True, False, True, False])
        self.assertEqual([event.tuplet_stop for event in events], [False, True, False, True])

    def test_eighth_note_triplet_is_not_shortened_or_padded(self):
        score, part, measure = stream.Score(), stream.Part(), stream.Measure(number=1)
        measure.insert(0, meter.TimeSignature('4/4'))
        for i, pitch in enumerate(('C4', 'D4', 'E4')):
            event = note.Note(pitch)
            event.duration.quarterLength = Fraction(1, 3)
            measure.insert(Fraction(i, 3), event)
        measure.insert(1, note.Rest(quarterLength=3))
        part.append(measure)
        score.insert(0, part)
        measures, _ = extract_jianpu_measures(score, 0)
        events = measures[0]
        self.assertEqual(len(events), 4)
        self.assertEqual([n.midi for n in events[:3]], [60, 62, 64])
        self.assertTrue(all(abs(n.duration - 1/3) < 1e-7 for n in events[:3]))
        self.assertEqual([jianpu_note_token(n) for n in events], ['3[ q1', 'q2', 'q3 ]', '0 - -'])
        self.assertEqual(measure_units(events), 64)
        with tempfile.TemporaryDirectory() as tmp:
            mxl = Path(tmp) / 'score.musicxml'
            score.write('musicxml', fp=mxl)
            self.assertEqual(audit(mxl)['mismatch_count'], 0)
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
                import pymupdf
                source = Path(tmp) / 'source.pdf'
                doc = pymupdf.open()
                page = doc.new_page(width=600, height=400)
                for y in (100, 105, 110, 115, 120):
                    page.draw_line((40, y), (540, y), width=.5)
                page.draw_line((540, 100), (540, 120), width=.5)
                doc.save(source)
                doc.close()
                entry = Path(__file__).resolve().parents[1] / 'jianpu.py'
                cli = subprocess.run([sys.executable, str(entry), str(source), '--mxl', str(mxl),
                                      '--mode', 'both', '-o', str(Path(tmp) / 'output')],
                                     capture_output=True, text=True, env={**os.environ, 'TMPDIR': tmp}, timeout=90)
                self.assertEqual(cli.returncode, 0, cli.stdout + cli.stderr)
                self.assertIn('placed 1/1 lines', cli.stdout)
                self.assertNotIn('标准 jianpu-ly 路径失败', cli.stdout)
                self.assertTrue((Path(tmp) / 'output.jianpu.pdf').is_file())
                self.assertTrue((Path(tmp) / 'output.insitu.pdf').is_file())

    def test_triplet_can_include_rest_and_two_adjacent_groups(self):
        score, part, measure = stream.Score(), stream.Part(), stream.Measure(number=1)
        measure.insert(0, meter.TimeSignature('4/4'))
        for i in range(6):
            event = note.Rest() if i == 1 else note.Note(('C4', 'D4', 'E4', 'F4', 'G4', 'A4')[i])
            event.duration.quarterLength = Fraction(1, 3)
            measure.insert(Fraction(i, 3), event)
        measure.insert(2, note.Rest(quarterLength=2))
        part.append(measure)
        score.insert(0, part)
        events = extract_jianpu_measures(score, 0)[0][0]
        tokens = [jianpu_note_token(event) for event in events]
        self.assertEqual(tokens[:6], ['3[ q1', 'q0', 'q3 ]', '3[ q4', 'q5', 'q6 ]'])
        self.assertEqual(measure_units(events), 64)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'score.musicxml'
            score.write('musicxml', fp=path)
            self.assertEqual(audit(path)['mismatch_count'], 0)

    def test_sixty_fourth_note_triplet_uses_four_underlines(self):
        score, part, measure = stream.Score(), stream.Part(), stream.Measure(number=1)
        for i, pitch in enumerate(('C4', 'D4', 'E4')):
            event = note.Note(pitch)
            event.duration.quarterLength = Fraction(1, 24)
            measure.insert(Fraction(i, 24), event)
        part.append(measure)
        score.insert(0, part)
        events = extract_jianpu_measures(score, 0)[0][0]
        self.assertEqual([jianpu_note_token(n) for n in events], ['3[ h1', 'h2', 'h3 ]'])
        self.assertEqual(measure_units(events), 2)

    def test_nonstandard_tuplet_ratio_is_preserved(self):
        score, part, measure = stream.Score(), stream.Part(), stream.Measure(number=1)
        measure.insert(0, meter.TimeSignature('4/4'))
        for i, pitch in enumerate(('C4', 'D4', 'E4')):
            event = note.Note(pitch, type='eighth')
            event.duration.appendTuplet(duration.Tuplet(3, 1))
            measure.insert(Fraction(i, 6), event)
        measure.insert(Fraction(1, 2), note.Rest(quarterLength=3))
        measure.insert(Fraction(7, 2), note.Rest(quarterLength=Fraction(1, 2)))
        part.append(measure)
        score.insert(0, part)
        events = extract_jianpu_measures(score, 0)[0][0]
        tokens = [jianpu_note_token(n) for n in events]
        self.assertEqual(tokens[:3], ['3:1[ q1', 'q2', 'q3 ]'])
        self.assertEqual(measure_units(events), 64)
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'line.txt'
            source.write_text('1=C\n4/4\n' + ' '.join(tokens) + ' |\n')
            script = Path(__file__).resolve().parents[1] / 'scripts/jianpu-ly.py'
            result = subprocess.run([sys.executable, str(script), str(source)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('\\times 1/3', result.stdout)
