"""Regression: chord MusicXML -> jianpu tokens -> LilyPond pitch spelling."""
import importlib.util
import os
from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
SUMISORA = Path(os.environ.get('SUMISORA_ROOT',
    str(Path(__file__).resolve().parents[1])))
sys.path.insert(0, str(SUMISORA))
from music21 import chord, converter
from core.notation.jianpu import note_to_jianpu, jianpu_note_token


class ShengChordRenderingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('sheng_jly', SUMISORA / 'scripts/jianpu-ly.py')
        cls.jly = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.jly)
        cls.source = cls.jly.process_input('1=C\nOctavesAfter\n135 - - -\n')

    def test_real_score_chords_keep_musicxml_octaves_through_parser(self):
        source = ROOT / 'tests/fixtures/sheng_original.musicxml'
        self.assertTrue(source.exists())
        measures = ET.parse(source).findall('./part/measure')
        expected = [('F4', 'D4', 'B-4'), ('G4', 'E-4', 'B-3')]
        score = converter.parse(source)
        for index, pitches, lily in zip((1, 2), expected,
                ("< d' f' ais' >", "< ais dis' g' >")):
            with self.subTest(measure=index + 1):
                xml_pitches = tuple(n.findtext('pitch/step')
                    + ('-' if n.findtext('pitch/alter') == '-1' else '')
                    + n.findtext('pitch/octave')
                    for n in measures[index].findall('note[pitch]'))
                self.assertEqual(xml_pitches, pitches)
                c = next(iter(score.parts[0].measure(index + 1).recurse().getElementsByClass(chord.Chord)))
                token = jianpu_note_token(note_to_jianpu(c, 0)).split()[0]
                self.assertEqual(self.jly.chordNotes_markup(token, token, token)[2], lily)

    def test_mixed_octaves_and_chromatic_chords(self):
        for pitches, lily in [
            (('C4', 'E4', 'G4'), "< c' e' g' >"),
            (('C3', 'E4', 'G5'), "< c e' g'' >"),
            (('D4', 'F4', 'B-4'), "< d' f' ais' >"),
        ]:
            with self.subTest(pitches=pitches):
                token = jianpu_note_token(note_to_jianpu(chord.Chord(pitches), 0))
                self.assertEqual(self.jly.chordNotes_markup(token, token, token)[2], lily)

    def test_hidden_jianpu_staff_also_hides_ledger_lines(self):
        self.assertIn("\\override LedgerLineSpanner #'stencil = ##f", self.source)


if __name__ == '__main__':
    unittest.main()
