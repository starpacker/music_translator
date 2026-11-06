"""Fixed-pitch jianpu can use any reference note, not only C."""
import unittest

from music21 import key, meter, note, stream

from core.notation.jianpu import build_jianpu_ly_text, parse_score_to_jianpu


def d_major_measure():
    score, part = stream.Score(), stream.Part()
    measure = stream.Measure(number=1)
    measure.insert(0, key.KeySignature(2))
    measure.insert(0, meter.TimeSignature('4/4'))
    for beat, pitch in enumerate(('D4', 'E4', 'F#4', 'G4')):
        measure.insert(beat, note.Note(pitch, quarterLength=1))
    part.append(measure)
    score.insert(0, part)
    return score


def music_line(text):
    return next(line for line in text.splitlines() if '|' in line)


class FixedTonicTest(unittest.TestCase):
    def test_default_fixed_mode_still_uses_c(self):
        text = build_jianpu_ly_text(d_major_measure(), 'test', pitch_mode='fixed')
        self.assertIn('1=C', text)
        self.assertTrue(music_line(text).startswith('2 '))

    def test_fixed_tonic_d_numbers_d_as_one(self):
        text = build_jianpu_ly_text(d_major_measure(), 'test', pitch_mode='fixed',
                                    fixed_tonic='D')
        self.assertIn('1=D', text)
        self.assertNotIn('1=C', text)
        self.assertEqual(music_line(text).split(' |')[0],
                         '1 ^\\markup{\\concat{\\hspace#3"4/4"}} 2 3 4')

    def test_parse_score_to_jianpu_reports_the_reference(self):
        measures, header_lines, _ = parse_score_to_jianpu(
            d_major_measure(), pitch_mode='fixed', fixed_tonic='D')
        self.assertEqual(len(measures), 1)
        self.assertTrue(header_lines[0].startswith('1=D'))

    def test_invalid_reference_is_rejected(self):
        with self.assertRaisesRegex(ValueError, '固定调主音'):
            build_jianpu_ly_text(d_major_measure(), 'test', pitch_mode='fixed',
                                 fixed_tonic='H')


if __name__ == '__main__':
    unittest.main()
