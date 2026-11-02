"""Distinct short tuplet onsets must never be merged as one onset."""
import unittest
from fractions import Fraction
from music21 import note, stream
from core.notation.jianpu.extract import _canonical_offset, extract_jianpu_measures


class OnsetPrecisionTest(unittest.TestCase):
    def test_only_floating_point_noise_is_deduplicated(self):
        seen = {}
        first = _canonical_offset(seen, 1 / 24)
        self.assertEqual(first, _canonical_offset(seen, 1 / 24 + 1e-10))
        self.assertNotEqual(first, _canonical_offset(seen, 1 / 12))
        self.assertEqual(len(seen), 2)

    def test_short_tuplet_notes_keep_their_order(self):
        score = stream.Score()
        part = stream.Part()
        measure = stream.Measure(number=1)
        for i, pitch in enumerate(('C4', 'D4', 'E4')):
            event = note.Note(pitch)
            event.duration.quarterLength = Fraction(1, 24)
            measure.insert(Fraction(i, 24), event)
        part.append(measure)
        score.insert(0, part)
        measures, _ = extract_jianpu_measures(score, 0)
        actual = [n.midi for n in measures[0] if not n.is_rest and not n.is_grace]
        self.assertEqual(actual, [60, 62, 64])
