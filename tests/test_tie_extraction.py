"""Tie handling must work with the installed music21 API."""
import unittest
from music21 import meter, note, stream, tie
from core.notation.jianpu import extract_jianpu_measures


def tied_note(pitch, duration, kind):
    event = note.Note(pitch, quarterLength=duration)
    event.tie = tie.Tie(kind)
    return event


class TieExtractionTest(unittest.TestCase):
    def test_same_measure_ties_are_merged_without_mutating_source(self):
        score, part, measure = stream.Score(), stream.Part(), stream.Measure(number=1)
        measure.insert(0, meter.TimeSignature('4/4'))
        measure.append([tied_note('C4', 2, 'start'), tied_note('C4', 2, 'stop')])
        part.append(measure)
        score.insert(0, part)
        extracted, _ = extract_jianpu_measures(score, 0)
        self.assertEqual([(n.symbol, n.duration) for n in extracted[0]], [('1', 4.0)])
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
