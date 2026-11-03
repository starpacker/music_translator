"""Rebuild slurs from engraved PDF crescents without touching the source XML."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

from core.omr.vector_slurs import write_enriched_slurs

NOTES = ('<note><pitch><step>{step}</step><octave>4</octave></pitch>'
         '<duration>1</duration><type>quarter</type>{marks}</note>')


def make_musicxml(path: Path, steps='CDEF', marks=None):
    marks = marks or [''] * len(steps)
    return ('''<score-partwise version="4.0"><part-list><score-part id="P1">
<part-name>Flute</part-name></score-part></part-list><part id="P1"><measure number="1">
<attributes><divisions>1</divisions><key><fifths>0</fifths></key>
<time><beats>4</beats><beat-type>4</beat-type></time>
<clef><sign>G</sign><line>2</line></clef></attributes>'''
            + ''.join(NOTES.format(step=step, marks=mark) for step, mark in zip(steps, marks))
            + '</measure></part></score-partwise>')


def link(start, stop, start_order, stop_order):
    return SimpleNamespace(start_measure=0, start_note=start, stop_measure=0,
                           stop_note=stop, start_order=start_order,
                           stop_order=stop_order)


def slurs(path: Path):
    part = ET.parse(path).getroot().find('part')
    return [(measure_index, note_index, slur.get('type'), slur.get('number'))
            for measure_index, measure in enumerate(part.findall('measure'))
            for note_index, note in enumerate(measure.findall('note'))
            for notations in note.findall('notations')
            for slur in notations.findall('slur')]


class EnrichedSlurWriterTest(unittest.TestCase):
    def test_replaces_omr_slurs_with_measured_endpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'score.musicxml'
            output = root / 'derived.musicxml'
            source.write_text(make_musicxml(
                source, marks=['<notations><slur type="start" number="1"/></notations>',
                               '', '<notations><slur type="stop" number="1"/></notations>', '']))
            result = write_enriched_slurs(source, output, [link(0, 2, 0, 2)])
            self.assertEqual(result, {'removed_slurs': 2, 'added_marks': 2, 'slurs': 1})
            self.assertEqual(slurs(output), [(0, 0, 'start', '1'), (0, 2, 'stop', '1')])
            # The original MusicXML is never modified.
            self.assertIn('type="stop"', source.read_text())

    def test_overlapping_slurs_get_distinct_numbers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'score.musicxml'
            output = root / 'derived.musicxml'
            source.write_text(make_musicxml(source, 'CDEF'))
            write_enriched_slurs(source, output, [link(0, 2, 0, 2), link(2, 3, 2, 3)])
            marks = slurs(output)
            self.assertIn((0, 2, 'start', '2'), marks)
            self.assertIn((0, 2, 'stop', '1'), marks)

    def test_refuses_to_overwrite_the_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'score.musicxml'
            source.write_text(make_musicxml(source))
            with self.assertRaisesRegex(ValueError, '不能覆盖'):
                write_enriched_slurs(source, source, [link(0, 1, 0, 1)])


if __name__ == '__main__':
    unittest.main()
