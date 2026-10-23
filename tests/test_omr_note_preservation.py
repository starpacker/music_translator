"""Regressions for note loss caused by unreliable OMR time signatures."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

SUMISORA = Path(os.environ.get('SUMISORA_ROOT',
    str(Path(__file__).resolve().parents[1])))
sys.path.insert(0, str(SUMISORA))
from core.render.lilypond_runner import _fix_omr_artifacts_in_mxl
from core.notation.jianpu import extract_jianpu_measures
from music21 import converter


def note(duration, pitch='C', kind='16th'):
    return (f'<note><pitch><step>{pitch}</step><octave>5</octave></pitch>'
            f'<duration>{duration}</duration><voice>1</voice><type>{kind}</type></note>')


class NotePreservationTest(unittest.TestCase):
    def test_inherited_and_explicit_meters_never_truncate_content(self):
        # Force an independent, valid artifact removal so this tests the XML
        # writer as well as the unchanged-path case.
        phantom = ('<forward><duration>12</duration></forward>'
                   '<note><rest measure="yes"/><duration>12</duration>'
                   '<voice>9</voice></note><backup><duration>24</duration></backup>')
        for signature in ('', '<attributes><time><beats>3</beats>'
                             '<beat-type>4</beat-type></time></attributes>'):
            with self.subTest(explicit=bool(signature)), tempfile.TemporaryDirectory() as tmp:
                src = Path(tmp) / 'source.musicxml'
                src.write_text('<?xml version="1.0"?>\n<score-partwise version="4.0">'
                    '<part-list><score-part id="P1"><part-name>Flute</part-name>'
                    '</score-part></part-list><part id="P1">'
                    '<measure number="81"><attributes><divisions>12</divisions>'
                    '<time><beats>3</beats><beat-type>4</beat-type></time></attributes>'
                    + note(36, kind='half') + '</measure><measure number="82">'
                    + signature + '<note><rest/><duration>24</duration><voice>1</voice>'
                    '<type>half</type></note>'
                    + ''.join(note(3, p) for p in 'AACC' * 2) + phantom
                    + '</measure><measure number="83">'
                    + ''.join(note(3, p) for p in 'CDEF' * 4)
                    + '</measure><measure number="84">' + note(48, kind='whole')
                    + '</measure></part></score-partwise>')
                fixed = _fix_omr_artifacts_in_mxl(src, Path(tmp))
                measures = ET.parse(fixed).findall('./part/measure')
                self.assertEqual([len(m.findall('note[pitch]')) for m in measures], [1, 8, 16, 1])
                for before, after in zip(ET.parse(src).findall('./part/measure'), measures):
                    self.assertEqual([ET.tostring(n) for n in before.findall('note[pitch]')],
                                     [ET.tostring(n) for n in after.findall('note[pitch]')])
                self.assertFalse(measures[1].findall('forward'))
                extracted, _ = extract_jianpu_measures(converter.parse(fixed), 0)
                self.assertEqual([sum(n.duration for n in m) for m in extracted[1:]], [4, 4, 4])
                self.assertEqual([sum(not n.is_rest for n in m) for m in extracted[1:3]], [8, 16])

    def test_actual_bangdi_measures(self):
        source = Path(__file__).resolve().parents[1] / 'tests/fixtures/bangdi.musicxml'
        if not source.exists():
            self.skipTest('Saved Bangdi OMR input is unavailable')
        with tempfile.TemporaryDirectory() as tmp:
            fixed = _fix_omr_artifacts_in_mxl(source, Path(tmp))
            measures = ET.parse(fixed).findall('./part/measure')
            self.assertEqual(len(measures), 162)
            self.assertEqual([len(m.findall('note[pitch]')) for m in measures[81:84]], [8, 16, 1])
            extracted, _ = extract_jianpu_measures(converter.parse(fixed), 0)
            self.assertEqual([sum(n.duration for n in m) for m in extracted[81:84]], [4, 4, 4])


if __name__ == '__main__':
    unittest.main()
