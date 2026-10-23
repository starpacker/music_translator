"""Source-geometry regressions for merged and missing OMR chord bars."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.environ.get('SUMISORA_ROOT',
    str(Path(__file__).resolve().parents[1])))
import pymupdf
from core.omr.vector_whole_chords import whole_chords_on_page, recover_vector_whole_chords
from core.omr.multirest_collapse import collapse_page_multirests
from core.notation.jianpu import extract_jianpu_measures
from music21 import converter


class VectorChordTest(unittest.TestCase):
    def test_source_pitches_use_bar_geometry_and_displaced_seconds(self):
        with pymupdf.open(ROOT / 'tests/fixtures/sheng.pdf') as doc:
            chords, count = whole_chords_on_page(doc[0])
        self.assertEqual(count, 25)
        self.assertEqual(chords[3], [('C', 0, 4), ('E', -1, 4), ('A', -1, 4)])
        self.assertEqual(chords[4], [('D', -1, 4), ('F', 0, 4), ('A', -1, 4)])
        self.assertEqual(chords[5], [('C', 0, 4), ('D', 0, 4), ('G', 0, 4)])
        self.assertNotIn(8, chords)  # Two half-note chords must stay untouched.

    def test_real_page_repair_preserves_other_bars_and_is_idempotent(self):
        src = ROOT / 'tests/fixtures/sheng_page1.musicxml'
        with tempfile.TemporaryDirectory() as tmp:
            dst = Path(tmp) / 'score.musicxml'
            dst.write_bytes(src.read_bytes())
            before = ET.parse(dst).findall('./part/measure')
            recover_vector_whole_chords(ROOT / 'tests/fixtures/sheng.pdf', dst)
            after = ET.parse(dst).findall('./part/measure')
            self.assertEqual(len(before), len(after))
            self.assertEqual(ET.tostring(before[8]), ET.tostring(after[8]))
            for m in after[3:6]:
                notes = m.findall('note')
                self.assertEqual(len(notes), 3)
                self.assertEqual(sum(int(n.findtext('duration')) for n in notes
                                     if n.find('chord') is None), 8)
            extracted, _ = extract_jianpu_measures(converter.parse(dst), 0)
            self.assertEqual([sum(n.duration for n in m) for m in extracted[3:6]], [4, 4, 4])
            original = dst.read_bytes()
            self.assertEqual(recover_vector_whole_chords(ROOT / 'tests/fixtures/sheng.pdf', dst), 0)
            self.assertEqual(dst.read_bytes(), original)

    def test_ambiguous_alignment_must_not_invent_note_bar_rests(self):
        part = ET.fromstring('<part><measure number="1"><note><pitch><step>C</step>'
            '<octave>4</octave></pitch><duration>8</duration></note></measure></part>')
        original = ET.tostring(part)
        with self.assertRaisesRegex(ValueError, 'Ambiguous'):
            collapse_page_multirests(part, [['N', 'N']])
        self.assertEqual(ET.tostring(part), original)


if __name__ == '__main__':
    unittest.main()
