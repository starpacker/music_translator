"""Only unambiguous same-pitch paired curves may be recovered as ties."""
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch
from core.notation.tie_reconstruction import reconstruct_ties_in_musicxml
from jianpu import omr_to_musicxml


def n(pitch, slur=''):
    notation = f'<notations>{slur}</notations>' if slur else ''
    return (f'<note><pitch><step>{pitch}</step><octave>4</octave></pitch>'
            f'<duration>2</duration><type>half</type>{notation}</note>')


XML = ('<?xml version="1.0"?><score-partwise version="4.0"><part-list><score-part id="P1">'
       '<part-name>Flute</part-name></score-part></part-list><part id="P1">'
       '<measure number="1"><attributes><divisions>1</divisions><time><beats>4</beats>'
       '<beat-type>4</beat-type></time></attributes>'
       + n('C', '<slur type="start" number="1"/>') + n('C', '<slur type="stop" number="1"/>') + '</measure>'
       '<measure number="2">' + n('D', '<slur type="start" number="1"/>') + n('D', '<slur type="stop" number="2"/>') + '</measure>'
       '<measure number="3">' + n('E', '<slur type="start" number="3"/>') + n('F', '<slur type="stop" number="3"/>') + '</measure>'
       '<measure number="4">' + n('G') + n('G') + '</measure></part></score-partwise>')


class PairedTieRecoveryTest(unittest.TestCase):
    def test_only_matching_paired_slurs_are_recovered(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'score.musicxml'
            path.write_text(XML)
            self.assertEqual(reconstruct_ties_in_musicxml(path, paired_slurs_only=True), 1)
            root = ET.parse(path).getroot()
            measures = root.findall('./part/measure')
            self.assertEqual(len(root.findall('.//note')), 8)
            self.assertEqual([x.get('type') for x in measures[0].findall('.//note/tie')], ['start', 'stop'])
            self.assertEqual(len(measures[0].findall('.//slur')), 0)
            self.assertEqual(len(measures[1].findall('.//slur')), 2)
            self.assertEqual(len(measures[2].findall('.//slur')), 2)
            self.assertEqual(len(measures[3].findall('.//tie')), 0)
            self.assertEqual(reconstruct_ties_in_musicxml(path, paired_slurs_only=True), 0)

    def test_omr_cache_recovers_ties_without_changing_raw_omr(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, work = root / 'score.pdf', root / 'work'
            source.write_bytes(b'input')
            work.mkdir()
            def recognize(src, out, use_gpu_inference=None):
                (out / 'score.musicxml').write_text(XML)
                return out
            with patch('homr_shim.install'), patch('core.omr.homr_runner.run_homr_batch', side_effect=recognize):
                cached = omr_to_musicxml(source, work)
            self.assertEqual(len(ET.parse(cached).findall('.//tie')), 2)
            raw = work / 'score.omr' / 'score.musicxml'
            self.assertEqual(raw.read_text(), XML)
