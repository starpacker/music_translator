"""Every recognised measure must be assigned exactly once."""
import contextlib
import io
import unittest
from insitu.align import align


class AlignmentTest(unittest.TestCase):
    def test_split_and_drifting_systems_conserve_measures(self):
        for pdf, omr in [([2, 2], [1, 1, 2]), ([2, 2], [2, 4]),
                         ([3, 3, 3], [9]), ([2, 3], [1, 1])]:
            with self.subTest(pdf=pdf, omr=omr), contextlib.redirect_stdout(io.StringIO()):
                slices = align(pdf, omr)
                self.assertEqual(len(slices), len(pdf))
                indices = [i for s in slices for i in range(s.omr_start, s.omr_end)]
                self.assertEqual(indices, list(range(sum(omr))))

class CompressedMusicXMLTest(unittest.TestCase):
    def test_mxl_container_and_new_page_break(self):
        import tempfile
        import zipfile
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch
        from insitu.align import align_from_files

        xml = b'''<score-partwise xmlns="http://www.musicxml.org/ns/musicxml"><part id="P1"><measure number="1"/>
                  <measure number="2"><print new-page="yes"/></measure></part></score-partwise>'''
        container = b'''<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
                       <rootfiles><rootfile full-path="scores/main.musicxml"/></rootfiles></container>'''
        layouts = [SimpleNamespace(barlines=[1]), SimpleNamespace(barlines=[1])]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'score.mxl'
            with zipfile.ZipFile(path, 'w') as archive:
                archive.writestr('META-INF/container.xml', container)
                archive.writestr('scores/main.musicxml', xml)
            with patch('insitu.pdf_structure.layout_for_document', return_value=layouts):
                _, slices = align_from_files('dummy.pdf', path)
            self.assertEqual([(s.omr_start, s.omr_end) for s in slices], [(0, 1), (1, 2)])
            self.assertEqual([s.expected_measures for s in slices], [1, 1])
