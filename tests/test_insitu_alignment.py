"""Every recognised measure must be assigned exactly once."""
import contextlib
import io
import unittest
from insitu.align import align
from pathlib import Path


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


class PageBoundaryTest(unittest.TestCase):
    """A page turn must never hand one page's measures to another page."""

    def test_truncated_second_page_does_not_steal_first_page_measures(self):
        # 05_中音笙1-2 in miniature: page 1 engraves 6+6 measures and the OMR
        # reads them exactly; page 2 engraves 7+7 but the OMR stops after 2.
        from types import SimpleNamespace
        from unittest.mock import patch
        from insitu.align import align_from_files
        import tempfile

        xml = b'''<score-partwise xmlns="http://www.musicxml.org/ns/musicxml"><part id="P1">
          <measure number="1"/><measure number="2"/>
          <measure number="3"><print new-system="yes"/></measure>
          <measure number="4"/>
          <measure number="5"><print new-page="yes"/></measure>
          <measure number="6"/></part></score-partwise>'''
        layouts = [SimpleNamespace(staff=SimpleNamespace(page=1), barlines=[1] * 2),
                   SimpleNamespace(staff=SimpleNamespace(page=1), barlines=[1] * 2),
                   SimpleNamespace(staff=SimpleNamespace(page=2), barlines=[1] * 3),
                   SimpleNamespace(staff=SimpleNamespace(page=2), barlines=[1] * 3)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'score.musicxml'
            path.write_bytes(xml)
            with patch('insitu.pdf_structure.layout_for_document', return_value=layouts), \
                    contextlib.redirect_stdout(io.StringIO()):
                _, slices = align_from_files('dummy.pdf', path)
        # Page 1's OMR is complete and must stay with page 1's staves.
        self.assertEqual([(s.omr_start, s.omr_end) for s in slices[:2]],
                         [(0, 2), (2, 4)])
        # Page 2 is truncated to two measures; the deficit stays on page 2
        # instead of pulling page 1's bars forward.
        self.assertEqual(slices[2].omr_start, 4)
        self.assertEqual(slices[-1].omr_end, 6)
        self.assertEqual([s.omr_measures for s in slices[2:]], [1, 1])

    def test_page_merge_marks_the_page_turn(self):
        from core.omr.homr_runner import _merge_homr_musicxml_pages
        import tempfile
        import xml.etree.ElementTree as ET

        def page(numbers):
            measures = ''.join(f'<measure number="{n}"/>' for n in numbers)
            return (b'<score-partwise><part id="P1">' + measures.encode()
                    + b'</part></score-partwise>')

        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / 'p1.musicxml'
            second = Path(tmp) / 'p2.musicxml'
            merged = Path(tmp) / 'merged.musicxml'
            first.write_bytes(page([1, 2]))
            second.write_bytes(page([1, 2]))
            self.assertTrue(_merge_homr_musicxml_pages([first, second], merged))
            root = ET.parse(str(merged)).getroot()
            measures = root.findall('part/measure')
            self.assertEqual(len(measures), 4)
            # The first measure of page 2 must announce the page turn, even
            # though Homr gave its own first measure no <print> element.
            self.assertEqual(measures[2].find('print').get('new-page'), 'yes')
            self.assertIsNone(measures[0].find('print'))
