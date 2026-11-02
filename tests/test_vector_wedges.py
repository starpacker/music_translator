"""PDF hairpins require verified note boundaries and preserve source data."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import xml.etree.ElementTree as ET

import pymupdf
import jianpu
import core.omr.vector_wedges as recovery
from core.notation.musicxml_wedges import read_wedges
from test_vector_tempos import make_sources


def add_lines(pdf, x0=100, x1=200, split=False):
    doc = pymupdf.open(pdf)
    page = doc[0]
    if split:
        page.draw_line((x0, 132), (x1, 128), width=.7)
        page.draw_line((x0, 132), (x1, 136), width=.7)
    else:
        shape = page.new_shape()
        shape.draw_line((x0, 132), (x1, 128))
        shape.draw_line((x0, 132), (x1, 136))
        shape.finish(color=(0, 0, 0), width=.7, closePath=False)
        shape.commit()
    output = pdf.with_name('with-wedge.pdf')
    doc.save(output)
    doc.close()
    return output


class VectorWedgeTest(unittest.TestCase):
    def discover(self, pdf, mxl, heads=None):
        if heads is None:
            heads = [(100, 125, 20), (200, 122.5, 20)]
        with mock.patch.object(recovery, '_page_marks', return_value=([], heads, [])):
            return recovery.discover_pdf_wedges(pdf, mxl)

    def test_exact_pair_is_written_without_modifying_notes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf, mxl = make_sources(root)
            pdf = add_lines(pdf)
            before = mxl.read_bytes()
            with mock.patch.object(recovery, '_page_marks', return_value=([], [(100, 125, 20), (200, 122.5, 20)], [])):
                out = jianpu.recover_pdf_wedges(pdf, mxl, mxl, root, 'out')
            self.assertNotEqual(out, mxl)
            self.assertEqual(before, mxl.read_bytes())
            self.assertEqual([ET.tostring(e) for e in ET.parse(mxl).findall('.//note')],
                             [ET.tostring(e) for e in ET.parse(out).findall('.//note')])
            wedges, issues = read_wedges(out)
            self.assertFalse(issues)
            self.assertEqual(len(wedges[0]), 1)
            report = json.loads((root / 'out.pdf-wedges.json').read_text())
            self.assertEqual(report['inserted'][0]['end_offset_quarters'], '2')

    def test_mismatched_notes_ambiguous_endpoint_and_continuation_are_reported(self):
        for x0, x1, heads, reason in [
            (100, 200, [(100, 130, 20), (200, 122.5, 20)], 'unverified_measures'),
            (150, 200, [(100, 125, 20), (200, 122.5, 20)], 'ambiguous_endpoints'),
            (45, 200, [(100, 125, 20), (200, 122.5, 20)], 'possible_system_continuation'),
        ]:
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as tmp:
                pdf, mxl = make_sources(Path(tmp))
                candidates, report = self.discover(add_lines(pdf, x0, x1), mxl, heads)
                self.assertFalse(candidates)
                self.assertIn(reason, [item['reason'] for item in report])

    def test_split_lines_are_only_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdf, mxl = make_sources(Path(tmp))
            candidates, report = self.discover(add_lines(pdf, split=True), mxl)
            self.assertFalse(candidates)
            self.assertIn('split_strokes_or_possible_continuation', [item['reason'] for item in report])

    def test_legal_continuation_does_not_break_pdf_analysis(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf, mxl = make_sources(root)
            tree = ET.parse(mxl)
            measure = tree.find('./part/measure')
            notes = measure.findall('note')
            for position, kind in ((list(measure).index(notes[0]), 'crescendo'),
                                   (list(measure).index(notes[1]) + 1, 'continue'),
                                   (len(measure) + 2, 'stop')):
                direction = ET.Element('direction')
                dtype = ET.SubElement(direction, 'direction-type')
                ET.SubElement(dtype, 'wedge', {'type': kind})
                measure.insert(position, direction)
            tree.write(mxl)
            original = mxl.read_bytes()
            candidates, report = self.discover(add_lines(pdf), mxl)
            self.assertEqual(len(candidates), 1, report)
            self.assertEqual(mxl.read_bytes(), original)

    def test_existing_wedges_and_invalid_boundary_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, mxl = make_sources(root)
            out = root / 'derived.musicxml'
            candidate = {'start_measure_index': 1, 'start_offset_quarters': '0',
                         'end_measure_index': 1, 'end_offset_quarters': '2', 'kind': 'crescendo'}
            inserted, skipped = recovery.write_enriched_wedges(mxl, out, [candidate])
            self.assertEqual(len(inserted), 1)
            self.assertFalse(skipped)
            inserted, skipped = recovery.write_enriched_wedges(out, root / 'other.xml', [candidate])
            self.assertFalse(inserted)
            self.assertEqual(skipped[0]['reason'], 'existing_musicxml_wedges')
            candidate['end_offset_quarters'] = '1/3'
            inserted, skipped = recovery.write_enriched_wedges(mxl, root / 'invalid.xml', [candidate])
            self.assertFalse(inserted)
            self.assertEqual(skipped[0]['reason'], 'unrepresentable_boundary')


if __name__ == '__main__':
    unittest.main()
