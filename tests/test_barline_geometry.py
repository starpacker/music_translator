"""Regression: a nearly full-height note stem is not a barline."""
import unittest
import pymupdf
from core.omr.multirest_collapse import page_system_slot_kinds
from insitu.pdf_structure import extract_staves, staff_barlines


class BarlineGeometryTest(unittest.TestCase):
    def test_near_staff_height_stem_is_not_a_barline(self):
        doc = pymupdf.open()
        page = doc.new_page(width=600, height=800)
        for y in (100, 105, 110, 115, 120):
            page.draw_line((40, y), (540, y), width=0.5)
        for x in (300, 540):
            page.draw_line((x, 100), (x, 120), width=0.5)
        page.draw_line((120, 100.36), (120, 119.64), width=0.5)
        doc.reload_page(page)
        page = doc[0]
        self.assertEqual(page_system_slot_kinds(page), [['N', 'N']])
        staves = extract_staves(page, 1)
        self.assertEqual(len(staves), 1)
        self.assertEqual(len(staff_barlines(page, staves[0])), 2)
        doc.close()


if __name__ == '__main__':
    unittest.main()

class StaffAndRepeatTest(unittest.TestCase):
    def test_duplicate_rule_does_not_hide_staff(self):
        doc = pymupdf.open()
        page = doc.new_page(width=600, height=800)
        for y in (100, 105, 110, 115, 120):
            page.draw_line((40, y), (540, y), width=0.5)
        page.draw_line((200, 110), (530, 110), width=1.0)
        doc.reload_page(page)
        self.assertEqual(len(extract_staves(doc[0], 1)), 1)
        doc.close()

    def test_opening_repeat_is_not_a_measure(self):
        doc = pymupdf.open()
        page = doc.new_page(width=600, height=800)
        for y in (100, 105, 110, 115, 120):
            page.draw_line((40, y), (540, y), width=0.5)
        page.draw_line((65, 100), (65, 120), width=0.5)
        page.draw_line((68, 100), (68, 120), width=2.0)
        for x in (300, 540):
            page.draw_line((x, 100), (x, 120), width=0.5)
        doc.reload_page(page)
        staff = extract_staves(doc[0], 1)[0]
        self.assertEqual(len(staff_barlines(doc[0], staff)), 2)
        doc.close()
