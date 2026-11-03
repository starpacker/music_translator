"""Regression: a nearly full-height note stem is not a barline."""
import unittest
import pymupdf
from core.omr.multirest_collapse import page_system_slot_kinds
from insitu.pdf_structure import StaffMetrics, extract_staves, staff_barlines


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


class EngravingScaleTest(unittest.TestCase):
    """The same score engraved at a smaller size must still be detected.

    A source can be exported at any staff size: the bundled 梆笛 score uses a
    5.13pt staff space, while a MuseScore 4 export of the same music uses
    3.72pt.  All tolerances are derived from the measured staff space, so the
    reduced engraving must yield the same staves and barlines.
    """

    @staticmethod
    def _page(spacing: float, overshoot: float):
        doc = pymupdf.open()
        page = doc.new_page(width=600, height=800)
        top = 100.0
        for index in range(5):
            y = top + index * spacing
            page.draw_line((40, y), (540, y), width=0.4)
        bottom = top + 4 * spacing
        for x in (300.0, 540.0):
            page.draw_line((x, top), (x, bottom + overshoot * spacing), width=0.4)
        doc.reload_page(page)
        return doc, top, bottom

    def test_reduced_engraving_is_detected(self):
        doc, top, bottom = self._page(3.72, 0.047)
        staves = extract_staves(doc[0], 1)
        self.assertEqual(len(staves), 1)
        self.assertAlmostEqual(staves[0].line_spacing, 3.72, places=2)
        self.assertAlmostEqual(staves[0].top, top, places=2)
        self.assertAlmostEqual(staves[0].bottom, bottom, places=2)
        self.assertEqual(len(staff_barlines(doc[0], staves[0])), 2)
        doc.close()

    def test_barline_without_overshoot_still_matches(self):
        # MuseScore 4 exports end a barline exactly on the bottom rule; the
        # scaled overshoot is part of a tolerance, not a required endpoint.
        doc, _, _ = self._page(3.72, 0.0)
        staves = extract_staves(doc[0], 1)
        self.assertEqual(len(staff_barlines(doc[0], staves[0])), 2)
        doc.close()

    def test_metrics_follow_staff_spacing(self):
        small = StaffMetrics.from_spacing(3.72)
        large = StaffMetrics.from_spacing(5.13)
        self.assertAlmostEqual(large.barline_overshoot, 0.24, places=2)
        self.assertAlmostEqual(large.barline_merge, 5.49, places=2)
        self.assertLess(small.barline_overshoot, large.barline_overshoot)
        self.assertLess(small.rest_count_top, large.rest_count_top)
        self.assertLess(small.repeat_margin, large.repeat_margin)
