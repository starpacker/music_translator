"""Measure-aligned vector placement must not stretch glyphs when gaps suffice."""
import shutil
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pymupdf


@unittest.skipUnless(shutil.which('lilypond'), 'LilyPond is required by the renderer')
class CellAlignmentTest(unittest.TestCase):
    def test_each_barline_lands_under_its_source_boundary(self):
        from insitu.render_insitu import _place_tile
        from insitu.pdf_structure import Staff, StaffLayout
        with tempfile.TemporaryDirectory() as tmp:
            tile = Path(tmp) / 'tile.pdf'
            source = pymupdf.open()
            page = source.new_page(width=300, height=100)
            for x in (10, 80, 180):
                page.draw_line((x, 10), (x, 30), width=1)
            for x in (30, 110):
                page.draw_rect(pymupdf.Rect(x, 15, x + 8, 25), fill=(0, 0, 0), width=0)
            source.save(tile)
            source.close()
            output = pymupdf.open()
            page = output.new_page(width=350, height=100)
            layout = StaffLayout(Staff(1, 0, 20, 20, 300, 5), [120, 300])
            result = _place_tile(page, tile, pymupdf.Rect(9, 9, 181, 31), layout, 40, 2)
            self.assertIn(result, ('aligned', 'scaled'))
            pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), colorspace=pymupdf.csGRAY, alpha=False)
            data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.stride)
            for x in (20, 120, 300):
                self.assertLess(data[100, x * 2 - 2:x * 2 + 3].min(), 100, f'missing bar at {x}')
            output.close()

    def test_bar_detection_and_elastic_spacing(self):
        from insitu.render_insitu import _tile_barlines, _place_cell
        source = pymupdf.open()
        page = source.new_page(width=300, height=100)
        for x in (10, 80, 180):
            page.draw_line((x, 10), (x, 30), width=.5)
        page.draw_rect(pymupdf.Rect(25, 15, 35, 25), fill=(0, 0, 0), width=0)
        page.draw_rect(pymupdf.Rect(50, 15, 60, 25), fill=(0, 0, 0), width=0)
        source.reload_page(page)
        ink = pymupdf.Rect(9, 9, 181, 31)
        bars = _tile_barlines(source[0], ink, 2)
        self.assertEqual(len(bars), 3)
        self.assertEqual([round(x) for x in bars], [10, 80, 180])
        target = pymupdf.open()
        output = target.new_page(width=350, height=100)
        original = pymupdf.Rect(10, 9, 80, 31)
        desired = pymupdf.Rect(100, 40, 240, 62)
        self.assertTrue(_place_cell(output, source, original, desired))
        pix = output.get_pixmap(colorspace=pymupdf.csGRAY, alpha=False)
        row = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.stride)[51]
        indices = np.flatnonzero(row[105:235] < 100) + 105
        runs = np.split(indices, np.where(np.diff(indices) > 1)[0] + 1)
        widths = [len(r) for r in runs if len(r) >= 5]
        self.assertEqual(widths, [10, 10])
        compressed = target.new_page(width=350, height=100)
        self.assertTrue(_place_cell(compressed, source, original, pymupdf.Rect(100, 40, 150, 62)))
        pix = compressed.get_pixmap(colorspace=pymupdf.csGRAY, alpha=False)
        row = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.stride)[51]
        indices = np.flatnonzero(row[105:145] < 100) + 105
        runs = np.split(indices, np.where(np.diff(indices) > 1)[0] + 1)
        self.assertEqual([len(r) for r in runs if len(r) >= 5], [10, 10])
        target.close()
        source.close()
