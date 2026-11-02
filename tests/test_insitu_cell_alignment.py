"""Measure-aligned vector placement must not stretch glyphs when gaps suffice."""
import shutil
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pymupdf


@unittest.skipUnless(shutil.which('lilypond'), 'LilyPond is required by the renderer')
class CellAlignmentTest(unittest.TestCase):
    def test_real_lilypond_tie_retains_digit_width(self):
        from insitu.render_insitu import (_jianpu_ly_source, _place_cell, _render_ly_tile,
                                          _run_jianpu_ly, _tile_barlines)
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            measures = ["1' - - -", '- - - 6', "3' - - -", "- - - 2'",
                        "3' 6 1' -", "- - - 2'", '7 - - -', '- - - 6']
            ly = _run_jianpu_ly(_jianpu_ly_source([measures], '', '4/4', 'test'), work)
            tile, ink = _render_ly_tile(ly, work)
            with pymupdf.open(tile) as source:
                bars = _tile_barlines(source[0], ink, len(measures))
                self.assertIsNotNone(bars)
                clip = pymupdf.Rect(bars[2], ink.y0, bars[3], ink.y1)
                target = pymupdf.open()
                page = target.new_page(width=200, height=100)
                self.assertTrue(_place_cell(page, source, clip,
                                            pymupdf.Rect(20, 20, 72.2, 20 + clip.height)))
                # The numerical glyph should retain its natural width even
                # though the arc over it has been fitted to a narrower bar.
                def glyph_width(page, area):
                    pix = page.get_pixmap(matrix=pymupdf.Matrix(4, 4), clip=area,
                                          colorspace=pymupdf.csGRAY, alpha=False)
                    pixels = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.stride)[:, :pix.width]
                    columns = np.flatnonzero(np.any(pixels < 100, axis=0))
                    return int(columns[-1] - columns[0] + 1)
                original_glyph = pymupdf.Rect(clip.x0 + 4, 38.7, clip.x0 + 12, 46.7)
                # Restrict the destination crop to the first digit, away from
                # the barline and rhythm dashes.
                result_glyph = pymupdf.Rect(22, 20 + 38.7 - clip.y0, 31.5, 20 + 46.7 - clip.y0)
                self.assertLessEqual(abs(glyph_width(source[0], original_glyph)
                                         - glyph_width(page, result_glyph)), 1)
                target.close()

    def test_slur_band_can_shrink_without_distorting_digits(self):
        from insitu.render_insitu import _place_cell, _slur_seam
        source = pymupdf.open()
        page = source.new_page(width=160, height=80)
        for x in (10, 110):
            page.draw_line((x, 7), (x, 39), width=.5)
        shape = page.new_shape()
        shape.draw_bezier((20, 16), (35, 6), (85, 6), (100, 16))
        shape.finish(color=(0, 0, 0), width=.6, closePath=False)
        shape.commit()
        for x in (20, 43, 66, 89):
            page.draw_rect(pymupdf.Rect(x, 23, x + 9, 32), color=None, fill=(0, 0, 0))
        clip = pymupdf.Rect(10, 5, 110, 40)
        seam = _slur_seam(page, clip, 90)
        self.assertIsNotNone(seam)
        target = pymupdf.open()
        output = target.new_page(width=160, height=80)
        self.assertTrue(_place_cell(output, source, clip, pymupdf.Rect(20, 20, 110, 55)))
        pix = output.get_pixmap(matrix=pymupdf.Matrix(2, 2), colorspace=pymupdf.csGRAY, alpha=False)
        data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.stride)
        row = data[2 * (20 + 23 - 5 + 4)]
        indices = np.flatnonzero(row[44:216] < 100) + 44
        runs = np.split(indices, np.where(np.diff(indices) > 1)[0] + 1)
        self.assertEqual([len(run) for run in runs if len(run) >= 10], [18, 18, 18, 18])
        self.assertTrue(np.any(data[45:65, 60:180] < 100))  # curve remains visible
        target.close()
        source.close()

    def test_slur_partition_rejects_other_ink_above_seam(self):
        from insitu.render_insitu import _slur_seam
        source = pymupdf.open()
        page = source.new_page(width=160, height=80)
        shape = page.new_shape()
        shape.draw_bezier((20, 16), (35, 6), (85, 6), (100, 16))
        shape.finish(color=(0, 0, 0), width=.6, closePath=False)
        shape.commit()
        for x in (20, 43, 66, 89):
            page.draw_rect(pymupdf.Rect(x, 23, x + 9, 32), color=None, fill=(0, 0, 0))
        page.draw_rect(pymupdf.Rect(30, 6, 34, 9), color=None, fill=(0, 0, 0))
        self.assertIsNone(_slur_seam(page, pymupdf.Rect(10, 5, 110, 40), 80))
        source.close()

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
