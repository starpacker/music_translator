"""Per-line in-situ rendering must not invent an end-of-piece barline."""
import shutil
import tempfile
import unittest
from pathlib import Path

import pymupdf


class NormalLineEndBarlineTest(unittest.TestCase):
    def test_final_barline_becomes_ordinary_barline(self):
        from insitu.render_insitu import _normal_line_end_barline

        for command in (r'\bar "|."', '\\bar\n  "|."'):
            with self.subTest(command=command):
                text = 'c1 | ' + command + ' }'
                self.assertEqual(_normal_line_end_barline(text), 'c1 | \\bar "|" }')

    def test_repeat_barlines_are_left_alone(self):
        from insitu.render_insitu import _normal_line_end_barline

        text = (r'\bar ".|:" c1 | \bar ":|." d1 | '
                r'\bar ":|.|:" e1 | \bar "|."')
        self.assertEqual(_normal_line_end_barline(text),
                         text.replace(r'\bar "|."', r'\bar "|"'))

    def test_synthetic_final_after_repeat_is_removed(self):
        from insitu.render_insitu import _normal_line_end_barline

        for repeat in (r'\bar ".|:"', r'\bar ":|."', r'\bar ":|.|:"'):
            with self.subTest(repeat=repeat):
                prefix = 'c1 ' + repeat + ' |'
                text = prefix + ' \\bar "|." }'
                # A following ordinary bar command would override the repeat
                # at the same musical moment, even if both remain in the text.
                self.assertEqual(_normal_line_end_barline(text), prefix + ' }')

    def test_source_without_final_barline_is_unchanged(self):
        from insitu.render_insitu import _normal_line_end_barline

        text = r'c1 | \bar "||" d1 | \bar "|"'
        self.assertEqual(_normal_line_end_barline(text), text)


@unittest.skipUnless(shutil.which('lilypond'), 'LilyPond is required by the renderer')
class RenderedLineEndBarlineTest(unittest.TestCase):
    @staticmethod
    def _vertical_strokes(page):
        # LilyPond encodes barlines as vertical stroked paths, whose bounding
        # rectangles have zero width regardless of the actual stroke width.
        return sorted(
            (drawing for drawing in page.get_drawings()
             if drawing['type'] == 's' and drawing['rect'].height >= 8
             and drawing['rect'].width < 0.01),
            key=lambda drawing: drawing['rect'].x0,
        )

    def test_tile_ends_with_single_thin_barline(self):
        from insitu.render_insitu import (_jianpu_ly_source, _render_ly_tile,
                                          _run_jianpu_ly)

        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            source = _jianpu_ly_source([['1 2 3 4', '5 6 7 1']],
                                       '1=C', '4/4', 'test')
            ly = _run_jianpu_ly(source, work)
            # The shared preprocessor still emits its normal final barline;
            # only the in-situ tile renderer should change it.
            self.assertIn(r'\bar "|."', ly)
            tile, ink = _render_ly_tile(ly, work)
            written = (work / 'line.ly').read_text(encoding='utf-8')
            self.assertNotIn(r'\bar "|."', written)
            self.assertIn(r'\bar "|"', written)
            with pymupdf.open(tile) as doc:
                strokes = self._vertical_strokes(doc[0])
                # One internal barline and one line-end barline, both thin;
                # a final barline would add a third, thicker stroke.
                self.assertEqual(len(strokes), 2)
                self.assertAlmostEqual(strokes[0]['width'], strokes[1]['width'], places=3)
                self.assertLessEqual(strokes[-1]['rect'].x1, ink.x1)

    def test_line_end_repeat_is_not_overridden_by_ordinary_barline(self):
        from core.render.jianpu_runner import _insert_repeat_bar_commands
        from insitu.render_insitu import (_jianpu_ly_source, _render_ly_tile,
                                          _run_jianpu_ly)

        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            source = _jianpu_ly_source([['1 2 3 4', '5 6 7 1']],
                                       '1=C', '4/4', 'test')
            ly = _run_jianpu_ly(source, work)
            ly = _insert_repeat_bar_commands(ly, {1: {'end': True}})
            tile, _ = _render_ly_tile(ly, work)
            written = (work / 'line.ly').read_text(encoding='utf-8')
            self.assertIn(r'\bar ":|."', written)
            self.assertNotIn(r'\bar "|."', written)
            with pymupdf.open(tile) as doc:
                strokes = self._vertical_strokes(doc[0])
                self.assertEqual(len(strokes), 3)
                self.assertGreater(strokes[-1]['width'], 2 * strokes[-2]['width'])
