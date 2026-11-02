"""End-to-end check that the expanded layout reserves room for each line."""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
import json
import hashlib
from unittest import mock
from pathlib import Path
import pymupdf
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class PiecewiseCutTest(unittest.TestCase):
    def test_empty_pixel_row_inside_one_object_is_not_a_safe_cut(self):
        from insitu.render_insitu import _blank_cut, _cut_has_ink
        source = pymupdf.open()
        page = source.new_page(width=200, height=160)
        # One vector symbol with disconnected upper and lower strokes.
        shape = page.new_shape()
        shape.draw_rect(pymupdf.Rect(80, 50, 90, 60))
        shape.draw_rect(pymupdf.Rect(80, 85, 90, 95))
        shape.finish(color=None, fill=(0, 0, 0))
        shape.commit()
        pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), colorspace=pymupdf.csGRAY, alpha=False)
        pixels = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.stride)
        self.assertTrue(np.all(pixels[2 * 72, :pix.width] >= 245))
        self.assertTrue(_cut_has_ink(page, 72))
        cut = _blank_cut(page, 40, 110)
        self.assertTrue(_cut_has_ink(page, cut))
        from insitu.render_insitu import _piecewise_blank_cut
        safe, overrides = _piecewise_blank_cut(page, 40, 110)
        self.assertFalse(_cut_has_ink(page, safe, overrides))
        self.assertTrue(safe <= 50 or safe >= 95)
        page = source.new_page(width=200, height=160)
        shape = page.new_shape()
        shape.draw_line((80, 50), (80, 60))
        shape.draw_line((80, 85), (80, 95))
        shape.finish(color=(0, 0, 0), width=.5)
        shape.commit()
        self.assertTrue(_cut_has_ink(page, 72), 'zero-width path bounds must also be protected')
        page = source.new_page(width=200, height=160)
        page.insert_text((80, 105), 'i', fontsize=60)
        pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), colorspace=pymupdf.csGRAY, alpha=False)
        pixels = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.stride)
        ys = np.flatnonzero(np.any(pixels[:, :pix.width] < 245, axis=1))
        parts = np.split(ys, np.where(np.diff(ys) > 1)[0] + 1)
        self.assertEqual(len(parts), 2)
        between = (parts[0][-1] + parts[1][0]) / 4
        self.assertTrue(_cut_has_ink(page, between), 'the dot and stem belong to the same glyph')
        source.close()

    def test_piecewise_cut_preserves_disjoint_interleaved_source_ink(self):
        from insitu.render_insitu import (_blank_cut, _copy_with_column_cuts,
                                          _cut_has_ink, _piecewise_blank_cut)
        source = pymupdf.open()
        page = source.new_page(width=200, height=160)
        page.draw_rect(pymupdf.Rect(25, 35, 50, 90), color=None, fill=(0, 0, 0))
        page.draw_rect(pymupdf.Rect(120, 65, 150, 120), color=None, fill=(0, 0, 0))
        self.assertTrue(_cut_has_ink(page, _blank_cut(page, 50, 105)))
        result = _piecewise_blank_cut(page, 50, 105)
        self.assertIsNotNone(result)
        cut, overrides = result
        self.assertTrue(overrides)
        output = pymupdf.open()
        target = output.new_page(width=200, height=200)
        _copy_with_column_cuts(target, source, 0, [(cut, 40, 0, None, None, None, None, overrides)])
        pix = target.get_pixmap(matrix=pymupdf.Matrix(2, 2), colorspace=pymupdf.csGRAY, alpha=False)
        pixels = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.stride)
        # Each rectangle must remain a single continuous rectangle at its
        # original size, even though they receive different translations.
        for x in (35, 130):
            ys = np.flatnonzero(pixels[:, x * 2] < 128)
            self.assertEqual(len(ys), 110)
            self.assertTrue(np.all(np.diff(ys) == 1))
        output.close()
        source.close()

    def test_no_safe_cut_through_ink_spanning_the_whole_gap(self):
        from insitu.render_insitu import _piecewise_blank_cut
        source = pymupdf.open()
        page = source.new_page(width=200, height=160)
        page.draw_rect(pymupdf.Rect(50, 35, 60, 120), color=None, fill=(0, 0, 0))
        self.assertIsNone(_piecewise_blank_cut(page, 50, 105))
        source.close()

    def test_tempo_moves_with_next_staff_while_previous_dynamic_stays(self):
        from insitu.render_insitu import _copy_with_column_cuts, _cut_has_ink, _tempo_cut_overrides
        with tempfile.TemporaryDirectory() as tmp:
            src_path = Path(tmp) / 'source.pdf'
            source = pymupdf.open()
            page = source.new_page(width=600, height=400)
            page.insert_text((290, 80), 'q = 46', fontsize=12)
            page.insert_text((150, 147), 'mp', fontsize=12)
            page.insert_text((300, 165), 'q = 56', fontsize=12)
            page.draw_line((40, 190), (560, 190), width=.5)
            source.save(src_path)
            source.close()
            source = pymupdf.open(src_path)
            overrides = _tempo_cut_overrides(source[0], 125, 168, 190)
            self.assertEqual(len(overrides), 1)
            left, right, cut = overrides[0]
            self.assertLess(cut, 165)
            self.assertLess(left, 300)
            self.assertGreater(right, 330)
            self.assertFalse(_cut_has_ink(source[0], 168, overrides))
            output = pymupdf.open()
            target = output.new_page(width=600, height=450)
            _copy_with_column_cuts(target, source, 0,
                                   [(168, 50, 0, None, None, None, 0, overrides)])
            for marker in ('q = 46', 'q = 56', 'mp'):
                self.assertEqual(target.get_text().count(marker), 1)
            # The two source markings are disjoint in x. Their pixels must
            # survive exactly, but receive different vertical translations.
            a = source[0].get_pixmap(matrix=pymupdf.Matrix(2, 2), colorspace=pymupdf.csGRAY, alpha=False)
            b = target.get_pixmap(matrix=pymupdf.Matrix(2, 2), colorspace=pymupdf.csGRAY, alpha=False)
            original = np.frombuffer(a.samples, dtype=np.uint8).reshape(a.height, a.stride)
            result = np.frombuffer(b.samples, dtype=np.uint8).reshape(b.height, b.stride)
            np.testing.assert_array_equal(original[260:300, 280:360], result[260:300, 280:360])
            np.testing.assert_array_equal(original[300:340, 590:690], result[400:440, 590:690])
            # The following staff is shifted uniformly across the column seams.
            np.testing.assert_array_equal(original[375:385, 80:1120], result[475:485, 80:1120])
            output.close()
            source.close()


@unittest.skipUnless(shutil.which('lilypond'), 'LilyPond is required')
class SpacedInsituTest(unittest.TestCase):
    def test_interleaved_marks_use_piecewise_cut_in_real_build(self):
        import insitu.render_insitu as renderer
        from test_vector_tempos import make_sources
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original, xml = make_sources(root)
            source, output = root / 'interleaved.pdf', root / 'out.insitu.pdf'
            with pymupdf.open(original) as doc:
                page = doc[0]
                page.draw_rect(pymupdf.Rect(200, 125, 215, 155), color=None, fill=(0, 0, 0))
                page.draw_rect(pymupdf.Rect(400, 145, 415, 180), color=None, fill=(0, 0, 0))
                doc.save(source)
            renderer.build(source, xml, output, expand=True)
            report = json.loads((root / 'out.insitu.report.json').read_text())
            self.assertEqual(report['status'], 'needs_review')
            self.assertEqual(report['piecewise_source_cut_count'], 1)
            self.assertEqual(report['rows'][0]['source_cut'], 'piecewise')
            self.assertTrue(report['rows'][0]['rendered'])
            with pymupdf.open(output) as doc:
                pix = doc[0].get_pixmap(matrix=pymupdf.Matrix(2, 2), colorspace=pymupdf.csGRAY, alpha=False)
                pixels = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.stride)
                for x, height in [(207, 60), (407, 70)]:
                    ys = np.flatnonzero(pixels[:, 2 * x] < 128)
                    runs = np.split(ys, np.where(np.diff(ys) > 1)[0] + 1)
                    self.assertTrue(any(len(run) == height for run in runs), (x, [len(run) for run in runs]))

    def test_unsafe_cut_keeps_source_and_marks_row_incomplete(self):
        import insitu.render_insitu as renderer
        from test_vector_tempos import make_sources
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, xml = make_sources(root)
            output = root / 'out.insitu.pdf'
            with mock.patch.object(renderer, '_cut_has_ink', return_value=True), \
                 mock.patch.object(renderer, '_piecewise_blank_cut', return_value=None):
                renderer.build(source, xml, output, expand=True)
            report = json.loads((root / 'out.insitu.report.json').read_text())
            self.assertEqual(report['status'], 'incomplete')
            self.assertEqual(report['rendered_staff_count'], 0)
            self.assertEqual(report['rows'][0]['voices'][0]['error'], 'no_safe_source_cut')
            with pymupdf.open(source) as original, pymupdf.open(output) as result:
                before = original[0].get_pixmap(colorspace=pymupdf.csGRAY, alpha=False)
                after = result[0].get_pixmap(colorspace=pymupdf.csGRAY, alpha=False)
                self.assertEqual(before.samples, after.samples)
                self.assertEqual(len(result), 2)
                self.assertEqual(report['rows'][0]['voices'][0]['detached_page'], 2)
                self.assertIn('Source page 1, staff 1', result[1].get_text())

    def test_spaced_layout_preserves_source_and_inserts_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf = root / 'source.pdf'
            xml = root / 'source.musicxml'
            doc = pymupdf.open()
            page = doc.new_page(width=600, height=500)
            for top in (100, 260):
                for dy in (0, 5, 10, 15, 20):
                    page.draw_line((40, top + dy), (540, top + dy), width=.5)
                page.draw_line((540, top), (540, top + 20), width=.5)
            page.insert_text((200, 145), 'Tempo 146')
            doc.save(pdf)
            doc.close()
            xml.write_text('''<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="4.0"><part-list><score-part id="P1"><part-name>Flute</part-name></score-part></part-list>
<part id="P1"><measure number="1"><attributes><divisions>1</divisions><key><fifths>0</fifths></key>
<time><beats>4</beats><beat-type>4</beat-type></time><clef><sign>G</sign><line>2</line></clef></attributes>
<note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration><type>whole</type></note></measure>
<measure number="2"><print new-system="yes"/><note><pitch><step>D</step><octave>4</octave></pitch>
<duration>4</duration><type>whole</type></note></measure></part></score-partwise>''', encoding='utf-8')
            # A legal cross-system wedge continuation previously made the
            # music21 parser reject the entire in-situ conversion.
            content = xml.read_text()
            content = content.replace('<note><pitch><step>C</step>',
                '<direction><direction-type><wedge type="crescendo"/></direction-type></direction>'
                '<note><pitch><step>C</step>', 1)
            content = content.replace('<print new-system="yes"/>',
                '<print new-system="yes"/><direction><direction-type>'
                '<wedge type="continue"/></direction-type></direction>', 1)
            content = content.replace('</note></measure></part></score-partwise>',
                '</note><direction><direction-type><wedge type="stop"/></direction-type></direction>'
                '</measure></part></score-partwise>', 1)
            xml.write_text(content)
            result = subprocess.run([sys.executable, str(ROOT / 'jianpu.py'), str(pdf), '--mxl', str(xml),
                                     '--mode', 'insitu', '-o', str(root / 'out')], cwd=ROOT,
                                    capture_output=True, text=True, env={**os.environ, 'TMPDIR': tmp}, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('placed 2/2 lines', result.stdout)
            self.assertEqual(xml.read_text(), content)
            report = json.loads((root / 'out.insitu.report.json').read_text())
            self.assertEqual(report['staff_count'], 2)
            self.assertEqual(report['rendered_staff_count'], 2)
            self.assertEqual(report['incomplete_staff_count'], 0)
            self.assertEqual(len(report['rows']), 2)
            self.assertTrue(all(row['voices'][0]['placement'] in ('aligned', 'scaled', 'line')
                                for row in report['rows']))
            self.assertEqual(report['source_musicxml_sha256'], hashlib.sha256(xml.read_bytes()).hexdigest())
            self.assertEqual(report['output_pdf_sha256'], hashlib.sha256((root / 'out.insitu.pdf').read_bytes()).hexdigest())
            with pymupdf.open(root / 'out.insitu.pdf') as output:
                self.assertEqual(len(output), 1)
                self.assertGreaterEqual(output[0].rect.height, 568)
                self.assertIn('Tempo 146', output[0].get_text())
                # Check rendered pixels: MuPDF's drawing extractor also
                # exposes clipped-away strokes inside reused PDF forms.
                pix = output[0].get_pixmap(matrix=pymupdf.Matrix(2, 2),
                                           colorspace=pymupdf.csGRAY, alpha=False)
                def dark_near(x, y):
                    return any(pix.samples[yy * pix.stride + x * 2] < 180
                               for yy in range(int(y * 2 - 2), int(y * 2 + 3)))
                self.assertTrue(dark_near(50, 100))
                self.assertFalse(dark_near(50, 260))
                self.assertTrue(any(dark_near(50, y) for y in range(294, 361)))
            mxl = root / 'compressed.mxl'
            with zipfile.ZipFile(mxl, 'w') as archive:
                archive.writestr('META-INF/container.xml', '''<?xml version="1.0"?>
<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles>
<rootfile full-path="score.musicxml" media-type="application/vnd.recordare.musicxml+xml"/>
</rootfiles></container>''')
                archive.writestr('score.musicxml', xml.read_bytes())
            result = subprocess.run([sys.executable, str(ROOT / 'jianpu.py'), str(pdf), '--mxl', str(mxl),
                                     '--mode', 'insitu', '-o', str(root / 'compressed')], cwd=ROOT,
                                    capture_output=True, text=True, env={**os.environ, 'TMPDIR': tmp}, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('placed 2/2 lines', result.stdout)

    def test_failed_row_is_reported_as_incomplete_but_pdf_is_saved(self):
        from insitu.render_insitu import build
        from test_vector_tempos import make_sources
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, xml = make_sources(root)
            output = root / 'out.insitu.pdf'
            with mock.patch('insitu.render_insitu._run_jianpu_ly', side_effect=RuntimeError('test failure')):
                build(source, xml, output, expand=True)
            report = json.loads((root / 'out.insitu.report.json').read_text())
            self.assertTrue(output.is_file())
            self.assertEqual(report['status'], 'incomplete')
            self.assertEqual(report['staff_count'], 1)
            self.assertEqual(report['rendered_staff_count'], 0)
            self.assertEqual(report['incomplete_staff_count'], 1)
            self.assertEqual(report['rows'][0]['voices'][0]['error'], 'test failure')

    def test_interrupted_run_does_not_leave_previous_success_report(self):
        from insitu.render_insitu import build
        from test_vector_tempos import make_sources
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, xml = make_sources(root)
            report = root / 'out.insitu.report.json'
            report.write_text('{"status": "rendered"}')
            with mock.patch('insitu.render_insitu.align_from_files', side_effect=RuntimeError('test interruption')):
                with self.assertRaisesRegex(RuntimeError, 'test interruption'):
                    build(source, xml, root / 'out.insitu.pdf', expand=True)
            self.assertEqual(json.loads(report.read_text())['status'], 'in_progress')

    def test_missing_middle_row_does_not_let_cut_cross_its_staff(self):
        import insitu.render_insitu as renderer
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, xml = root / 'score.pdf', root / 'score.musicxml'
            doc = pymupdf.open()
            page = doc.new_page(width=600, height=350)
            for top in (100, 145, 230):
                for dy in (0, 5, 10, 15, 20):
                    page.draw_line((40, top + dy), (540, top + dy), width=.5)
                page.draw_line((540, top), (540, top + 20), width=.5)
            doc.save(source)
            doc.close()
            xml.write_text('''<score-partwise version="4.0"><part-list><score-part id="P1">
<part-name>Flute</part-name></score-part></part-list><part id="P1">
<measure number="1"><attributes><divisions>1</divisions><key><fifths>0</fifths></key>
<time><beats>4</beats><beat-type>4</beat-type></time><clef><sign>G</sign><line>2</line></clef>
</attributes><note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration><type>whole</type></note></measure>
<measure number="2"><print new-system="yes"/><note><pitch><step>D</step><octave>4</octave></pitch>
<duration>4</duration><type>whole</type></note></measure>
<measure number="3"><print new-system="yes"/><note><pitch><step>E</step><octave>4</octave></pitch>
<duration>4</duration><type>whole</type></note></measure></part></score-partwise>''')
            original = renderer._run_jianpu_ly
            calls = []
            def render_line(text, workdir):
                calls.append(workdir)
                if len(calls) == 2:
                    raise RuntimeError('middle row unavailable')
                return original(text, workdir)
            with mock.patch.object(renderer, '_run_jianpu_ly', side_effect=render_line), \
                 mock.patch.object(renderer, '_blank_cut', wraps=renderer._blank_cut) as cuts:
                output = root / 'out.insitu.pdf'
                renderer.build(source, xml, output, expand=True)
            report = json.loads((root / 'out.insitu.report.json').read_text())
            self.assertEqual(report['status'], 'incomplete')
            self.assertEqual([row['rendered'] for row in report['rows']], [True, False, True])
            self.assertLessEqual(cuts.call_args_list[0].args[2], 140)
            with pymupdf.open(output) as result:
                pix = result[0].get_pixmap(matrix=pymupdf.Matrix(2, 2), colorspace=pymupdf.csGRAY, alpha=False)
                pixels = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.stride)
                ys = np.flatnonzero(pixels[:, 100] < 180)
                centers = [int(run[len(run) // 2]) for run in np.split(ys, np.where(np.diff(ys) > 1)[0] + 1) if len(run)]
                self.assertTrue(any(all(any(abs(y - (start + 10 * j)) <= 1 for y in centers)
                                        for j in range(5))
                                    for start in centers if 2 * 179 <= start < 2 * 264),
                                'the skipped middle staff should remain five continuous lines')
