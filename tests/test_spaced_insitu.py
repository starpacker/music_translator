"""Whole-object reflow: source-sized pages, intact notation, no appendix fallback."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

import numpy as np
import pymupdf as fitz

from insitu.pdf_structure import Staff, StaffLayout
from insitu.reflow import SourceObjects, _programs, paginate

ROOT = Path(__file__).resolve().parents[1]


def make_sources(root, tops=(70, 140, 210), height=300):
    pdf, xml = root / 'score.pdf', root / 'score.musicxml'
    with fitz.open() as doc:
        page = doc.new_page(width=600, height=height)
        for i, top in enumerate(tops):
            for dy in (0, 5, 10, 15, 20):
                page.draw_line((40, top + dy), (540, top + dy), width=.5)
            page.draw_line((540, top), (540, top + 20), width=.5)
            page.insert_text((180, top + 32), f'OriginalRow{i + 1}', fontsize=8)
        doc.save(pdf)
    measures = []
    for i in range(len(tops)):
        attrs = ('''<attributes><divisions>1</divisions><key><fifths>0</fifths></key>
<time><beats>4</beats><beat-type>4</beat-type></time><clef><sign>G</sign><line>2</line></clef>
</attributes>''' if i == 0 else '<print new-system="yes"/>')
        measures.append(f'''<measure number="{i + 1}">{attrs}<note><pitch><step>{'CDEFGAB'[i % 7]}</step>
<octave>4</octave></pitch><duration>4</duration><type>whole</type></note></measure>''')
    xml.write_text('''<score-partwise version="4.0"><part-list><score-part id="P1">
<part-name>Flute</part-name></score-part></part-list><part id="P1">'''
                   + ''.join(measures) + '</part></score-partwise>', encoding='utf-8')
    return pdf, xml


class SourceObjectsTest(unittest.TestCase):
    def test_pdf_tokens_do_not_split_literal_strings_or_names(self):
        first = b'q BT (q \\( Q \\) /Q) Tj <7151> Tj ET Q'
        second = b'q /q /Q % Q q\n 1 0 0 rg Q'
        self.assertEqual(_programs(b'q ' + first + b' ' + second + b' Q'), [first, second])
        self.assertEqual(_programs(first), [first])
        with self.assertRaisesRegex(ValueError, 'unbalanced'):
            _programs(b'q q Q')

    def test_stroked_text_and_disconnected_paths_replay_intact(self):
        with fitz.open() as doc, fitz.open() as out:
            page = doc.new_page(width=600, height=300)
            for y in (80, 85, 90, 95, 100):
                page.draw_line((40, y), (540, y), width=.5)
            page.insert_text((100, 55), 'Heading q (Q)', fontsize=22, render_mode=2, border_width=.2)
            shape = page.new_shape()
            shape.draw_rect(fitz.Rect(70, 110, 80, 120))
            shape.draw_rect(fitz.Rect(70, 145, 80, 155))
            shape.finish(color=None, fill=(0, 0, 0))
            shape.commit()
            layouts = [(0, StaffLayout(Staff(1, 80, 100, 40, 540, 5), [540]))]
            objects = SourceObjects(doc, 0, layouts)
            try:
                target = out.new_page(width=600, height=300)
                objects.place(target, objects.graphics)
                a = page.get_pixmap(colorspace=fitz.csGRAY)
                b = target.get_pixmap(colorspace=fitz.csGRAY)
                self.assertEqual(a.samples, b.samples)
                self.assertFalse(objects.outlined)
                rectangles = [g for g in objects.rows[0].programs if g.bounds.height >= 45]
                self.assertEqual(len(rectangles), 1, 'the disconnected symbol must remain indivisible')
            finally:
                objects.close()

    def test_paginate_keeps_mixed_source_sizes_and_all_text(self):
        with fitz.open() as doc, fitz.open() as out, tempfile.TemporaryDirectory() as tmp:
            tiles = {}
            layouts, reports = [], []
            for i, (width, height) in enumerate(((600, 260), (620, 280))):
                page = doc.new_page(width=width, height=height)
                for j, top in enumerate((60, 150)):
                    n = i * 2 + j
                    for dy in (0, 5, 10, 15, 20):
                        page.draw_line((40, top + dy), (540, top + dy), width=.5)
                    page.insert_text((150, top + 32), f'Row{n}', fontsize=8)
                    layouts.append(StaffLayout(Staff(i + 1, top, top + 20, 40, 540, 5), [540]))
                    tiles[n] = [(None, fitz.Rect(0, 0, 180, 50))]
                    reports.append({})
            def place(page, index, row_tiles, top):
                page.insert_text((40, top + 12), f'Jianpu{index}')
                return top + row_tiles[0][1].height
            pages = paginate(doc, layouts, tiles, reports, out, place)
            self.assertGreater(len(out), len(doc))
            for page, info in zip(out, pages):
                self.assertEqual(page.rect, doc[info['source_page'] - 1].rect)
            text = ''.join(p.get_text() for p in out)
            for i in range(4):
                self.assertEqual(text.count(f'Row{i}'), 1)
                self.assertEqual(text.count(f'Jianpu{i}'), 1)
                self.assertEqual(reports[i]['source_placement'], 'whole_objects')
                self.assertLess(reports[i]['output_source_bounds'][1], out[reports[i]['output_page'] - 1].rect.height)

    def test_consecutive_same_size_source_pages_flow_without_a_blank_page(self):
        """Source page 2 continues on page 1 when page 1 has room (A4 printing)."""
        with fitz.open() as doc, fitz.open() as out:
            layouts, reports, tiles = [], [], {}
            for page_index, tops in enumerate(((60, 160), (60,))):
                page = doc.new_page(width=600, height=600)
                for top in tops:
                    n = len(layouts)
                    for dy in (0, 5, 10, 15, 20):
                        page.draw_line((40, top + dy), (540, top + dy), width=.5)
                    page.insert_text((150, top + 32), f'Row{n}', fontsize=8)
                    layouts.append(StaffLayout(Staff(page_index + 1, top, top + 20, 40, 540, 5), [540]))
                    tiles[n] = [(None, fitz.Rect(0, 0, 180, 40))]
                    reports.append({})
            def place(page, index, row_tiles, top):
                page.insert_text((40, top + 12), f'Jianpu{index}')
                return top + row_tiles[0][1].height
            pages = paginate(doc, layouts, tiles, reports, out, place)
            # Both source pages share one output page; no blank page is started.
            self.assertEqual(len(out), 1)
            self.assertEqual(pages[0]['source_pages'], [1, 2])
            self.assertEqual(out[0].rect, fitz.Rect(0, 0, 600, 600))
            text = ''.join(p.get_text() for p in out)
            for i in range(3):
                self.assertEqual(text.count(f'Row{i}'), 1)
                self.assertEqual(text.count(f'Jianpu{i}'), 1)

    def test_wrapped_continuation_pages_never_clip_an_oversized_tile(self):
        with fitz.open() as doc, fitz.open() as out:
            page = doc.new_page(width=600, height=200)
            for y in (40, 45, 50, 55, 60):
                page.draw_line((40, y), (540, y))
            layouts = [StaffLayout(Staff(1, 40, 60, 40, 540, 5), [540])]
            reports = [{}]
            tiles = {0: [(None, fitz.Rect(0, 0, 200, 50)) for _ in range(4)]}
            def place(page, index, row_tiles, top):
                return top + row_tiles[0][1].height
            paginate(doc, layouts, tiles, reports, out, place)
            self.assertGreater(len(out), 1)
            self.assertTrue(reports[0]['continuation_pages'])
            self.assertTrue(all(p.rect == doc[0].rect for p in out))
            with self.assertRaisesRegex(ValueError, 'Jianpu.*taller'):
                paginate(doc, layouts, {0: [(None, fitz.Rect(0, 0, 200, 300))]}, [{}], out, place)


@unittest.skipUnless(shutil.which('lilypond'), 'LilyPond is required')
class SpacedInsituTest(unittest.TestCase):
    def test_dense_source_marks_no_longer_require_a_safe_cut(self):
        import insitu.render_insitu as renderer
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf, xml = make_sources(root, (70, 170), 300)
            source = root / 'interleaved.pdf'
            with fitz.open(pdf) as doc:
                # One intact object spans the old inter-staff cutting band.
                doc[0].draw_rect(fitz.Rect(200, 95, 215, 158), color=None, fill=(0, 0, 0))
                doc.save(source)
            output = root / 'out.insitu.pdf'
            renderer.build(source, xml, output)
            report = json.loads(output.with_suffix('.report.json').read_text())
            self.assertEqual(report['rendered_staff_count'], 2)
            self.assertEqual(report['incomplete_staff_count'], 0)
            self.assertTrue(all(r['source_placement'] == 'whole_objects' for r in report['rows']))
            self.assertNotIn('no_safe_source_cut', json.dumps(report))
            self.assertNotIn('detached_page', json.dumps(report))
            with fitz.open(output) as doc:
                self.assertTrue(all(p.rect == fitz.Rect(0, 0, 600, 300) for p in doc))
                intact = [d for p in doc for d in p.get_drawings()
                          if abs(d['rect'].width - 15) < .1 and abs(d['rect'].height - 63) < .1]
                self.assertTrue(intact)

    def test_cli_paginates_without_resizing_and_supports_compressed_musicxml(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf, xml = make_sources(root)
            original_xml = xml.read_bytes()
            for compressed in (False, True):
                musicxml = xml
                if compressed:
                    musicxml = root / 'compressed.mxl'
                    with zipfile.ZipFile(musicxml, 'w') as archive:
                        archive.writestr('META-INF/container.xml', '''<container
xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles><rootfile
full-path="score.musicxml" media-type="application/vnd.recordare.musicxml+xml"/>
</rootfiles></container>''')
                        archive.writestr('score.musicxml', original_xml)
                prefix = root / ('compressed' if compressed else 'out')
                result = subprocess.run([sys.executable, str(ROOT / 'jianpu.py'), str(pdf),
                                         '--mxl', str(musicxml), '--mode', 'insitu', '-o', str(prefix)],
                                        cwd=ROOT, capture_output=True, text=True,
                                        env={**os.environ, 'TMPDIR': tmp}, timeout=120)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn('placed 3/3 lines', result.stdout)
                report = json.loads(prefix.with_suffix('.insitu.report.json').read_text())
                self.assertEqual(report['staff_count'], 3)
                self.assertEqual(report['rendered_staff_count'], 3)
                self.assertEqual(report['incomplete_staff_count'], 0)
                self.assertEqual(report['layout'], 'single')
                output = prefix.with_suffix('.insitu.pdf')
                self.assertEqual(report['output_pdf_sha256'], hashlib.sha256(output.read_bytes()).hexdigest())
                with fitz.open(pdf) as source, fitz.open(output) as doc:
                    self.assertGreater(len(doc), 1)
                    self.assertTrue(all(p.rect == source[0].rect for p in doc))
                    text = ''.join(p.get_text() for p in doc)
                    for i in range(3):
                        self.assertEqual(text.count(f'OriginalRow{i + 1}'), 1)
                self.assertEqual(xml.read_bytes(), original_xml)

    def test_failed_middle_row_keeps_all_original_staffs(self):
        import insitu.render_insitu as renderer
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf, xml = make_sources(root)
            original = renderer._run_jianpu_ly
            def render(text, workdir):
                if 'sys_1_' in str(workdir):
                    raise RuntimeError('middle row unavailable')
                return original(text, workdir)
            output = root / 'out.pdf'
            with mock.patch.object(renderer, '_run_jianpu_ly', side_effect=render):
                renderer.build(pdf, xml, output)
            report = json.loads(output.with_suffix('.report.json').read_text())
            self.assertEqual(report['status'], 'incomplete')
            self.assertEqual([r['rendered'] for r in report['rows']], [True, False, True])
            with fitz.open(output) as doc:
                text = ''.join(p.get_text() for p in doc)
                for i in range(3):
                    self.assertEqual(text.count(f'OriginalRow{i + 1}'), 1)
                for row in report['rows']:
                    page = doc[row['output_page'] - 1]
                    # Staff rules are whole paths, retaining their 500pt width.
                    lines = [d for d in page.get_drawings() if abs(d['rect'].width - 500) < .1]
                    self.assertGreaterEqual(len(lines), 5)

    def test_failed_row_is_reported_as_incomplete_but_pdf_is_saved(self):
        import insitu.render_insitu as renderer
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf, xml = make_sources(root, (70,), 200)
            output = root / 'out.pdf'
            with mock.patch.object(renderer, '_run_jianpu_ly', side_effect=RuntimeError('test failure')):
                renderer.build(pdf, xml, output)
            report = json.loads(output.with_suffix('.report.json').read_text())
            self.assertTrue(output.is_file())
            self.assertEqual(report['status'], 'incomplete')
            self.assertEqual(report['rendered_staff_count'], 0)
            self.assertEqual(report['rows'][0]['voices'][0]['error'], 'test failure')

    def test_interrupted_run_does_not_leave_previous_success_report(self):
        import insitu.render_insitu as renderer
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf, xml = make_sources(root, (70,), 200)
            report = root / 'out.report.json'
            report.write_text('{"status":"rendered"}')
            with mock.patch.object(renderer, 'align_from_files', side_effect=RuntimeError('test interruption')):
                with self.assertRaisesRegex(RuntimeError, 'test interruption'):
                    renderer.build(pdf, xml, root / 'out.pdf')
            self.assertEqual(json.loads(report.read_text())['status'], 'in_progress')

    def test_stamp_reports_no_room_instead_of_clipping(self):
        import insitu.render_insitu as renderer
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf, xml = make_sources(root, (70, 110), 200)
            output = root / 'out.pdf'
            renderer.build(pdf, xml, output, expand=False)
            report = json.loads(output.with_suffix('.report.json').read_text())
            self.assertEqual(report['status'], 'incomplete')
            self.assertIn('insufficient_overlay_space', report['rows'][0]['voices'][0]['error'])
            with fitz.open(pdf) as source, fitz.open(output) as doc:
                self.assertEqual(len(doc), 1)
                self.assertEqual(doc[0].rect, source[0].rect)


if __name__ == '__main__':
    unittest.main()
