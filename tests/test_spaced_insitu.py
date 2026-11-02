"""End-to-end check that the expanded layout reserves room for each line."""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
import pymupdf

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which('lilypond'), 'LilyPond is required')
class SpacedInsituTest(unittest.TestCase):
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
            result = subprocess.run([sys.executable, str(ROOT / 'jianpu.py'), str(pdf), '--mxl', str(xml),
                                     '--mode', 'insitu', '-o', str(root / 'out')], cwd=ROOT,
                                    capture_output=True, text=True, env={**os.environ, 'TMPDIR': tmp}, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('placed 2/2 lines', result.stdout)
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
