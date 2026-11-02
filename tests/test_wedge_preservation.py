"""MusicXML hairpin boundaries are preserved at exact musical positions."""
from fractions import Fraction
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock
import zipfile

import pymupdf
from music21 import converter
from core.notation.jianpu import build_jianpu_ly_text
from core.notation.musicxml_wedges import read_wedges


XML = '''<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="4.0"><part-list><score-part id="P1"><part-name>Flute</part-name></score-part></part-list>
<part id="P1"><measure number="1"><attributes><divisions>2</divisions>
<key><fifths>0</fifths><mode>major</mode></key><time><beats>4</beats><beat-type>4</beat-type></time>
<clef><sign>G</sign><line>2</line></clef></attributes>
<direction><direction-type><dynamics><p/></dynamics></direction-type></direction>
<direction><direction-type><wedge type="crescendo" number="1"/></direction-type></direction>
<note><pitch><step>C</step><octave>4</octave></pitch><duration>2</duration><type>quarter</type></note>
<note><pitch><step>D</step><octave>4</octave></pitch><duration>2</duration><type>quarter</type></note>
<direction><direction-type><wedge type="stop" number="1"/></direction-type></direction>
<direction><direction-type><wedge type="diminuendo" number="2"/></direction-type></direction>
<note><pitch><step>E</step><octave>4</octave></pitch><duration>2</duration><type>quarter</type></note>
<note><pitch><step>F</step><octave>4</octave></pitch><duration>2</duration><type>quarter</type></note>
</measure><measure number="2"><note><pitch><step>G</step><octave>4</octave></pitch>
<duration>8</duration><type>whole</type></note>
<direction><direction-type><wedge type="stop" number="2"/></direction-type></direction>
</measure></part></score-partwise>'''


class WedgePreservationTest(unittest.TestCase):
    def test_exact_boundaries_and_compact_tokens(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'score.musicxml'
            source.write_text(XML)
            pairs, issues = read_wedges(source)
            self.assertEqual(issues, [])
            self.assertEqual(pairs[0], [((0, Fraction(0)), (0, Fraction(2)), 'crescendo'),
                                        ((0, Fraction(2)), (1, Fraction(4)), 'diminuendo')])
            text = build_jianpu_ly_text(converter.parse(str(source)), 'test', _source_wedges=pairs)
            self.assertIn('WedgeCrescendo 1 \\p ', text)
            self.assertIn('2 WedgeStop WedgeDiminuendo 3', text)
            self.assertIn('5 - - - WedgeStop', text)

    def test_midnote_endpoint_is_reported_without_partial_wedge(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'score.musicxml'
            # Stop halfway through D4: this position is not a rendered event boundary.
            source.write_text(XML.replace('<wedge type="stop" number="1"/></direction-type>',
                                          '<wedge type="stop" number="1"/></direction-type><offset>-1</offset>', 1))
            pairs, issues = read_wedges(source)
            self.assertEqual(issues, [])
            with mock.patch('core.notation.jianpu.log_message') as log:
                text = build_jianpu_ly_text(converter.parse(str(source)), 'test', _source_wedges=pairs)
            self.assertNotIn('WedgeCrescendo', text)
            self.assertEqual(text.count('WedgeStop'), 1)
            self.assertTrue(any('端点不在简谱事件边界' in str(call) for call in log.call_args_list))

    def test_overlapping_wedges_are_reported_not_rewritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'score.musicxml'
            modified = XML.replace('<wedge type="stop" number="1"/></direction-type>',
                                   '<wedge type="stop" number="1"/></direction-type><offset>2</offset>', 1)
            source.write_text(modified)
            pairs, issues = read_wedges(source)
            self.assertEqual(pairs[0], [])
            self.assertTrue(any('重叠' in issue[2] for issue in issues))

    def test_compressed_musicxml_uses_container_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            mxl = Path(tmp) / 'score.mxl'
            with zipfile.ZipFile(mxl, 'w') as archive:
                archive.writestr('META-INF/container.xml', '''<?xml version="1.0"?>
<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles>
<rootfile full-path="nested/score.musicxml" media-type="application/vnd.recordare.musicxml+xml"/>
</rootfiles></container>''')
                archive.writestr('nested/score.musicxml', XML)
            pairs, issues = read_wedges(mxl)
            self.assertEqual(issues, [])
            self.assertEqual(len(pairs[0]), 2)

    def test_valid_continuation_does_not_stop_rendering(self):
        from core.render.renderer import generate_jianpu_pdf_from_mxl
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'score.musicxml'
            xml = XML.replace('<note><pitch><step>D</step>',
                              '<direction><direction-type><wedge type="continue" number="1"/>'
                              '</direction-type></direction><note><pitch><step>D</step>', 1)
            source.write_text(xml)
            before = source.read_bytes()
            with mock.patch('core.render.renderer.render_score_to_jianpu_pdf', return_value=(True, [], [])) as render:
                self.assertTrue(generate_jianpu_pdf_from_mxl(source, root / 'out.pdf', root))
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(len(render.call_args.kwargs['source_wedges'][0]), 2)

    @unittest.skipUnless(shutil.which('lilypond'), 'LilyPond is required')
    def test_real_pdf_renders_hairpins_and_dynamics(self):
        from core.render.renderer import generate_jianpu_pdf_from_mxl
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, output, workspace = root / 'score.musicxml', root / 'out.pdf', root / 'workspace'
            source.write_text(XML)
            self.assertTrue(generate_jianpu_pdf_from_mxl(source, output, root, editor_workspace_dir=workspace))
            text = next(workspace.glob('*.jianpu.txt')).read_text()
            self.assertIn('WedgeCrescendo', text)
            self.assertIn('WedgeDiminuendo', text)
            with pymupdf.open(output) as doc:
                page = doc[0]
                self.assertIn('p', page.get_text())
                diagonals = []
                for drawing in page.get_drawings():
                    for item in drawing['items']:
                        if item[0] == 'l':
                            a, b = item[1], item[2]
                            if abs(a.x - b.x) > 5 and .2 < abs(a.y - b.y) < 6:
                                diagonals.append((a, b))
                self.assertGreaterEqual(len(diagonals), 4)


if __name__ == '__main__':
    unittest.main()
