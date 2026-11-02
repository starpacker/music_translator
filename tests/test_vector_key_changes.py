"""Move a delayed key only when source geometry uniquely corroborates it."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest import mock
import xml.etree.ElementTree as ET

import pymupdf
import jianpu
import core.omr.vector_key_changes as recovery
from core.notation.jianpu import (_get_score_key_tonic, extract_jianpu_measures,
                                  jianpu_note_token)
from core.notation.musicxml_wedges import parse_musicxml_compatible
from insitu.align import SystemSlice
from insitu.pdf_structure import Staff, StaffLayout


def make_case(root, direction):
    pdf, xml = root / 'source.pdf', root / 'source.musicxml'
    doc = pymupdf.open()
    doc.new_page(width=300, height=300)
    doc.save(pdf)
    doc.close()
    old, new = (0, 1) if direction == 'sharp' else (1, 0)
    second_pitch = '<step>B</step><octave>4</octave>' if direction == 'sharp' else '<step>F</step><alter>1</alter><octave>5</octave>'
    third_pitch = '<step>E</step><octave>5</octave>' if direction == 'sharp' else '<step>C</step><octave>5</octave>'
    xml.write_text(f'''<score-partwise version="4.0"><part-list><score-part id="P1"><part-name>Flute</part-name></score-part></part-list>
<part id="P1"><measure number="1"><attributes><divisions>1</divisions><key><fifths>{old}</fifths></key>
<time><beats>4</beats><beat-type>4</beat-type></time><clef><sign>G</sign><line>2</line></clef></attributes>
<note><pitch><step>E</step><octave>5</octave></pitch><duration>4</duration><type>whole</type></note></measure>
<measure number="2"><note><pitch>{second_pitch}</pitch><duration>4</duration><type>whole</type></note></measure>
<measure number="3"><print new-system="yes"/><attributes><key><fifths>{new}</fifths></key></attributes>
<note><pitch>{third_pitch}</pitch><duration>4</duration><type>whole</type></note></measure>
<measure number="4"><note><pitch><step>D</step><octave>5</octave></pitch><duration>4</duration><type>whole</type></note></measure>
</part></score-partwise>''')
    first = StaffLayout(Staff(1, 100, 120, 40, 200, 5), [120, 200])
    second = StaffLayout(Staff(1, 200, 220, 40, 200, 5), [120, 200])
    slices = [SystemSlice(0, 0, 2, 2), SystemSlice(1, 2, 4, 2)]
    if direction == 'sharp':
        marks = [('#', 125, 100, 20), ('#', 64, 200, 20)]
        heads = [(80, 102.5, 20), (145, 110, 20), (80, 202.5, 20), (145, 205, 20)]
    else:
        marks = [('#', 64, 100, 20), ('n', 125, 100, 20), ('#', 137, 100, 20)]
        heads = [(80, 102.5, 20), (145, 100, 20), (80, 207.5, 20), (145, 205, 20)]
    return pdf, xml, [first, second], slices, marks, heads


def discover(pdf, xml, layouts, slices, marks, heads):
    with mock.patch.object(recovery, 'align_from_files', return_value=(layouts, slices)), \
         mock.patch.object(recovery, '_accidentals', return_value=marks), \
         mock.patch.object(recovery, '_page_marks', return_value=([], heads, [])):
        return recovery.discover_pdf_key_changes(pdf, xml)


def tokens(path, measure):
    score = parse_musicxml_compatible(path)
    tonic, _ = _get_score_key_tonic(score)
    measures, _ = extract_jianpu_measures(score, tonic, follow_key_changes=True)
    return [jianpu_note_token(event) for event in measures[measure - 1]]


class KeyRecoveryTest(unittest.TestCase):
    def test_music_font_identity_must_be_unique(self):
        class Font:
            def __init__(self, name, matches):
                self.name, self.matches = name, matches
            def has_glyph(self, code):
                return self.matches.get(code, 0)
        doc, page = mock.Mock(), mock.Mock()
        page.get_fonts.return_value = [(1, None, None, 'ABCDEF+'), (2, None, None, 'GHIJKL+')]
        doc.extract_font.side_effect = lambda x: ('', '', '', bytes([x]))
        page.get_texttrace.return_value = [{'font': '', 'size': 20,
            'chars': [(35, 6, (125, 100), (125, 90, 130, 110))]}]
        music = Font('EngraverFontSet Regular', {0xF023: 6})
        other = Font('EngraverTextT Regular', {0xF023: 7})
        with mock.patch.object(recovery.pymupdf, 'Font', side_effect=[music, other]):
            self.assertEqual(recovery._accidentals(doc, page), [('#', 125., 100., 20.)])
        other.matches[0xF023] = 6
        with mock.patch.object(recovery.pymupdf, 'Font', side_effect=[music, other]):
            self.assertEqual(recovery._accidentals(doc, page), [])

    def test_key_is_moved_on_a_copy_and_changes_only_jianpu_reference(self):
        for direction, before, after in [('sharp', '7', '3'), ('natural', '7', "#4'")]:
            with self.subTest(direction=direction), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                pdf, xml, layouts, slices, marks, heads = make_case(root, direction)
                original = xml.read_bytes()
                candidates, decisions = discover(pdf, xml, layouts, slices, marks, heads)
                self.assertEqual(len(candidates), 1, decisions)
                self.assertEqual(candidates[0]['source_measure_index'], 2)
                self.assertEqual(candidates[0]['later_musicxml_measure_index'], 3)
                out = root / 'derived.musicxml'
                inserted, skipped = recovery.write_enriched_keys(xml, out, candidates)
                self.assertEqual(len(inserted), 1)
                self.assertFalse(skipped)
                self.assertEqual(xml.read_bytes(), original)
                self.assertEqual([ET.tostring(n) for n in ET.parse(xml).findall('.//note')],
                                 [ET.tostring(n) for n in ET.parse(out).findall('.//note')])
                self.assertTrue(tokens(xml, 2)[0].startswith(before), tokens(xml, 2))
                self.assertTrue(tokens(out, 2)[0].startswith(after), tokens(out, 2))
                self.assertEqual(len(ET.parse(out).findall('./part/measure[2]/attributes/key')), 1)
                self.assertFalse(ET.parse(out).findall('./part/measure[3]/attributes/key'))

    def test_missing_or_conflicting_evidence_never_moves_a_key(self):
        for direction in ('sharp', 'natural'):
            with self.subTest(direction=direction), tempfile.TemporaryDirectory() as tmp:
                pdf, xml, layouts, slices, marks, heads = make_case(Path(tmp), direction)
                for altered_marks in ([], None, marks[:-1]):
                    candidates, _ = discover(pdf, xml, layouts, slices, altered_marks, heads)
                    self.assertFalse(candidates)
                slices[0].pdf_measures += 1
                candidates, _ = discover(pdf, xml, layouts, slices, marks, heads)
                self.assertFalse(candidates)

    def test_later_key_or_existing_destination_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf, xml, layouts, slices, marks, heads = make_case(root, 'sharp')
            candidates, _ = discover(pdf, xml, layouts, slices, marks, heads)
            tree = ET.parse(xml)
            dest = tree.find('./part/measure[2]')
            attrs = ET.Element('attributes')
            ET.SubElement(ET.SubElement(attrs, 'key'), 'fifths').text = '-1'
            dest.insert(0, attrs)
            tree.write(xml)
            before = xml.read_bytes()
            inserted, skipped = recovery.write_enriched_keys(xml, root / 'out.xml', candidates)
            self.assertFalse(inserted)
            self.assertEqual(skipped[0]['reason'], 'unsafe_or_changed_musicxml')
            self.assertEqual(xml.read_bytes(), before)

    def test_cli_can_disable_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf, xml, *_ = make_case(root, 'sharp')
            with mock.patch.object(jianpu, 'recover_pdf_keys') as callback, \
                 mock.patch.object(jianpu, 'musicxml_to_jianpu'):
                self.assertEqual(jianpu.main([str(pdf), '--mxl', str(xml), '--mode', 'jianpu',
                                              '--no-vector-tempos', '--no-vector-dynamics', '--no-vector-wedges',
                                              '--no-pdf-audit', '--no-vector-keys', '-o', str(root / 'out')]), 0)
            callback.assert_not_called()

    def test_both_outputs_use_the_same_key_corrected_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf, xml, *_ = make_case(root, 'sharp')
            derived = root / 'derived.musicxml'
            derived.write_bytes(xml.read_bytes())
            with mock.patch.object(jianpu, 'recover_pdf_keys', return_value=derived), \
                 mock.patch.object(jianpu, 'write_pdf_notehead_audit'), \
                 mock.patch.object(jianpu, 'musicxml_to_jianpu') as standalone, \
                 mock.patch.object(jianpu, 'insitu_stamp') as insitu:
                self.assertEqual(jianpu.main([str(pdf), '--mxl', str(xml), '--mode', 'both',
                                              '--no-vector-tempos', '--no-vector-dynamics', '--no-vector-wedges',
                                              '-o', str(root / 'out')]), 0)
            self.assertEqual(standalone.call_args.args[0], derived)
            self.assertEqual(insitu.call_args.args[1], derived)

    def test_failed_inference_keeps_input_and_replaces_stale_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf, xml, *_ = make_case(root, 'sharp')
            report = root / 'out.pdf-keys.json'
            report.write_text('{"status":"inferred"}')
            original = xml.read_bytes()
            with mock.patch.object(recovery, 'discover_pdf_key_changes', side_effect=RuntimeError('failure')):
                self.assertEqual(jianpu.recover_pdf_keys(pdf, xml, root, 'out'), xml)
            self.assertEqual(json.loads(report.read_text())['status'], 'error')
            self.assertEqual(xml.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
