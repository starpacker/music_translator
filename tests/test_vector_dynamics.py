"""Source dynamics are recovered only from verified font glyphs and onsets."""
from fractions import Fraction
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock
import xml.etree.ElementTree as ET

import pymupdf
from music21 import converter, dynamics
import jianpu
import core.omr.vector_dynamics as recovery
from core.omr.vector_tempos import write_enriched_tempos
from test_vector_tempos import fake_heads, make_sources


class FakeFont:
    def __init__(self, name, glyphs):
        self.name = name
        self.glyphs = glyphs

    def has_glyph(self, code):
        return self.glyphs.get(code, 0)


class FontIdentityTest(unittest.TestCase):
    def test_only_unambiguous_matching_music_font_glyph_is_used(self):
        doc, page = mock.Mock(), mock.Mock()
        page.get_fonts.return_value = [(1, None, None, 'ABCDEF+', None, None, 0),
                                       (2, None, None, 'GHIJKL+', None, None, 0)]
        doc.extract_font.side_effect = lambda x: ('', '', '', bytes([x]))
        page.get_texttrace.return_value = [{'font': '', 'size': 20,
            'chars': [(ord('F'), 41, (100, 140), (100, 130, 116, 150))]}]
        first = FakeFont('EngraverFontSet Regular', {0xF046: 41})
        second = FakeFont('EngraverTextT Regular', {0xF046: 31})
        with mock.patch.object(recovery.pymupdf, 'Font', side_effect=[first, second]):
            self.assertEqual(recovery._dynamic_glyphs(doc, page)[0][-1], 'mf')
        second.glyphs[0xF046] = 41
        with mock.patch.object(recovery.pymupdf, 'Font', side_effect=[first, second]):
            self.assertEqual(recovery._dynamic_glyphs(doc, page), [])

    def test_ambiguous_or_distant_onset_is_rejected(self):
        events = [((100,), 0, False), ((106,), 1, False)]
        self.assertIsNone(recovery._closest_event(103, events, 5))
        self.assertIsNone(recovery._closest_event(120, events, 5))
        self.assertEqual(recovery._closest_event(100, events, 5), Fraction(0))


class VectorDynamicRecoveryTest(unittest.TestCase):
    def test_nonzero_onset_and_existing_tempo_survive_in_derived_xml(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf, mxl = make_sources(root)
            intermediate = root / 'tempo.musicxml'
            write_enriched_tempos(mxl, intermediate, [{'measure_index': 1, 'bpm': 120}])
            original = mxl.read_bytes()
            # F is centred on the second note at x=200.
            with mock.patch.object(recovery, '_page_marks', return_value=([], [(100, 125, 20), (200, 122.5, 20)], [])), \
                 mock.patch.object(recovery, '_dynamic_glyphs', return_value=[(195.5, 140, 16, 20, 'mf')]):
                output = jianpu.recover_pdf_dynamics(pdf, mxl, intermediate, root, 'out')
            self.assertEqual(mxl.read_bytes(), original)
            self.assertNotEqual(output, intermediate)
            source_notes = [ET.tostring(e) for e in ET.parse(mxl).findall('./part/measure/note')]
            output_notes = [ET.tostring(e) for e in ET.parse(output).findall('./part/measure/note')]
            self.assertEqual(source_notes, output_notes)
            self.assertEqual(len(ET.parse(output).findall('.//metronome')), 1)
            score = converter.parse(str(output))
            measure = score.parts[0].measure(1)
            marks = list(measure.recurse().getElementsByClass(dynamics.Dynamic))
            self.assertEqual([(d.value, float(d.getOffsetInHierarchy(measure))) for d in marks], [('mf', 2.0)])
            report = json.loads((root / 'out.pdf-dynamics.json').read_text())
            self.assertEqual(report['inserted'][0]['offset_quarters'], '2')

    def test_existing_dynamics_and_unrepresentable_offsets_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, mxl = make_sources(root)
            tree = ET.parse(mxl)
            measure = tree.find('./part/measure')
            direction = ET.Element('direction')
            dtype = ET.SubElement(direction, 'direction-type')
            ET.SubElement(ET.SubElement(dtype, 'dynamics'), 'p')
            measure.insert(1, direction)
            tree.write(mxl)
            out = root / 'derived.musicxml'
            candidate = {'measure_index': 1, 'offset_quarters': '0', 'value': 'f'}
            inserted, skipped = recovery.write_enriched_dynamics(mxl, out, [candidate])
            self.assertEqual(inserted, [])
            self.assertEqual(skipped[0]['reason'], 'existing_musicxml_dynamics')
            self.assertFalse(out.exists())
            measure.remove(direction)
            measure.find('./attributes/divisions').text = '1'
            tree.write(mxl)
            candidate['offset_quarters'] = '1/2'
            inserted, skipped = recovery.write_enriched_dynamics(mxl, out, [candidate])
            self.assertEqual(inserted, [])
            self.assertEqual(skipped[0]['reason'], 'unrepresentable_offset')

    def test_two_verified_changes_in_one_measure_are_both_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, mxl = make_sources(root)
            candidates = [{'measure_index': 1, 'offset_quarters': '0', 'value': 'p'},
                          {'measure_index': 1, 'offset_quarters': '2', 'value': 'f'}]
            out = root / 'derived.musicxml'
            inserted, skipped = recovery.write_enriched_dynamics(mxl, out, candidates)
            self.assertEqual(len(inserted), 2)
            self.assertEqual(skipped, [])
            measure = converter.parse(str(out)).parts[0].measure(1)
            marks = [(d.value, float(d.getOffsetInHierarchy(measure)))
                     for d in measure.recurse().getElementsByClass(dynamics.Dynamic)]
            self.assertEqual(marks, [('p', 0.0), ('f', 2.0)])

    def test_midmeasure_divisions_change_is_not_guessed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, mxl = make_sources(root)
            tree = ET.parse(mxl)
            measure = tree.find('./part/measure')
            children = list(measure)
            first_note = next(i for i, child in enumerate(children) if child.tag == 'note')
            attrs = ET.Element('attributes')
            ET.SubElement(attrs, 'divisions').text = '3'
            measure.insert(first_note + 1, attrs)
            tree.write(mxl)
            inserted, skipped = recovery.write_enriched_dynamics(
                mxl, root / 'out.musicxml', [{'measure_index': 1, 'offset_quarters': '0', 'value': 'f'}])
            self.assertEqual(inserted, [])
            self.assertEqual(skipped[0]['reason'], 'unrepresentable_offset')

    def test_distant_mark_and_conflicting_marks_are_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf, mxl = make_sources(root)
            with mock.patch.object(recovery, '_page_marks', return_value=([], [(100, 125, 20), (200, 122.5, 20)], [])), \
                 mock.patch.object(recovery, '_dynamic_glyphs', return_value=[
                     (145, 140, 7, 20, 'p'), (95.5, 140, 16, 20, 'mf'),
                     (95.5, 140, 16, 20, 'mp')]):
                candidates, report = recovery.discover_pdf_dynamics(pdf, mxl)
            self.assertEqual(candidates, [])
            reasons = [item['reason'] for item in report]
            self.assertEqual(reasons.count('no_unique_note_onset'), 1)
            self.assertEqual(reasons.count('multiple_dynamics_at_onset'), 2)

    def test_disable_option_keeps_original_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf, mxl = make_sources(root)
            with mock.patch.object(jianpu, 'recover_pdf_dynamics') as recover, \
                 mock.patch.object(jianpu, 'musicxml_to_jianpu') as render:
                result = jianpu.main([str(pdf), '--mxl', str(mxl), '--mode', 'jianpu',
                                      '--no-vector-tempos', '--no-vector-dynamics', '-o', str(root / 'out')])
            self.assertEqual(result, 0)
            recover.assert_not_called()
            self.assertEqual(render.call_args.args[0], mxl)

    def test_cli_chains_tempo_and_dynamic_without_changing_insitu_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf, mxl = make_sources(root)
            original = mxl.read_bytes()
            heads = [(100, 125, 20), (200, 122.5, 20)]
            with fake_heads(heads), \
                 mock.patch.object(recovery, '_page_marks', return_value=([], heads, [])), \
                 mock.patch.object(recovery, '_dynamic_glyphs', return_value=[(195.5, 140, 16, 20, 'mf')]), \
                 mock.patch.object(jianpu, 'musicxml_to_jianpu') as render, \
                 mock.patch.object(jianpu, 'insitu_stamp') as insitu:
                code = jianpu.main([str(pdf), '--mxl', str(mxl), '--mode', 'both', '-o', str(root / 'out')])
            self.assertEqual(code, 0)
            derived = render.call_args.args[0]
            self.assertNotEqual(derived, mxl)
            self.assertEqual(insitu.call_args.args[1], mxl)
            self.assertEqual(mxl.read_bytes(), original)
            tree = ET.parse(derived)
            self.assertEqual(len(tree.findall('.//metronome')), 1)
            self.assertEqual(len(tree.findall('.//dynamics/mf')), 1)

    def test_failed_or_empty_recovery_keeps_input_and_refreshes_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf, mxl = make_sources(root)
            heads = [(100, 125, 20), (200, 122.5, 20)]
            with mock.patch.object(recovery, '_page_marks', return_value=([], heads, [])), \
                 mock.patch.object(recovery, '_dynamic_glyphs', return_value=[(195.5, 140, 16, 20, 'mf')]):
                first = jianpu.recover_pdf_dynamics(pdf, mxl, mxl, root, 'out')
            self.assertNotEqual(first, mxl)
            with mock.patch.object(recovery, '_dynamic_glyphs', return_value=[]):
                second = jianpu.recover_pdf_dynamics(pdf, mxl, mxl, root, 'out')
            self.assertEqual(second, mxl)
            report = json.loads((root / 'out.pdf-dynamics.json').read_text())
            self.assertEqual(report['inserted'], [])
            with mock.patch.object(recovery, 'discover_pdf_dynamics', side_effect=RuntimeError('failure')):
                self.assertEqual(jianpu.recover_pdf_dynamics(pdf, mxl, mxl, root, 'out'), mxl)
            report = json.loads((root / 'out.pdf-dynamics.json').read_text())
            self.assertEqual(report['error'], 'failure')
            self.assertEqual(report['inserted'], [])

    def test_generated_name_collision_never_overwrites_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf, xml = make_sources(root)
            source = root / 'out.pdf-dynamics.musicxml'
            source.write_bytes(xml.read_bytes())
            before = source.read_bytes()
            heads = [(100, 125, 20), (200, 122.5, 20)]
            with mock.patch.object(recovery, '_page_marks', return_value=([], heads, [])), \
                 mock.patch.object(recovery, '_dynamic_glyphs', return_value=[(195.5, 140, 16, 20, 'mf')]):
                derived = jianpu.recover_pdf_dynamics(pdf, source, source, root, 'out')
            self.assertNotEqual(derived, source)
            self.assertEqual(source.read_bytes(), before)

    @unittest.skipUnless(shutil.which('lilypond'), 'LilyPond is required')
    def test_recovered_dynamic_is_visible_in_real_pdf(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf, mxl = make_sources(root)
            with mock.patch.object(recovery, '_page_marks', return_value=([], [(100, 125, 20), (200, 122.5, 20)], [])), \
                 mock.patch.object(recovery, '_dynamic_glyphs', return_value=[(195.5, 140, 16, 20, 'mf')]):
                derived = jianpu.recover_pdf_dynamics(pdf, mxl, mxl, root, 'out')
            output = root / 'jianpu.pdf'
            jianpu.musicxml_to_jianpu(derived, output, 'test')
            with pymupdf.open(output) as doc:
                self.assertIn('mf', ''.join(page.get_text() for page in doc))


if __name__ == '__main__':
    unittest.main()
