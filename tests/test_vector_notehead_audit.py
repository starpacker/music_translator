"""PDF evidence must not be confused with full transcription verification."""
import json
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import jianpu
import core.omr.vector_notehead_audit as audit
from test_vector_tempos import make_sources


class NoteheadAuditTest(unittest.TestCase):
    def test_supported_positions_agree_but_accidentals_and_time_remain_unchecked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf, xml = make_sources(root)
            before = (pdf.read_bytes(), xml.read_bytes())
            with mock.patch.object(audit, '_page_marks', return_value=([], [(100, 125, 20), (200, 122.5, 20)], [])):
                report = audit.audit_pdf_noteheads(pdf, xml)
            self.assertEqual(report['status'], 'partial_evidence')
            self.assertEqual(report['summary'], {'supported_head_positions_agree': 1})
            self.assertEqual(report['measures'][0]['musicxml_pitched_heads'], 2)
            self.assertEqual(report['note_value_summary']['head_shapes_unverified'], 2)
            self.assertIn('accidentals_not_checked', report['limitations'])
            self.assertIn('beam_counts_are_lower_bounds_not_complete_durations', report['limitations'])
            self.assertEqual((pdf.read_bytes(), xml.read_bytes()), before)

    def test_incompatible_shape_is_reported_without_changing_the_musicxml(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdf, xml = make_sources(Path(tmp))
            before = xml.read_bytes()
            glyphs = [{'x': 100., 'y': 125., 'size': 20., 'kind': 'whole', 'bbox': (100, 110, 107, 127)},
                      {'x': 200., 'y': 122.5, 'size': 20., 'kind': 'half', 'bbox': (200, 108, 207, 125)}]
            with mock.patch.object(audit, '_page_marks', return_value=([], [(100, 125, 20), (200, 122.5, 20)], [])), \
                 mock.patch.object(audit, 'verified_notehead_glyphs', return_value=glyphs):
                result = audit.audit_pdf_noteheads(pdf, xml)
            self.assertEqual(result['summary']['supported_head_positions_agree'], 1)
            self.assertEqual(result['note_value_summary']['head_shape_conflicts'], 1)
            self.assertEqual(result['measures'][0]['note_value_evidence']['head_shape_conflicts'][0]['musicxml_type'], 'half')
            self.assertEqual(xml.read_bytes(), before)

    def test_missing_or_different_heads_are_review_not_guessed_corrections(self):
        cases = [([], 'no_supported_noteheads_detected'),
                 ([(100, 125, 20)], 'head_count_differs_review'),
                 ([(100, 130, 20), (200, 122.5, 20)], 'head_positions_or_grouping_differ_review')]
        for heads, expected in cases:
            with self.subTest(status=expected), tempfile.TemporaryDirectory() as tmp:
                pdf, xml = make_sources(Path(tmp))
                before = xml.read_bytes()
                with mock.patch.object(audit, '_page_marks', return_value=([], heads, [])):
                    result = audit.audit_pdf_noteheads(pdf, xml)
                self.assertEqual(result['measures'][0]['status'], expected)
                self.assertEqual(xml.read_bytes(), before)

    def test_printed_multirest_count_is_preserved_only_when_uniquely_corroborated(self):
        for has_count in (True, False):
            with self.subTest(has_count=has_count), tempfile.TemporaryDirectory() as tmp:
                pdf, xml = make_sources(Path(tmp), rest=True, rest_count=has_count)
                result = audit.audit_pdf_noteheads(pdf, xml)
                measure = result['measures'][0]
                if has_count:
                    self.assertEqual(measure['status'], 'printed_multirest_count_detected')
                    self.assertEqual(measure['printed_rest_measures'], 3)
                else:
                    self.assertEqual(measure['status'], 'no_pitched_heads_to_compare')
                    self.assertNotIn('printed_rest_measures', measure)

    def test_mismatched_system_is_not_mapped_by_guess(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdf, xml = make_sources(Path(tmp))
            layouts, slices = audit.align_from_files(pdf, xml)
            slices[0].pdf_measures += 1
            with mock.patch.object(audit, 'align_from_files', return_value=(layouts, slices)):
                report = audit.audit_pdf_noteheads(pdf, xml)
            self.assertEqual(report['measures'][0]['status'], 'ambiguous_measure_alignment')
            self.assertNotIn('musicxml_measure_index', report['measures'][0])
            self.assertEqual(report['unmapped_musicxml_indices'], [1])

    def test_no_detected_measure_slots_is_unsupported(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdf, xml = make_sources(Path(tmp))
            layouts, slices = audit.align_from_files(pdf, xml)
            layouts[0].barlines = []
            with mock.patch.object(audit, 'align_from_files', return_value=(layouts, slices)):
                result = audit.audit_pdf_noteheads(pdf, xml)
            self.assertEqual((result['status'], result['reason']), ('unsupported', 'no_measure_slots'))

    def test_failed_audit_replaces_stale_success_report_without_stopping(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf, xml = make_sources(root)
            report = root / 'out.pdf-noteheads.json'
            report.write_text('{"status": "partial_evidence"}')
            with mock.patch.object(audit, 'audit_pdf_noteheads', side_effect=RuntimeError('failure')):
                self.assertEqual(jianpu.write_pdf_notehead_audit(pdf, xml, root, 'out'), report)
            self.assertEqual(json.loads(report.read_text())['status'], 'error')

    def test_saved_report_records_input_hashes_and_does_not_modify_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf, xml = make_sources(root, rest=True)
            before = (pdf.read_bytes(), xml.read_bytes())
            path = jianpu.write_pdf_notehead_audit(pdf, xml, root, 'out')
            data = json.loads(path.read_text())
            self.assertEqual(data['source_pdf_sha256'], hashlib.sha256(before[0]).hexdigest())
            self.assertEqual(data['source_musicxml_sha256'], hashlib.sha256(before[1]).hexdigest())
            self.assertEqual(data['measures'][0]['printed_rest_measures'], 3)
            self.assertEqual((pdf.read_bytes(), xml.read_bytes()), before)

    def test_cli_can_disable_optional_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf, xml = make_sources(root)
            with mock.patch.object(jianpu, 'write_pdf_notehead_audit') as callback, \
                 mock.patch.object(jianpu, 'musicxml_to_jianpu'):
                self.assertEqual(jianpu.main([str(pdf), '--mxl', str(xml), '--mode', 'jianpu',
                                              '--no-vector-tempos', '--no-vector-dynamics',
                                              '--no-vector-wedges', '--no-pdf-audit', '-o', str(root / 'out')]), 0)
            callback.assert_not_called()


if __name__ == '__main__':
    unittest.main()
