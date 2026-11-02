"""Conservative recovery of source PDF metronome marks."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import xml.etree.ElementTree as ET
import zipfile

import pymupdf
from music21 import clef, converter, key, meter, note, stream, tempo
import jianpu
import core.omr.vector_tempos as recovery
from scripts.audit_jianpu_events import audit


def make_sources(root: Path, *, marker_x=100, rest=False, existing=False, rest_count=True):
    pdf, mxl = root / 'source.pdf', root / 'source.musicxml'
    doc = pymupdf.open()
    page = doc.new_page(width=600, height=300)
    for y in (100, 105, 110, 115, 120):
        page.draw_line((40, y), (540, y), width=.5)
    page.draw_line((540, 100), (540, 120), width=.5)
    page.insert_text((marker_x, 80), 'q = 120', fontsize=10)
    if rest:
        page.draw_line((100, 110), (150, 110), width=2.4)
        if rest_count:
            page.insert_text((120, 94), '3', fontsize=10)
    doc.save(pdf)
    doc.close()
    score, part, measure = stream.Score(), stream.Part(), stream.Measure(number=1)
    measure.insert(0, clef.TrebleClef())
    measure.insert(0, key.Key('C'))
    measure.insert(0, meter.TimeSignature('4/4'))
    if existing:
        measure.insert(0, tempo.MetronomeMark(number=90))
    if rest:
        event = note.Rest(quarterLength=4)
        event.fullMeasure = True
        measure.insert(0, event)
    else:
        measure.insert(0, note.Note('C4', quarterLength=2))
        measure.insert(2, note.Note('D4', quarterLength=2))
    part.append(measure)
    score.insert(0, part)
    score.write('musicxml', fp=mxl)
    return pdf, mxl


def fake_heads(heads):
    original = recovery._page_marks
    def read(page):
        tempos, _, numbers = original(page)
        return tempos, heads, numbers
    return mock.patch.object(recovery, '_page_marks', side_effect=read)


class VectorTempoRecoveryTest(unittest.TestCase):
    def test_verified_first_onset_is_added_to_a_copy_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf, mxl = make_sources(root)
            original = mxl.read_bytes()
            with fake_heads([(100, 125, 20), (200, 122.5, 20)]):
                enriched = jianpu.recover_pdf_tempos(pdf, mxl, root, 'out')
            self.assertNotEqual(enriched, mxl)
            self.assertEqual(mxl.read_bytes(), original)
            source_notes = [ET.tostring(e) for e in ET.parse(mxl).findall('./part/measure/note')]
            result_notes = [ET.tostring(e) for e in ET.parse(enriched).findall('./part/measure/note')]
            self.assertEqual(source_notes, result_notes)
            tempos = list(converter.parse(str(enriched)).recurse().getElementsByClass(tempo.MetronomeMark))
            self.assertEqual([t.number for t in tempos], [120])
            self.assertEqual(audit(enriched)['mismatch_count'], 0)
            report = json.loads((root / 'out.pdf-tempos.json').read_text())
            self.assertEqual(report['inserted'][0]['evidence'], 'notehead_sequence')

    def test_pitched_mismatch_or_later_onset_is_not_guessed(self):
        for marker, heads, expected_reason in (
                (100, [(100, 130, 20), (200, 122.5, 20)], 'no_verified_onset'),
                (200, [(100, 125, 20), (200, 122.5, 20)], 'not_at_first_onset')):
            with self.subTest(marker=marker), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                pdf, mxl = make_sources(root, marker_x=marker)
                with fake_heads(heads):
                    candidates, report = recovery.discover_pdf_tempos(pdf, mxl)
                self.assertEqual(candidates, [])
                self.assertEqual(report[0]['reason'], expected_reason)

    def test_full_measure_rest_requires_stroke_and_printed_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdf, mxl = make_sources(Path(tmp), marker_x=96, rest=True)
            candidates, report = recovery.discover_pdf_tempos(pdf, mxl)
            self.assertEqual(len(candidates), 1, report)
            self.assertEqual(candidates[0]['evidence'], 'multi_measure_rest')

        with tempfile.TemporaryDirectory() as tmp:
            pdf, mxl = make_sources(Path(tmp), marker_x=96, rest=True, rest_count=False)
            candidates, report = recovery.discover_pdf_tempos(pdf, mxl)
            self.assertEqual(candidates, [])
            self.assertEqual(report[0]['reason'], 'no_verified_onset')

    def test_first_note_after_leading_rest_is_not_measure_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf, mxl = make_sources(root)
            score = converter.parse(str(mxl))
            measure = score.parts[0].measure(1)
            for element in list(measure.recurse().notesAndRests):
                measure.remove(element, recurse=True)
            measure.insert(0, note.Rest(quarterLength=1))
            measure.insert(1, note.Note('C4', quarterLength=3))
            score.write('musicxml', fp=mxl)
            with fake_heads([(100, 125, 20)]):
                candidates, report = recovery.discover_pdf_tempos(pdf, mxl)
            self.assertEqual(candidates, [])
            self.assertEqual(report[0]['reason'], 'no_verified_onset')

    def test_multiple_rest_slots_require_a_verified_pitched_neighbour(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf, mxl = make_sources(root, marker_x=96, rest=True)
            with pymupdf.open(pdf) as doc:
                doc[0].draw_line((290, 100), (290, 120), width=.5)
                changed = root / 'two-bars.pdf'
                doc.save(changed)
            score = converter.parse(str(mxl))
            next_measure = stream.Measure(number=2)
            next_measure.insert(0, note.Note('C4', quarterLength=4))
            score.parts[0].append(next_measure)
            score.write('musicxml', fp=mxl)
            candidates, report = recovery.discover_pdf_tempos(changed, mxl)
            self.assertEqual(candidates, [])
            self.assertEqual(report[0]['reason'], 'no_verified_onset')

    def test_disable_option_uses_original_musicxml(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf, mxl = make_sources(root)
            with mock.patch.object(jianpu, 'recover_pdf_tempos') as recover, \
                 mock.patch.object(jianpu, 'musicxml_to_jianpu') as render:
                result = jianpu.main([str(pdf), '--mxl', str(mxl), '--mode', 'jianpu',
                                      '--no-vector-tempos', '-o', str(root / 'out')])
            self.assertEqual(result, 0)
            recover.assert_not_called()
            self.assertEqual(render.call_args.args[0], mxl)

    def test_multiple_parts_do_not_receive_an_assumed_tempo(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf, mxl = make_sources(root)
            score = converter.parse(str(mxl))
            other, measure = stream.Part(), stream.Measure(number=1)
            measure.insert(0, note.Note('E4', quarterLength=4))
            other.append(measure)
            score.insert(0, other)
            score.write('musicxml', fp=mxl)
            candidates, report = recovery.discover_pdf_tempos(pdf, mxl)
            self.assertEqual(candidates, [])
            self.assertEqual(report[0]['reason'], 'multiple_parts')

    def test_multiple_candidates_in_same_measure_are_not_guessed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf, mxl = make_sources(root)
            with pymupdf.open(pdf) as doc:
                doc[0].insert_text((101, 76), 'q = 90', fontsize=10)
                replacement = root / 'two-marks.pdf'
                doc.save(replacement)
            with fake_heads([(100, 125, 20), (200, 122.5, 20)]):
                candidates, report = recovery.discover_pdf_tempos(replacement, mxl)
            self.assertEqual(candidates, [])
            self.assertEqual([item['reason'] for item in report],
                             ['multiple_tempos_in_measure', 'multiple_tempos_in_measure'])

    def test_existing_musicxml_tempo_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf, mxl = make_sources(root, existing=True)
            before = hashlib.sha256(mxl.read_bytes()).digest()
            with fake_heads([(100, 125, 20), (200, 122.5, 20)]):
                result = jianpu.recover_pdf_tempos(pdf, mxl, root, 'out')
            self.assertEqual(result, mxl)
            self.assertEqual(hashlib.sha256(mxl.read_bytes()).digest(), before)
            report = json.loads((root / 'out.pdf-tempos.json').read_text())
            self.assertEqual(report['skipped'][0]['reason'], 'existing_musicxml_tempo')

    def test_no_candidates_do_not_reuse_a_stale_derived_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf, mxl = make_sources(root)
            with fake_heads([(100, 125, 20), (200, 122.5, 20)]):
                first = jianpu.recover_pdf_tempos(pdf, mxl, root, 'out')
            self.assertNotEqual(first, mxl)
            with mock.patch.object(recovery, '_page_marks', return_value=([], [], [])):
                second = jianpu.recover_pdf_tempos(pdf, mxl, root, 'out')
            self.assertEqual(second, mxl)
            report = json.loads((root / 'out.pdf-tempos.json').read_text())
            self.assertEqual(report['inserted'], [])
            self.assertEqual(report['decisions'], [])

    def test_compressed_input_and_cli_route_derived_xml_only_to_jianpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf, xml = make_sources(root)
            mxl = root / 'source.mxl'
            with zipfile.ZipFile(mxl, 'w') as archive:
                archive.writestr('META-INF/container.xml', '''<?xml version="1.0"?>
<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles>
<rootfile full-path="score.musicxml" media-type="application/vnd.recordare.musicxml+xml"/>
</rootfiles></container>''')
                archive.writestr('score.musicxml', xml.read_bytes())
            original = mxl.read_bytes()
            with fake_heads([(100, 125, 20), (200, 122.5, 20)]), \
                 mock.patch.object(jianpu, 'musicxml_to_jianpu') as to_jianpu, \
                 mock.patch.object(jianpu, 'insitu_stamp') as to_insitu:
                result = jianpu.main([str(pdf), '--mxl', str(mxl), '--mode', 'both', '-o', str(root / 'out')])
            self.assertEqual(result, 0)
            self.assertNotEqual(to_jianpu.call_args.args[0], mxl)
            self.assertEqual(to_insitu.call_args.args[1], mxl)
            self.assertEqual(mxl.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
