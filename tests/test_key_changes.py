"""Measure-boundary key changes retain written Jianpu pitches and markers."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import pymupdf
from music21 import converter, key, meter, note, stream
from core.notation.jianpu import _get_score_key_tonic, build_jianpu_ly_text, extract_jianpu_measures
from scripts.audit_jianpu_events import audit


def make_score():
    score, part = stream.Score(), stream.Part()
    for n, signature, pitch in ((1, key.Key('C'), 'C4'), (2, key.Key('G'), 'G4'),
                                (3, key.KeySignature(1), 'A4'), (4, key.Key('C'), 'C4')):
        measure = stream.Measure(number=n)
        if n == 1:
            measure.insert(0, meter.TimeSignature('4/4'))
        measure.insert(0, signature)
        measure.insert(0, note.Note(pitch, quarterLength=4))
        part.append(measure)
    score.insert(0, part)
    return score


class KeyChangeTest(unittest.TestCase):
    def test_initial_signature_is_not_replaced_by_later_explicit_key(self):
        score = make_score()
        first = score.parts[0].measure(1)
        first.remove(first.recurse().getElementsByClass(key.Key)[0])
        first.insert(0, key.KeySignature(0))
        tonic, _ = _get_score_key_tonic(score)
        self.assertEqual(tonic, 0)

    def test_key_markers_and_degrees_follow_changes(self):
        score = make_score()
        measures, _ = extract_jianpu_measures(score, 0, follow_key_changes=True)
        self.assertEqual(measures.key_changes, {1: (7, '1=G'), 3: (0, '1=C')})
        self.assertEqual([m[0].symbol for m in measures], ['1', '1', '2', '1'])
        text = build_jianpu_ly_text(score, 'test')
        self.assertIn('1=G 1 - - -', text)
        self.assertIn('1=C 1 - - -', text)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'score.musicxml'
            score.write('musicxml', fp=path)
            self.assertEqual(audit(path)['mismatch_count'], 0)

    def test_midmeasure_change_is_deferred_and_reported(self):
        score = make_score()
        first = score.parts[0].measure(1)
        first.insert(2, key.Key('G'))
        with mock.patch('core.notation.jianpu.extract.log_message') as log:
            measures, _ = extract_jianpu_measures(score, 0, follow_key_changes=True)
        self.assertEqual(measures[0][0].symbol, '1')
        self.assertEqual(measures.key_changes[1], (7, '1=G'))
        self.assertTrue(any('中途变调' in str(call) for call in log.call_args_list))

    def test_explicit_relative_minor_change_keeps_same_reference_but_updates_label(self):
        score = make_score()
        second = score.parts[0].measure(2)
        second.remove(second.recurse().getElementsByClass(key.Key)[0])
        second.insert(0, key.Key('A', 'minor'))
        measures, _ = extract_jianpu_measures(score, 0, follow_key_changes=True)
        self.assertEqual(measures.key_changes[1], (0, '6=A'))
        self.assertEqual(measures[1][0].symbol, '5')

    @unittest.skipUnless(shutil.which('lilypond'), 'LilyPond is required')
    def test_actual_render_and_insitu_include_changed_key(self):
        import insitu.render_insitu as renderer
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            score = make_score()
            mxl, pdf, output = root / 'score.musicxml', root / 'staff.pdf', root / 'out.pdf'
            score.write('musicxml', fp=mxl)
            doc = pymupdf.open()
            page = doc.new_page(width=600, height=300)
            for y in (100, 105, 110, 115, 120):
                page.draw_line((40, y), (540, y), width=.5)
            for x in (165, 290, 415, 540):
                page.draw_line((x, 100), (x, 120), width=.5)
            doc.save(pdf)
            doc.close()
            with mock.patch.object(renderer, '_run_jianpu_ly', wraps=renderer._run_jianpu_ly) as run:
                renderer.build(pdf, mxl, output, expand=True)
            self.assertTrue(output.is_file())
            self.assertIn('1=G', run.call_args_list[0].args[0])
            text = build_jianpu_ly_text(converter.parse(str(mxl)), 'test')
            source = root / 'score.txt'
            source.write_text(text)
            script = Path(__file__).resolve().parents[1] / 'scripts' / 'jianpu-ly.py'
            parsed = subprocess.run([sys.executable, str(script), str(source)], capture_output=True, text=True)
            self.assertEqual(parsed.returncode, 0, parsed.stderr)
            ly = root / 'score.ly'
            ly.write_text(parsed.stdout)
            rendered = subprocess.run(['lilypond', '-o', str(root / 'score'), str(ly)],
                                      capture_output=True, text=True, env={**os.environ, 'TMPDIR': tmp})
            self.assertEqual(rendered.returncode, 0, rendered.stderr)


if __name__ == '__main__':
    unittest.main()
