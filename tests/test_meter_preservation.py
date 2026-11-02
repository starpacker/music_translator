"""Written meters remain visible independently of hidden layout bar lengths."""
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import pymupdf
from music21 import key, meter, note, stream
from core.notation.jianpu import build_jianpu_ly_text


def make_score():
    score, part = stream.Score(), stream.Part()
    for number, signature, duration in ((1, '4/4', 4), (2, '3/4', 3),
                                        (3, '3/4', 3), (4, '6/8', 3)):
        measure = stream.Measure(number=number)
        if number == 1:
            measure.insert(0, key.Key('C'))
        measure.insert(0, meter.TimeSignature(signature))
        measure.insert(0, note.Rest(quarterLength=duration))
        part.append(measure)
    score.insert(0, part)
    return score


class MeterPreservationTest(unittest.TestCase):
    def test_only_actual_meter_changes_are_printed(self):
        text = build_jianpu_ly_text(make_score(), 'test')
        for signature in ('4/4', '3/4', '6/8'):
            self.assertEqual(text.count('"' + signature + '"'), 1)
        self.assertEqual(text.count('^\\markup{\\concat{\\hspace#3'), 3)

    def test_absent_meter_does_not_invent_a_printed_mark(self):
        score, part, measure = stream.Score(), stream.Part(), stream.Measure(number=1)
        measure.insert(0, note.Note('C4', quarterLength=4))
        part.append(measure)
        score.insert(0, part)
        text = build_jianpu_ly_text(score, 'test')
        self.assertNotIn('^\\markup{\\concat{\\hspace#3', text)

    def test_unplaceable_meter_is_reported(self):
        score = make_score()
        score.parts[0].measure(1).insert(2, meter.TimeSignature('5/8'))
        with mock.patch('core.notation.jianpu.log_message') as log:
            text = build_jianpu_ly_text(score, 'test')
        self.assertNotIn('"5/8"', text)
        self.assertTrue(any('拍号' in str(call) and '不在可表示的音符边界' in str(call)
                            for call in log.call_args_list))

    @unittest.skipUnless(shutil.which('lilypond'), 'LilyPond is required')
    def test_musicxml_to_pdf_prints_meters_with_initial_rest(self):
        from core.render.renderer import generate_jianpu_pdf_from_mxl
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, target = root / 'score.musicxml', root / 'output.pdf'
            make_score().write('musicxml', fp=source)
            self.assertTrue(generate_jianpu_pdf_from_mxl(source, target, root))
            with pymupdf.open(target) as pdf:
                text = ''.join(page.get_text() for page in pdf)
                for signature in ('4/4', '3/4', '6/8'):
                    self.assertIn(signature, text)
                spans = [span for page in pdf for block in page.get_text('dict')['blocks']
                         for line in block.get('lines', []) for span in line.get('spans', [])]
                key_rects = [pymupdf.Rect(span['bbox']) for span in spans if '1=C' in span['text']]
                meter_rects = [pymupdf.Rect(span['bbox']) for span in spans if '4/4' in span['text']]
                self.assertTrue(key_rects and meter_rects)
                self.assertFalse(key_rects[0].intersects(meter_rects[0]))


if __name__ == '__main__':
    unittest.main()
