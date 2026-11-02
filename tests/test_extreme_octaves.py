"""Very high and low notes retain their register and visible octave dots."""
from collections import Counter
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np
import pymupdf
from music21 import converter, key, layout, meter, note, stream

from core.notation.jianpu import _get_score_key_tonic, extract_jianpu_measures, jianpu_note_token
from core.notation.jianpu.primitives import note_to_jianpu
from scripts.audit_jianpu_events import audit


class ExtremeOctaveTest(unittest.TestCase):
    def test_pitch_outside_midi_range_is_not_folded_by_an_octave(self):
        with mock.patch('core.notation.jianpu.primitives._log_message') as log:
            converted = note_to_jianpu(note.Note('C10'), 0, tonic_step='C')
        self.assertEqual(converted.midi, 132)
        self.assertEqual(converted.upper_dots, 6)
        self.assertTrue(any('超出 MIDI' in str(call) for call in log.call_args_list))

    @unittest.skipUnless(shutil.which('lilypond'), 'LilyPond is required')
    def test_four_and_five_dots_visible_and_midi_register_correct(self):
        from insitu.render_insitu import _jianpu_ly_source, _render_ly_tile, _run_jianpu_ly
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            text = "1'''' 1''''' 1,,,, 1,,,,,"
            ly = _run_jianpu_ly(_jianpu_ly_source([[text]], '1=C', '4/4', 'test'), root)
            pdf, _ = _render_ly_tile(ly, root)
            midi = converter.parse(str(root / 'line.midi'), forceSource=True)
            self.assertEqual([n.pitch.midi for n in midi.recurse().notes], [108, 120, 12, 0])
            with pymupdf.open(pdf) as doc:
                dots = Counter(round(char[2][0], 1) for span in doc[0].get_texttrace()
                               for char in span['chars'] if char[0] == ord('.'))
                self.assertEqual(sorted(dots.values()), [4, 4, 5, 5])

    @unittest.skipUnless(shutil.which('lilypond'), 'LilyPond is required')
    def test_extreme_octaves_in_graces_and_chord(self):
        from insitu.render_insitu import _jianpu_ly_source, _render_ly_tile, _run_jianpu_ly
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            text = "g[1''''2,,,,] 1''''5,,,, - - -"
            ly = _run_jianpu_ly(_jianpu_ly_source([[text]], '1=C', '4/4', 'test'), root)
            pdf, _ = _render_ly_tile(ly, root)
            self.assertTrue(pdf.is_file())
            midi = converter.parse(str(root / 'line.midi'), forceSource=True)
            events = list(midi.recurse().notes)
            self.assertEqual(sorted(p.midi for p in events[-1].pitches), [19, 108])
            self.assertIn(r'\four-dots', ly)

    @unittest.skipUnless(shutil.which('lilypond'), 'LilyPond is required')
    def test_musicxml_to_standalone_pdf_preserves_extreme_register(self):
        from core.render import renderer
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            score, part, measure = stream.Score(), stream.Part(), stream.Measure(number=1)
            measure.insert(0, key.Key('C'))
            measure.insert(0, meter.TimeSignature('4/4'))
            for index, pitch in enumerate(('C8', 'C9', 'C0', 'C1')):
                measure.insert(index, note.Note(pitch, quarterLength=1))
            part.append(measure)
            score.insert(0, part)
            source = root / 'score.musicxml'
            score.write('musicxml', fp=source)
            parsed = converter.parse(source)
            tonic, _ = _get_score_key_tonic(parsed)
            measures, _ = extract_jianpu_measures(parsed, tonic, follow_key_changes=True)
            self.assertEqual([jianpu_note_token(n) for n in measures[0]],
                             ["1''''", "1'''''", '1,,,,', '1,,,'])
            self.assertEqual(audit(source)['mismatch_count'], 0)
            with mock.patch.object(renderer, 'render_jianpu_ly', wraps=renderer.render_jianpu_ly) as typeset:
                self.assertTrue(renderer.generate_jianpu_pdf_from_mxl(source, root / 'result.pdf', root))
            self.assertEqual(typeset.call_count, 1)
            with pymupdf.open(root / 'result.pdf') as result:
                self.assertGreaterEqual(result[0].get_text().count('.'), 13)

    @unittest.skipUnless(shutil.which('lilypond'), 'LilyPond is required')
    def test_insitu_keeps_five_high_dots_clear_of_the_next_staff(self):
        from insitu.render_insitu import build
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, xml, output = root / 'staff.pdf', root / 'score.musicxml', root / 'insitu.pdf'
            with pymupdf.open() as doc:
                page = doc.new_page(width=600, height=360)
                for top in (100, 190):
                    for dy in (0, 5, 10, 15, 20):
                        page.draw_line((40, top + dy), (540, top + dy), width=.5)
                    page.draw_line((540, top), (540, top + 20), width=.5)
                doc.save(source)
            score, part = stream.Score(), stream.Part()
            for number, pitch in ((1, 'C9'), (2, 'C0')):
                measure = stream.Measure(number=number)
                if number == 1:
                    measure.insert(0, key.Key('C'))
                    measure.insert(0, meter.TimeSignature('4/4'))
                else:
                    measure.insert(0, layout.SystemLayout(isNew=True))
                measure.insert(0, note.Note(pitch, quarterLength=4))
                part.append(measure)
            score.insert(0, part)
            score.write('musicxml', fp=xml)
            build(source, xml, output, expand=True)
            with pymupdf.open(output) as doc:
                page = doc[0]
                pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), colorspace=pymupdf.csGRAY, alpha=False)
                pixels = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.stride)
                candidates = np.flatnonzero(pixels[:, 1000] < 128)
                groups = np.split(candidates, np.where(np.diff(candidates) > 1)[0] + 1)
                centers = [int(group[len(group) // 2]) for group in groups if len(group)]
                staff_starts = [y for y in centers
                                if all(any(abs(v - (y + 10 * step)) <= 1 for v in centers)
                                       for step in range(5))]
                self.assertEqual(len(staff_starts), 2, staff_starts)
                first_bottom, second_top = (staff_starts[0] + 40) / 2, staff_starts[1] / 2
                # Placed PDF forms can expose text from clipped-away copies in
                # get_texttrace, so count only dot positions with visible ink.
                dots = set()
                for span in page.get_texttrace():
                    for char in span['chars']:
                        if char[0] != ord('.'):
                            continue
                        x, y = char[2]
                        x0, x1 = max(0, int((x + .2) * 2)), min(pix.width, int((x + 3) * 2))
                        y0, y1 = max(0, int((y - 2) * 2)), min(pix.height, int((y + .5) * 2))
                        if x1 > x0 and y1 > y0 and np.any(pixels[y0:y1, x0:x1] < 128):
                            dots.add((round(x, 1), round(y, 1)))
                self.assertEqual(sum(first_bottom < y < second_top for _, y in dots), 5)
                self.assertEqual(sum(y > second_top + 20 for _, y in dots), 4)


if __name__ == '__main__':
    unittest.main()
