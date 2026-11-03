"""Independent voices must not be discarded merely because they overlap."""
from pathlib import Path
import json
import shutil
import tempfile
import unittest
from unittest import mock

import pymupdf
from music21 import chord, converter, key, meter, note, stream
from core.notation.jianpu import build_jianpu_ly_text, extract_jianpu_measures


def without_meter_label(text):
    return text.replace(' ^\\markup{\\concat{\\hspace#3"4/4"}}', '')


def make_score():
    score, part, measure = stream.Score(), stream.Part(), stream.Measure(number=1)
    measure.insert(0, key.KeySignature(0))
    measure.insert(0, meter.TimeSignature('4/4'))
    upper, lower = stream.Voice(id='1'), stream.Voice(id='2')
    upper.insert(0, note.Note('C5', quarterLength=4))
    for beat, pitch in enumerate(('D4', 'E4', 'F4', 'G4')):
        lower.insert(beat, note.Note(pitch, quarterLength=1))
    measure.insert(0, upper)
    measure.insert(0, lower)
    part.append(measure)
    score.insert(0, part)
    return score


class PolyphonicPreservationTest(unittest.TestCase):
    def test_single_measure_voice_with_overlapping_notes_is_not_dropped(self):
        text, groups, _ = build_jianpu_ly_text(make_score(), 'test', _return_groups=True)
        self.assertEqual(groups, [[0], [1]])
        self.assertIn('1\' - - -', without_meter_label(text))
        self.assertIn('2 3 4 5', without_meter_label(text))
        self.assertEqual(text.count('NextPart'), 1)

    def test_more_than_four_voices_are_not_truncated(self):
        score = make_score()
        measure = score.parts[0].measure(1)
        for i in range(3, 7):
            voice = stream.Voice(id=str(i))
            voice.insert(0, note.Note('A4', quarterLength=4))
            measure.insert(0, voice)
        text, groups, _ = build_jianpu_ly_text(score, 'test', _return_groups=True)
        self.assertEqual(len(groups), 6)
        self.assertEqual(text.count('NextPart'), 5)

    def test_absent_voice_does_not_duplicate_an_unvoiced_chord(self):
        score = make_score()
        second = stream.Measure(number=2)
        second.insert(0, chord.Chord(['C4', 'E4'], quarterLength=4))
        score.parts[0].append(second)
        measures, _ = extract_jianpu_measures(score, 0, _part=score.parts[0], _voice_id='2',
                                              _multi_voice_mode=True, _is_primary_voice=False)
        self.assertTrue(all(event.is_rest for event in measures[1]))

    @unittest.skipUnless(shutil.which('lilypond'), 'LilyPond is required')
    def test_standalone_musicxml_pipeline_retains_both_voices(self):
        from core.render.renderer import generate_jianpu_pdf_from_mxl
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mxl, out, workspace = root / 'score.musicxml', root / 'out.pdf', root / 'workspace'
            make_score().write('musicxml', fp=mxl)
            self.assertTrue(generate_jianpu_pdf_from_mxl(mxl, out, root, editor_workspace_dir=workspace))
            text = next(workspace.glob('*.jianpu.txt')).read_text()
            self.assertEqual(text.splitlines().count('NextPart'), 1)
            self.assertIn('2 3 4 5', without_meter_label(text))

    @unittest.skipUnless(shutil.which('lilypond'), 'LilyPond is required')
    def test_insitu_renders_both_voices_as_separate_rows(self):
        import insitu.render_insitu as renderer
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf, mxl, out = root / 'staff.pdf', root / 'score.musicxml', root / 'out.pdf'
            doc = pymupdf.open()
            page = doc.new_page(width=600, height=300)
            for y in (100, 105, 110, 115, 120):
                page.draw_line((40, y), (540, y), width=.5)
            page.draw_line((540, 100), (540, 120), width=.5)
            doc.save(pdf)
            doc.close()
            make_score().write('musicxml', fp=mxl)
            parsed = converter.parse(str(mxl))
            text = build_jianpu_ly_text(parsed, 'test')
            self.assertEqual(text.count('NextPart'), 1)
            renderer.build(pdf, mxl, out)
            report = json.loads((root / 'out.report.json').read_text())
            voices = report['rows'][0]['voices']
            self.assertEqual(len(voices), 2)
            self.assertTrue(all(voice['lines'] for voice in voices))
            # The second voice starts below the first instead of overlapping it.
            self.assertGreater(voices[1]['lines'][0]['top'], voices[0]['lines'][0]['top'])
            with pymupdf.open(out) as output:
                self.assertEqual(output[0].rect, pymupdf.Rect(0, 0, 600, 300))


if __name__ == '__main__':
    unittest.main()
