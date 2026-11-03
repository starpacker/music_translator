"""Every staff renders exactly one jianpu line.

The source scores are monophonic.  OMR sometimes tags notes inside a phrase as
a second MusicXML voice; that fragment is recognition noise and must never
become a second jianpu line.
"""
from pathlib import Path
import json
import shutil
import tempfile
import unittest

import pymupdf
from music21 import converter, key, meter, note, stream
from core.notation.jianpu import build_jianpu_ly_text, extract_jianpu_measures


def without_meter_label(text):
    return text.replace(' ^\\markup{\\concat{\\hspace#3"4/4"}}', '')


def make_score():
    """One measure where OMR tagged a second voice under the melody."""
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


class SingleVoicePolicyTest(unittest.TestCase):
    def test_two_voices_in_one_measure_render_one_line(self):
        text, groups, _ = build_jianpu_ly_text(make_score(), 'test', _return_groups=True)
        self.assertEqual(groups, [[0]])
        self.assertEqual(text.count('NextPart'), 0)
        self.assertIn("1' - - -", without_meter_label(text))
        self.assertNotIn('2 3 4 5', without_meter_label(text))

    def test_second_voice_throughout_the_part_is_still_one_line(self):
        """Even a sustained second voice is merged: sources are monophonic."""
        score, part = stream.Score(), stream.Part()
        for number in range(1, 21):
            measure = stream.Measure(number=number)
            if number == 1:
                measure.insert(0, key.KeySignature(0))
                measure.insert(0, meter.TimeSignature('4/4'))
            upper = stream.Voice(id='1')
            upper.insert(0, note.Note('C5', quarterLength=4))
            lower = stream.Voice(id='2')
            lower.insert(0, note.Note('D4', quarterLength=4))
            measure.insert(0, upper)
            measure.insert(0, lower)
            part.append(measure)
        score.insert(0, part)
        text, groups, _ = build_jianpu_ly_text(score, 'test', _return_groups=True)
        self.assertEqual(groups, [[0]])
        self.assertEqual(text.count('NextPart'), 0)
        self.assertEqual(without_meter_label(text).count("1' - - -"), 20)
        self.assertNotIn('2 - - -', without_meter_label(text))

    def test_one_section_per_part_in_a_multi_part_score(self):
        score = stream.Score()
        for pitch in ('C5', 'G4'):
            part = stream.Part()
            measure = stream.Measure(number=1)
            measure.insert(0, key.KeySignature(0))
            measure.insert(0, meter.TimeSignature('4/4'))
            measure.insert(0, note.Note(pitch, quarterLength=4))
            part.append(measure)
            score.insert(0, part)
        text, groups, _ = build_jianpu_ly_text(score, 'test', _return_groups=True)
        self.assertEqual(groups, [[0], [1]])
        self.assertEqual(text.count('NextPart'), 1)

    def test_measure_without_the_primary_voice_keeps_its_notes(self):
        score = make_score()
        second = stream.Measure(number=2)
        only = stream.Voice(id='2')
        only.insert(0, note.Note('A4', quarterLength=4))
        second.insert(0, only)
        score.parts[0].append(second)
        measures, _ = extract_jianpu_measures(score, 0, _part=score.parts[0])
        self.assertFalse(all(event.is_rest for event in measures[1]))

    @unittest.skipUnless(shutil.which('lilypond'), 'LilyPond is required')
    def test_standalone_musicxml_pipeline_emits_one_line(self):
        from core.render.renderer import generate_jianpu_pdf_from_mxl
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mxl, out, workspace = root / 'score.musicxml', root / 'out.pdf', root / 'workspace'
            make_score().write('musicxml', fp=mxl)
            self.assertTrue(generate_jianpu_pdf_from_mxl(mxl, out, root, editor_workspace_dir=workspace))
            text = next(workspace.glob('*.jianpu.txt')).read_text()
            self.assertEqual(text.splitlines().count('NextPart'), 0)
            self.assertNotIn('2 3 4 5', without_meter_label(text))

    @unittest.skipUnless(shutil.which('lilypond'), 'LilyPond is required')
    def test_insitu_renders_a_second_voice_as_a_single_row(self):
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
            renderer.build(pdf, mxl, out)
            report = json.loads((root / 'out.report.json').read_text())
            voices = report['rows'][0]['voices']
            self.assertEqual(len(voices), 1)
            self.assertTrue(voices[0]['lines'])


if __name__ == '__main__':
    unittest.main()
