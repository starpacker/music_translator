"""Partial ties must not turn newly attacked chord pitches into dashes."""
from pathlib import Path
import json
import shutil
import tempfile
import unittest
from unittest import mock

import pymupdf
from music21 import chord, key, meter, note, stream, tie

from core.notation.jianpu import extract_jianpu_measures, jianpu_note_token
from scripts.audit_jianpu_events import audit


def score_with_ties(first_pitches, first_ties, second_pitches, second_ties):
    score, part = stream.Score(), stream.Part()
    for index, (pitches, kinds) in enumerate(((first_pitches, first_ties),
                                               (second_pitches, second_ties)), 1):
        measure = stream.Measure(number=index)
        if index == 1:
            measure.insert(0, key.Key('C'))
            measure.insert(0, meter.TimeSignature('4/4'))
        event = chord.Chord(pitches, quarterLength=4)
        for component, kind in zip(event.notes, kinds):
            if kind:
                component.tie = tie.Tie(kind)
        measure.insert(0, event)
        part.append(measure)
    score.insert(0, part)
    return score


class PartialChordTieTest(unittest.TestCase):
    def test_partial_tie_keeps_both_chords_and_is_reported(self):
        score = score_with_ties(['C4', 'E4'], ['start', None],
                                ['C4', 'G4'], ['stop', None])
        with mock.patch('core.notation.jianpu.extract.log_message') as log:
            measures, _ = extract_jianpu_measures(score, 0, follow_key_changes=True)
        self.assertEqual([jianpu_note_token(m[0]) for m in measures], ['13 - - -', '15 - - -'])
        self.assertTrue(any('部分延音' in str(call) for call in log.call_args_list))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'score.musicxml'
            score.write('musicxml', fp=path)
            result = audit(path)
            partial = [m for m in result['mismatches'] if m.get('reason') == 'partial_chord_tie_unrepresented']
            self.assertEqual([m['index'] for m in partial], [1, 2])
            self.assertEqual(partial[1]['pitches'], [60, 67])

    def test_entire_identical_chord_can_still_continue_as_dashes(self):
        score = score_with_ties(['C4', 'E4'], ['start', 'start'],
                                ['C4', 'E4'], ['stop', 'stop'])
        measures, _ = extract_jianpu_measures(score, 0, follow_key_changes=True)
        self.assertEqual([jianpu_note_token(m[0]) for m in measures], ['13 - - -', '- - - -'])
        self.assertFalse(any(event.tie_review for bar in measures for event in bar))
        with tempfile.TemporaryDirectory() as tmp:
            xml = Path(tmp) / 'score.musicxml'
            score.write('musicxml', fp=xml)
            self.assertEqual(audit(xml)['mismatch_count'], 0)

    def test_subset_of_previous_chord_does_not_repeat_a_released_pitch(self):
        score = score_with_ties(['C4', 'E4', 'G4'], ['start', 'start', 'start'],
                                ['C4', 'E4'], ['stop', 'stop'])
        with mock.patch('core.notation.jianpu.extract.log_message') as log:
            measures, _ = extract_jianpu_measures(score, 0, follow_key_changes=True)
        self.assertEqual([jianpu_note_token(m[0]) for m in measures], ['135 - - -', '13 - - -'])
        self.assertEqual(measures[1][0].tie_review, 'unmatched_tie')
        self.assertTrue(any('不匹配延音' in str(call) for call in log.call_args_list))

    def test_full_continuation_with_partial_outgoing_ties_is_flagged(self):
        score = score_with_ties(['C4', 'E4'], ['start', 'start'],
                                ['C4', 'E4'], ['continue', 'stop'])
        third = stream.Measure(number=3)
        event = note.Note('C4', quarterLength=4)
        event.tie = tie.Tie('stop')
        third.insert(0, event)
        score.parts[0].append(third)
        measures, _ = extract_jianpu_measures(score, 0, follow_key_changes=True)
        self.assertEqual([jianpu_note_token(m[0]) for m in measures],
                         ['13 - - -', '- - - -', '1 - - -'])
        self.assertEqual(measures[1][0].tie_review, 'partial_chord_tie')
        self.assertEqual(measures[2][0].tie_review, 'unmatched_tie')

    def test_orphan_stop_is_printed_as_a_note(self):
        score, part, measure = stream.Score(), stream.Part(), stream.Measure(number=1)
        measure.insert(0, key.Key('C'))
        event = note.Note('C4', quarterLength=4)
        event.tie = tie.Tie('stop')
        measure.insert(0, event)
        part.append(measure)
        score.insert(0, part)
        with mock.patch('core.notation.jianpu.extract.log_message') as log:
            measures, _ = extract_jianpu_measures(score, 0, follow_key_changes=True)
        self.assertEqual(jianpu_note_token(measures[0][0]), '1 - - -')
        self.assertTrue(any('不匹配延音' in str(call) for call in log.call_args_list))

    @unittest.skipUnless(shutil.which('lilypond'), 'LilyPond is required')
    def test_both_pdf_paths_keep_new_chord_pitch(self):
        from core.render import renderer
        from insitu import render_insitu
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            score = score_with_ties(['C4', 'E4'], ['start', None],
                                    ['C4', 'G4'], ['stop', None])
            xml, staff = root / 'score.musicxml', root / 'staff.pdf'
            score.write('musicxml', fp=xml)
            with pymupdf.open() as doc:
                page = doc.new_page(width=600, height=240)
                for y in (100, 105, 110, 115, 120):
                    page.draw_line((40, y), (540, y), width=.5)
                for x in (290, 540):
                    page.draw_line((x, 100), (x, 120), width=.5)
                doc.save(staff)
            workspace = root / 'workspace'
            self.assertTrue(renderer.generate_jianpu_pdf_from_mxl(
                xml, root / 'jianpu.pdf', root, editor_workspace_dir=workspace))
            text = next(workspace.glob('*.jianpu.txt')).read_text()
            self.assertIn('15 - - -', text)
            with mock.patch.object(render_insitu, '_run_jianpu_ly',
                                   wraps=render_insitu._run_jianpu_ly) as render:
                render_insitu.build(staff, xml, root / 'insitu.pdf', expand=True)
            self.assertIn('15 - - -', render.call_args_list[0].args[0])
            self.assertTrue((root / 'insitu.pdf').is_file())
            report = json.loads((root / 'insitu.report.json').read_text())
            self.assertEqual(report['status'], 'needs_review')
            self.assertEqual(report['tie_review_measure_count'], 2)
            self.assertEqual(report['rows'][0]['voices'][0]['tie_review_measures'],
                             [{'musicxml_index': 1, 'reasons': ['partial_chord_tie']},
                              {'musicxml_index': 2, 'reasons': ['partial_chord_tie']}])


if __name__ == '__main__':
    unittest.main()
