"""Keep MusicXML note marks through extraction and the real LilyPond path."""
from collections import Counter
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import pymupdf
from music21 import articulations, chord, converter, expressions, key, meter, note, stream, tie

from core.notation.jianpu import (_get_score_key_tonic, build_jianpu_ly_text,
                                  extract_jianpu_measures, jianpu_note_token)
from core.notation.jianpu.primitives import clone_jianpu_note, note_to_jianpu


def marked_score():
    score, part = stream.Score(), stream.Part()
    first = stream.Measure(number=1)
    first.insert(0, key.Key('C'))
    first.insert(0, meter.TimeSignature('4/4'))
    a = note.Note('C4', quarterLength=1)
    a.articulations.append(articulations.Accent())
    staccato = articulations.Staccato()
    staccato.placement = 'below'
    a.articulations.append(staccato)
    first.insert(0, a)
    ornament = note.Note('D4').getGrace()
    tenuto = articulations.Tenuto()
    tenuto.placement = 'above'
    ornament.articulations.append(tenuto)
    first.insert(1, ornament)
    b = note.Note('E4', quarterLength=3)
    b.expressions.extend([expressions.Fermata(), expressions.Trill()])
    first.insert(1, b)
    part.append(first)
    second = stream.Measure(number=2)
    simultaneous = chord.Chord(['C4', 'E4'], quarterLength=2)
    simultaneous.articulations.append(articulations.Accent())
    second.insert(0, simultaneous)
    start = note.Note('G4', quarterLength=2)
    start.tie = tie.Tie('start')
    second.insert(2, start)
    part.append(second)
    third = stream.Measure(number=3)
    stop = note.Note('G4', quarterLength=2)
    stop.tie = tie.Tie('stop')
    stop.expressions.append(expressions.Fermata())
    third.insert(0, stop)
    rest = note.Rest(quarterLength=2)
    rest.expressions.append(expressions.Fermata())
    third.insert(2, rest)
    part.append(third)
    score.insert(0, part)
    return score


class NotationPreservationTest(unittest.TestCase):
    def test_marks_follow_exact_notes_graces_chords_ties_and_rests(self):
        score = marked_score()
        tonic, _ = _get_score_key_tonic(score)
        measures, _ = extract_jianpu_measures(score, tonic, follow_key_changes=True)
        first = [jianpu_note_token(n) for n in measures[0]]
        self.assertEqual(first[0], r'1 \accent _\staccato')
        self.assertEqual(first[1], r'g[2] GraceMark:^\tenuto')
        self.assertEqual(first[2], r'3 \fermata ^\trill - -')
        self.assertEqual(jianpu_note_token(measures[1][0]), r'13 \accent -')
        self.assertEqual(jianpu_note_token(measures[2][0]), r'- \fermata -')
        self.assertEqual(jianpu_note_token(measures[2][1]), r'0 \fermata -')
        self.assertEqual([sum(n.duration for n in bar) for bar in measures], [4, 4, 4])
        # Reading marks and cloning do not alter the music21 source.
        source = list(score.parts[0].getElementsByClass(stream.Measure))[0]
        self.assertEqual([type(a).__name__ for a in source.notes[0].articulations], ['Accent', 'Staccato'])

    def test_long_note_and_tuplet_attach_before_dashes_or_group_end(self):
        original = note.Note('C4', quarterLength=3)
        original.articulations.append(articulations.Accent())
        jn = note_to_jianpu(original)
        self.assertEqual(jianpu_note_token(jn), r'1 \accent - -')
        jn.tuplet_start, jn.tuplet_stop = True, True
        jn.tuplet_actual, jn.tuplet_normal = 3, 2
        self.assertEqual(jianpu_note_token(jn), r'3[ 1 \accent - - ]')
        self.assertEqual(clone_jianpu_note(jn, 1, carry_lyrics=True).notations, (r'\accent',))
        self.assertEqual(clone_jianpu_note(jn, 1, carry_lyrics=False).notations, ())

    def test_unsupported_mark_and_ornament_accidental_are_reported(self):
        item = note.Note('C4')
        item.articulations.append(articulations.Fingering(2))
        trill = expressions.Trill()
        trill.accidental = 'sharp'
        item.expressions.append(trill)
        with mock.patch('core.notation.jianpu.primitives._log_message') as log:
            rendered = note_to_jianpu(item)
        self.assertIn(r'^\trill', rendered.notations)
        self.assertTrue(any('Fingering' in str(call) for call in log.call_args_list))
        self.assertTrue(any('升降号未完整表示' in str(call) for call in log.call_args_list))

    def test_real_fixture_counts_remain_attached(self):
        path = Path(__file__).resolve().parent / 'fixtures/bangdi.musicxml'
        score = converter.parse(path)
        tonic, _ = _get_score_key_tonic(score)
        measures, _ = extract_jianpu_measures(score, tonic, follow_key_changes=True)
        source = Counter(type(mark).__name__ for event in score.parts[0].recurse().notesAndRests
                         for mark in (*event.articulations, *event.expressions))
        extracted = Counter(mark.lstrip('^_') for bar in measures for event in bar for mark in event.notations)
        for name, lily in [('Accent', r'\accent'), ('Staccato', r'\staccato'),
                           ('Tenuto', r'\tenuto'), ('Trill', r'\trill')]:
            self.assertEqual(extracted[lily], source[name], name)

    def test_audit_finds_marks_lost_at_a_specific_event(self):
        from scripts.audit_jianpu_events import audit
        with tempfile.TemporaryDirectory() as tmp:
            xml = Path(tmp) / 'marked.musicxml'
            marked_score().write('musicxml', fp=xml)
            result = audit(xml)
            self.assertEqual(result['mismatch_count'], 0, result['mismatches'])
            self.assertEqual(result['source_supported_note_marks'], result['jianpu_supported_note_marks'])
            with mock.patch('core.notation.jianpu.primitives._source_notations', return_value=()):
                missing = audit(xml)
            self.assertTrue(any(item.get('reason') == 'note_mark_placement' for item in missing['mismatches']))

    def test_graphical_fallback_warns_that_marks_may_be_lost(self):
        from core.render import renderer
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with mock.patch.object(renderer, 'render_jianpu_ly', return_value=False), \
                 mock.patch.object(renderer, 'register_pdf_font', return_value='Helvetica'), \
                 mock.patch.object(renderer, 'create_pdf'), \
                 mock.patch.object(renderer, 'log_message') as log:
                ok, _, _ = renderer.render_score_to_jianpu_pdf(
                    marked_score(), 'test', root / 'out.pdf', root, root / 'out.txt', root / 'out.ly')
            self.assertTrue(ok)
            self.assertTrue(any('备用图形或文字排版不能完整显示' in str(call) for call in log.call_args_list))

    @unittest.skipUnless(shutil.which('lilypond'), 'LilyPond is required')
    def test_consecutive_marked_graces_keep_one_group_and_correct_targets(self):
        from insitu.render_insitu import _jianpu_ly_source, _render_ly_tile, _run_jianpu_ly
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            notes = r'g[2] GraceMark:\accent g[3] GraceMark:\staccato 1 - - -'
            ly = _run_jianpu_ly(_jianpu_ly_source([[notes]], '1=C', '4/4', 'test'), root)
            # One grace group is emitted for layout and one for the western
            # companion; the marks stay on their respective grace notes.
            self.assertEqual(ly.count(r'\grace {'), 2)
            self.assertIn(r"d'16 \accent e'16 \staccato", ly)
            self.assertTrue(_render_ly_tile(ly, root)[0].is_file())
            long_name = r'g[2] GraceMark:\staccatissimo g[3] 1 - - -'
            ly = _run_jianpu_ly(_jianpu_ly_source([[long_name]], '1=C', '4/4', 'test'), root)
            self.assertIn(r"d'16 \staccatissimo e'16", ly)
            self.assertTrue(_render_ly_tile(ly, root)[0].is_file())

    @unittest.skipUnless(shutil.which('lilypond'), 'LilyPond is required')
    def test_musicxml_to_real_pdf_renders_marks_and_grace_mark(self):
        from core.render.renderer import generate_jianpu_pdf_from_mxl
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            xml, output, workspace = root / 'score.musicxml', root / 'out.pdf', root / 'workspace'
            marked_score().write('musicxml', fp=xml)
            self.assertTrue(generate_jianpu_pdf_from_mxl(xml, output, root, editor_workspace_dir=workspace))
            text = next(workspace.glob('*.jianpu.txt')).read_text()
            self.assertIn(r'\accent', text)
            self.assertIn(r'\fermata', text)
            self.assertIn(r'\trill', text)
            self.assertIn(r'GraceMark:^\tenuto', text)
            with pymupdf.open(output) as doc:
                self.assertGreater(len(doc), 0)
                # Jianpu digits use Nimbus; Emmentaler glyphs here include the
                # rendered accent/tenuto/fermata/trill, not just commands.
                music_glyphs = sum(len(span['chars']) for page in doc for span in page.get_texttrace()
                                   if span.get('font', '').startswith('Emmentaler'))
                self.assertGreaterEqual(music_glyphs, 5)


if __name__ == '__main__':
    unittest.main()
