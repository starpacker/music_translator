"""Only unique music-font dots beside matched noteheads are partial evidence."""
from unittest import TestCase, mock
from music21 import note, stream

from core.omr.vector_augmentation_dots import compare_augmentation_dots, verified_dot_glyphs
from insitu.pdf_structure import Staff


def example(dots=1):
    measure = stream.Measure(number=1)
    item = note.Note('E4', type='half')
    item.duration.dots = dots
    measure.append(item)
    staff = Staff(1, 100, 120, 40, 250, 5)
    head = {'x': 100., 'y': 120., 'size': 20., 'kind': 'half', 'bbox': (100., 105., 107., 124.)}
    match = [((100.,), 0, False)]
    return measure, staff, head, match


def dot(x=110., y=118., size=20.):
    return {'x': x, 'y': y, 'size': size, 'bbox': (x, y - 10, x + 2, y + 2), 'code': 46}


class AugmentationDotTest(TestCase):
    def test_music_font_identity_must_be_unique_and_visible(self):
        class Font:
            def __init__(self, name, glyphs):
                self.name, self.glyphs = name, glyphs
            def has_glyph(self, code):
                return self.glyphs.get(code, 0)
        doc, page = mock.Mock(), mock.Mock()
        page.get_fonts.return_value = [(1, None, None, 'ABCDEF+'), (2, None, None, 'GHIJKL+')]
        doc.extract_font.side_effect = lambda x: ('', '', '', bytes([x]))
        page.get_texttrace.return_value = [{'font': '', 'size': 20, 'opacity': 1,
            'chars': [(46, 17, (110, 118), (110, 108, 112, 120))]}]
        music = Font('EngraverFontSet Regular', {0xF02E: 17})
        other = Font('EngraverTextT Regular', {0xF02E: 9})
        with mock.patch('core.omr.vector_note_values.pymupdf.Font', side_effect=[music, other]):
            self.assertEqual(len(verified_dot_glyphs(doc, page)), 1)
        other.glyphs[0xF02E] = 17
        with mock.patch('core.omr.vector_note_values.pymupdf.Font', side_effect=[music, other]):
            self.assertEqual(verified_dot_glyphs(doc, page), [])
        other.glyphs[0xF02E] = 9
        page.get_texttrace.return_value[0]['opacity'] = 0
        with mock.patch('core.omr.vector_note_values.pymupdf.Font', side_effect=[music, other]):
            self.assertEqual(verified_dot_glyphs(doc, page), [])

    def test_matched_dot_and_double_dot_are_positive_evidence(self):
        for expected, dots in ((1, [dot()]), (2, [dot(), dot(113)])):
            with self.subTest(expected=expected):
                measure, staff, head, matched = example(expected)
                result = compare_augmentation_dots(measure, staff, 40, 250, matched, [head], dots)
                self.assertEqual(result['consistent_dotted_noteheads'], 1)
                self.assertEqual(result['unverified_dotted_noteheads'], 0)
                self.assertEqual(result['consistent_associations'][0]['head_x'], 100)
                self.assertEqual(len(result['consistent_associations'][0]['dot_points']), expected)
                self.assertEqual(result['reviews'], [])

    def test_conflict_is_reviewed_and_absence_never_proves_no_dot(self):
        measure, staff, head, matched = example(0)
        result = compare_augmentation_dots(measure, staff, 40, 250, matched, [head], [dot()])
        self.assertEqual(result['reviews'][0]['musicxml_dot_count'], 0)
        self.assertEqual(result['reviews'][0]['uniquely_assigned_pdf_dots'], 1)
        measure, staff, head, matched = example(1)
        missing = compare_augmentation_dots(measure, staff, 40, 250, matched, [head], [])
        self.assertEqual(missing['unverified_dotted_noteheads'], 1)
        self.assertEqual(missing['reviews'], [])

    def test_ambiguous_and_wrong_position_dots_are_not_assigned(self):
        measure, staff, head, matched = example(1)
        second = note.Note('E4', type='half')
        second.duration.dots = 1
        measure.append(second)
        other = {**head, 'x': 102., 'bbox': (102., 105., 109., 124.)}
        result = compare_augmentation_dots(measure, staff, 40, 250,
                    [matched[0], ((102.,), 3, False)], [head, other], [dot(112.5)])
        self.assertEqual(result['ambiguous_pdf_dot_glyphs'], 1)
        self.assertEqual(result['ambiguous_dot_points'][0]['candidate_event_indices'], [1, 2])
        self.assertEqual(result['unverified_dotted_noteheads'], 2)
        self.assertEqual(result['reviews'], [])
        measure, staff, head, matched = example(0)
        wrong = compare_augmentation_dots(measure, staff, 40, 250, matched, [head], [dot(110, 110)])
        self.assertEqual(wrong['unassigned_pdf_dot_glyphs'], 1)
        self.assertEqual(wrong['reviews'], [])

    def test_duplicate_pdf_paths_do_not_make_a_double_dot(self):
        measure, staff, head, matched = example(1)
        result = compare_augmentation_dots(measure, staff, 40, 250, matched, [head], [dot(), dot()])
        self.assertEqual(result['detected_pdf_dot_glyphs'], 1)
        self.assertEqual(result['consistent_dotted_noteheads'], 1)
