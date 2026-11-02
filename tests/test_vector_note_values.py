"""Notehead shapes and complete connected beams provide only partial rhythm evidence."""
from unittest import TestCase, mock
import pymupdf
from music21 import note, stream

from core.omr.vector_note_values import compare_note_values, verified_notehead_glyphs
from insitu.pdf_structure import Staff


def line(a, b):
    return ('l', pymupdf.Point(*a), pymupdf.Point(*b))


def stem(x, bottom):
    return {'type': 's', 'rect': pymupdf.Rect(x, 95, x, bottom), 'width': .5,
            'items': [line((x, 95), (x, bottom))]}


def beam(y):
    pts = [(105, y), (205, y), (205, y + 2), (105, y + 2), (105, y)]
    return {'type': 'f', 'rect': pymupdf.Rect(105, y, 205, y + 2), 'fill': (0, 0, 0),
            'items': [line(a, b) for a, b in zip(pts, pts[1:])]}


def example(first='eighth', second='eighth'):
    measure = stream.Measure(number=1)
    measure.append(note.Note('C4', type=first))
    measure.append(note.Note('D4', type=second))
    staff = Staff(1, 100, 120, 40, 250, 5)
    glyphs = [{'x': 100., 'y': 125., 'size': 20., 'kind': 'filled', 'bbox': (100., 110., 105.24, 127.)},
              {'x': 200., 'y': 122.5, 'size': 20., 'kind': 'filled', 'bbox': (200., 108., 205.24, 125.)}]
    matched = [((100.,), 0, False), ((200.,), .5, False)]
    return measure, staff, glyphs, matched


class NoteValueEvidenceTest(TestCase):
    def test_font_identity_is_required_and_must_be_unique(self):
        class Font:
            def __init__(self, name, matches):
                self.name, self.matches = name, matches
            def has_glyph(self, code):
                return self.matches.get(code, 0)
        doc, page = mock.Mock(), mock.Mock()
        page.get_fonts.return_value = [(1, None, None, 'ABCDEF+'), (2, None, None, 'GHIJKL+')]
        doc.extract_font.side_effect = lambda x: ('', '', '', bytes([x]))
        page.get_texttrace.return_value = [{'font': '', 'size': 20,
            'chars': [(207, 169, (100, 125), (100, 110, 105.24, 127))]}]
        music = Font('EngraverFontSet Regular', {0xF0CF: 169})
        other = Font('EngraverTextT Regular', {0xF0CF: 8})
        with mock.patch('core.omr.vector_note_values.pymupdf.Font', side_effect=[music, other]):
            self.assertEqual(verified_notehead_glyphs(doc, page)[0]['kind'], 'filled')
        page.get_texttrace.return_value[0]['opacity'] = 0
        with mock.patch('core.omr.vector_note_values.pymupdf.Font', side_effect=[music, other]):
            self.assertEqual(verified_notehead_glyphs(doc, page), [])
        del page.get_texttrace.return_value[0]['opacity']
        other.matches[0xF0CF] = 169
        with mock.patch('core.omr.vector_note_values.pymupdf.Font', side_effect=[music, other]):
            self.assertEqual(verified_notehead_glyphs(doc, page), [])

    def test_two_connected_stems_supply_a_lower_bound(self):
        measure, staff, glyphs, matched = example()
        evidence = compare_note_values(measure, staff, 40, 250, matched, glyphs,
                                       [stem(105, 125), stem(205, 122.5), beam(95)])
        self.assertEqual(evidence['head_shapes_agree'], 2)
        self.assertFalse(evidence['head_shape_conflicts'])
        self.assertEqual([item['connected_beams_at_least'] for item in evidence['beam_evidence']], [1, 1])
        self.assertTrue(all(item['status'] == 'connected_beam_lower_bound_consistent'
                            for item in evidence['beam_evidence']))

    def test_more_visible_beams_than_written_are_flagged_for_review(self):
        measure, staff, glyphs, matched = example()
        evidence = compare_note_values(measure, staff, 40, 250, matched, glyphs,
                                       [stem(105, 125), stem(205, 122.5), beam(95), beam(98)])
        self.assertEqual(len(evidence['beam_evidence']), 2)
        self.assertTrue(all(item['status'] == 'more_beams_than_musicxml_review'
                            for item in evidence['beam_evidence']))

    def test_duplicate_and_nonclosed_polygons_do_not_invent_beams(self):
        measure, staff, glyphs, matched = example()
        duplicate = compare_note_values(measure, staff, 40, 250, matched, glyphs,
                                        [stem(105, 125), stem(205, 122.5), beam(95), beam(95)])
        self.assertEqual([item['connected_beams_at_least'] for item in duplicate['beam_evidence']], [1, 1])
        malformed = beam(95)
        malformed['items'][-1] = line((105, 97), (108, 95))
        evidence = compare_note_values(measure, staff, 40, 250, matched, glyphs,
                                       [stem(105, 125), stem(205, 122.5), malformed])
        self.assertEqual(evidence['beam_evidence'], [])

    def test_absent_or_unpaired_beam_never_proves_a_duration(self):
        measure, staff, glyphs, matched = example(first='16th', second='16th')
        missing = compare_note_values(measure, staff, 40, 250, matched, glyphs, [stem(105, 125), beam(95)])
        self.assertEqual(missing['beam_evidence'], [])
        partial = compare_note_values(measure, staff, 40, 250, matched, glyphs,
                                      [stem(105, 125), stem(205, 122.5), beam(95)])
        self.assertTrue(all(item['status'] == 'fewer_connected_beams_review'
                            for item in partial['beam_evidence']))

    def test_invisible_beam_is_not_rhythm_evidence(self):
        measure, staff, glyphs, matched = example()
        hidden = beam(95)
        hidden['fill_opacity'] = 0
        result = compare_note_values(measure, staff, 40, 250, matched, glyphs,
                                     [stem(105, 125), stem(205, 122.5), hidden])
        self.assertEqual(result['beam_evidence'], [])

    def test_unmatched_or_incompatible_head_is_not_hidden(self):
        measure, staff, glyphs, matched = example()
        glyphs[0]['kind'] = 'whole'
        evidence = compare_note_values(measure, staff, 40, 250, matched, glyphs, [])
        self.assertEqual(len(evidence['head_shape_conflicts']), 1)
        self.assertEqual(evidence['head_shape_conflicts'][0]['musicxml_type'], 'eighth')
        missing = compare_note_values(measure, staff, 40, 250, matched, [], [])
        self.assertEqual(missing['head_shapes_unverified'], 2)
        self.assertEqual(missing['beam_evidence'], [])

    def test_cue_sized_or_full_sized_grace_contradictions_are_only_review_hints(self):
        measure, staff, glyphs, matched = example()
        glyphs[0]['size'] = 12
        matched[1] = ((200.,), .5, True)
        evidence = compare_note_values(measure, staff, 40, 250, matched, glyphs, [])
        self.assertEqual([item['musicxml_is_grace'] for item in evidence['grace_size_reviews']], [False, True])
        self.assertEqual(evidence['head_shapes_agree'], 2)
