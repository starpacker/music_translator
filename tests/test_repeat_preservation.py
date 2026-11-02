"""Short repeat sections are valid even with very few notes."""
import unittest
import subprocess
import sys
import tempfile
from pathlib import Path
from music21 import bar, note, stream
from core.notation.jianpu.extract import _extract_part_repeat_barlines
from core.render.jianpu_runner import _inject_barlines_into_voice_block, _insert_repeat_bar_commands


class RepeatPreservationTest(unittest.TestCase):
    def test_single_note_repeat_is_retained(self):
        part = stream.Part()
        measure = stream.Measure(number=1)
        measure.leftBarline = bar.Repeat(direction='start')
        measure.rightBarline = bar.Repeat(direction='end')
        measure.append(note.Note('C4', quarterLength=4))
        part.append(measure)
        self.assertEqual(_extract_part_repeat_barlines(part),
                         {0: {'start': True, 'end': True}})
        block = _inject_barlines_into_voice_block('c1 | \\bar "|."',
                                                 {0: {'start': True, 'end': True}})
        self.assertIn('\\bar ".|:"', block)
        self.assertIn('\\bar ":|."', block)

    def test_repeats_on_adjacent_measures_are_retained(self):
        part = stream.Part()
        first = stream.Measure(number=1)
        first.leftBarline = bar.Repeat(direction='start')
        first.append(note.Rest(quarterLength=4))
        second = stream.Measure(number=2)
        second.rightBarline = bar.Repeat(direction='end')
        second.append(note.Note('D4', quarterLength=4))
        part.append([first, second])
        self.assertEqual(_extract_part_repeat_barlines(part),
                         {0: {'start': True, 'end': False}, 1: {'start': False, 'end': True}})

    def test_single_measure_repeat_is_inserted_into_generated_voice(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'line.txt'
            source.write_text('1=C\n4/4\n1 - - - |\n', encoding='utf-8')
            script = Path(__file__).resolve().parents[1] / 'scripts/jianpu-ly.py'
            ly = subprocess.run([sys.executable, str(script), str(source)],
                                check=True, capture_output=True, text=True).stdout
        rendered = _insert_repeat_bar_commands(ly, {0: {'start': True, 'end': True}})
        voice = rendered.split('% === BEGIN JIANPU STAFF ===', 1)[1].split('% === END JIANPU STAFF ===', 1)[0]
        self.assertIn('\\bar ".|:"', voice)
        self.assertIn('\\bar ":|."', voice)
