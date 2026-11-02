"""The event audit detects a lost note in an otherwise valid score."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from music21 import meter, note, stream
from scripts.audit_jianpu_events import audit
from core.notation.jianpu import extract_jianpu_measures


class EventAuditTest(unittest.TestCase):
    def test_reports_missing_extracted_pitch(self):
        score = stream.Score()
        part = stream.Part()
        m = stream.Measure(number=1)
        m.insert(0, meter.TimeSignature('4/4'))
        m.insert(0, note.Note('C4', quarterLength=2))
        m.insert(2, note.Note('D4', quarterLength=2))
        part.append(m)
        score.insert(0, part)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'score.musicxml'
            score.write('musicxml', fp=path)
            self.assertEqual(audit(path)['mismatch_count'], 0)
            def omit_note(*args, **kwargs):
                measures, ts = extract_jianpu_measures(*args, **kwargs)
                measures[0] = measures[0][:-1]
                return measures, ts
            with patch('scripts.audit_jianpu_events.extract_jianpu_measures', side_effect=omit_note):
                report = audit(path)
            self.assertEqual(report['mismatch_count'], 1)
            self.assertEqual(report['mismatches'][0]['measure_number'], '1')
