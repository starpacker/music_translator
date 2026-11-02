"""Cache invalidation must follow source content, not just its name."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from jianpu import omr_to_musicxml


class OMRCacheTest(unittest.TestCase):
    def test_partial_result_is_never_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'score.pdf'
            source.write_bytes(b'input')
            work = root / 'work'
            work.mkdir()
            calls = []
            def recognize(src, out, use_gpu_inference=None):
                calls.append(1)
                (out / 'score.musicxml').write_bytes(b'<score-partwise/>')
                (out / '.partial-omr').write_text('incomplete')
                return out
            with patch('homr_shim.install'), patch('core.omr.homr_runner.run_homr_batch', side_effect=recognize):
                omr_to_musicxml(source, work)
                omr_to_musicxml(source, work)
            self.assertEqual(len(calls), 2)
            self.assertFalse((work / 'score.omr-cache.json').exists())

    def test_content_change_with_same_name_forces_new_omr(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'score.pdf'
            source.write_bytes(b'first input')
            work = root / 'work'
            work.mkdir()
            calls = []
            def recognize(src, out, use_gpu_inference=None):
                calls.append(src.read_bytes())
                (out / 'score.musicxml').write_bytes(b'<score-partwise/>')
                return out
            with patch('homr_shim.install'), patch('core.omr.homr_runner.run_homr_batch', side_effect=recognize):
                omr_to_musicxml(source, work)
                omr_to_musicxml(source, work)
                self.assertEqual(len(calls), 1)
                source.write_bytes(b'other input')
                omr_to_musicxml(source, work)
                self.assertEqual(calls, [b'first input', b'other input'])
