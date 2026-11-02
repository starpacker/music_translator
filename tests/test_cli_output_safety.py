"""CLI outputs use this run's inputs and cannot replace them."""
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import jianpu


class CliOutputSafetyTest(unittest.TestCase):
    def test_strip_insitu_regenerates_existing_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'score.pdf'
            mxl = root / 'input.musicxml'
            source.write_bytes(b'pdf input')
            mxl.write_bytes(b'xml input')
            reference = root / 'out.jianpu.pdf'
            reference.write_text('old score')
            def render(_, output, title):
                output.write_text('current score')
                return output
            with mock.patch.object(jianpu, 'musicxml_to_jianpu', side_effect=render) as render_call, \
                 mock.patch.object(jianpu, 'insitu_strip') as strip:
                result = jianpu.main([str(source), '--mxl', str(mxl), '--mode', 'insitu', '--engine', 'strip',
                                      '--no-vector-tempos', '--no-vector-dynamics', '--no-vector-wedges',
                                      '-o', str(root / 'out')])
            self.assertEqual(result, 0)
            self.assertEqual(render_call.call_count, 1)
            self.assertEqual(reference.read_text(), 'current score')
            self.assertEqual(strip.call_args.args[1], reference)

    def test_both_mode_renders_reference_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, mxl = root / 'score.pdf', root / 'input.musicxml'
            source.write_bytes(b'pdf')
            mxl.write_bytes(b'xml')
            with mock.patch.object(jianpu, 'musicxml_to_jianpu') as render, \
                 mock.patch.object(jianpu, 'insitu_strip'):
                self.assertEqual(jianpu.main([str(source), '--mxl', str(mxl), '--mode', 'both', '--engine', 'strip',
                                              '--no-vector-tempos', '--no-vector-dynamics', '--no-vector-wedges',
                                              '-o', str(root / 'out')]), 0)
            self.assertEqual(render.call_count, 1)

    def test_output_collision_rejected_before_any_generation(self):
        for mode, input_name in [('insitu', 'out.insitu.pdf'), ('jianpu', 'out.jianpu.pdf')]:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source = root / input_name
                source.write_bytes(b'original')
                with mock.patch.object(jianpu, 'omr_to_musicxml') as omr, self.assertRaises(SystemExit):
                    jianpu.main([str(source), '--mode', mode, '-o', str(root / 'out')])
                omr.assert_not_called()
                self.assertEqual(source.read_bytes(), b'original')


if __name__ == '__main__':
    unittest.main()
