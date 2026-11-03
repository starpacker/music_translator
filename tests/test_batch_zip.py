"""The batch driver can pack its generated insitu PDFs into one archive."""
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import batch


class PackZipTest(unittest.TestCase):
    def test_only_existing_files_are_packed_flat(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / '第一首.insitu.pdf'
            second = root / '第二首.insitu.pdf'
            first.write_bytes(b'%PDF-1')
            second.write_bytes(b'%PDF-2')
            destination = root / 'pack' / 'insitu.zip'

            packed = batch.pack_zip([first, root / 'missing.insitu.pdf', second],
                                    destination)

            self.assertEqual([p.name for p in packed], [first.name, second.name])
            with zipfile.ZipFile(destination) as archive:
                self.assertEqual(sorted(archive.namelist()),
                                 sorted([first.name, second.name]))
            self.assertEqual(list(destination.parent.iterdir()), [destination])

    def test_batch_zip_option_collects_insitu_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / 'input'
            source_dir.mkdir()
            for name in ('one.pdf', 'two.pdf'):
                (source_dir / name).write_bytes(b'%PDF')
            output = root / 'output'

            def fake_run(command, stdout=None, stderr=None, **kwargs):
                stem = Path(command[command.index('-o') + 1]).stem
                stdout.write('ok\n')
                (output / f'{stem}.insitu.pdf').write_bytes(b'%PDF-insitu')
                (output / f'{stem}.jianpu.pdf').write_bytes(b'%PDF-jianpu')
                return type('Result', (), {'returncode': 0})()

            with patch('batch.subprocess.run', side_effect=fake_run):
                code = batch.main([str(source_dir), '-o', str(output),
                                   '--zip', str(output / 'insitu.zip')])

            self.assertEqual(code, 0)
            with zipfile.ZipFile(output / 'insitu.zip') as archive:
                self.assertEqual(sorted(archive.namelist()),
                                 ['one.insitu.pdf', 'two.insitu.pdf'])

    def test_zip_skips_scores_that_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / 'input'
            source_dir.mkdir()
            (source_dir / 'good.pdf').write_bytes(b'%PDF')
            (source_dir / 'bad.pdf').write_bytes(b'%PDF')
            output = root / 'output'

            def fake_run(command, stdout=None, stderr=None, **kwargs):
                stem = Path(command[command.index('-o') + 1]).stem
                stdout.write('run\n')
                if stem == 'bad':
                    return type('Result', (), {'returncode': 1})()
                (output / f'{stem}.insitu.pdf').write_bytes(b'%PDF-insitu')
                return type('Result', (), {'returncode': 0})()

            with patch('batch.subprocess.run', side_effect=fake_run):
                code = batch.main([str(source_dir), '-o', str(output),
                                   '--zip', str(output / 'insitu.zip')])

            self.assertEqual(code, 1)
            with zipfile.ZipFile(output / 'insitu.zip') as archive:
                self.assertEqual(archive.namelist(), ['good.insitu.pdf'])


if __name__ == '__main__':
    unittest.main()
