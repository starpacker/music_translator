#!/usr/bin/env python
"""Convert a directory of PDFs serially, retaining per-score logs and failures."""
from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path


def pack_zip(paths, destination):
    """Write ``paths`` into one zip archive, flat (file names only).

    Returns the list of files actually packed.  The archive is built through a
    temporary sibling first, so an interrupted run never leaves a truncated
    ``.zip`` that looks complete.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + '.part')
    packed = []
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            path = Path(path)
            if not path.is_file():
                continue
            archive.write(path, arcname=path.name)
            packed.append(path)
    temporary.replace(destination)
    return packed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='Directory containing PDF scores')
    parser.add_argument('-o', '--output', type=Path, default=Path('output'))
    parser.add_argument('--work', type=Path, default=None)
    parser.add_argument('--mode', choices=('both', 'jianpu', 'insitu'), default='both')
    parser.add_argument('--engine', choices=('spaced', 'strip', 'stamp'), default='spaced')
    parser.add_argument('--layout', choices=('single', 'wrap'), default='single')
    parser.add_argument('--pitch-mode', choices=('movable', 'fixed'), default='movable')
    parser.add_argument('--gpu', choices=('auto', 'on', 'off'), default='auto')
    parser.add_argument('--zip', dest='zip_path', type=Path, default=None,
                        metavar='PATH',
                        help='Also pack every generated insitu PDF into this zip '
                             'archive (e.g. output/insitu.zip).')
    args = parser.parse_args(argv)
    if not args.input.is_dir():
        parser.error(f'Input directory does not exist: {args.input}')
    scores = sorted(p for p in args.input.iterdir() if p.is_file() and p.suffix.lower() == '.pdf')
    if not scores:
        parser.error('No PDFs found')
    args.output.mkdir(parents=True, exist_ok=True)
    work = args.work or args.output / 'work'
    failures = []
    produced = []
    for i, score in enumerate(scores, 1):
        print(f'[{i}/{len(scores)}] {score.name}', flush=True)
        command = [sys.executable, '-u', str(Path(__file__).with_name('jianpu.py')),
                   str(score.resolve()), '-o', str((args.output / score.stem).resolve()),
                   '--work', str((work / score.stem).resolve()), '--mode', args.mode,
                   '--engine', args.engine, '--layout', args.layout,
                   '--pitch-mode', args.pitch_mode, '--gpu', args.gpu]
        log = args.output / f'{score.stem}.log'
        with log.open('w', encoding='utf-8') as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            failures.append(score.name)
            print(f'  FAILED: see {log}', flush=True)
        else:
            print(f'  OK: {args.output}', flush=True)
            insitu = args.output / f'{score.stem}.insitu.pdf'
            if insitu.is_file():
                produced.append(insitu)
    print(f'Completed: {len(scores)-len(failures)}/{len(scores)}')
    if failures:
        print('Failed: ' + ', '.join(failures))
    if args.zip_path is not None:
        packed = pack_zip(produced, args.zip_path)
        print(f'Packed {len(packed)} insitu PDF(s): {args.zip_path}')
        if not packed:
            print('  ! No insitu PDF was produced; the zip archive is empty.', flush=True)
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
