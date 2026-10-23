#!/usr/bin/env python
"""Convert a directory of PDFs serially, retaining per-score logs and failures."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='Directory containing PDF scores')
    parser.add_argument('-o', '--output', type=Path, default=Path('output'))
    parser.add_argument('--work', type=Path, default=None)
    parser.add_argument('--mode', choices=('both', 'jianpu', 'insitu'), default='both')
    parser.add_argument('--engine', choices=('strip', 'stamp'), default='strip')
    parser.add_argument('--gpu', choices=('auto', 'on', 'off'), default='auto')
    args = parser.parse_args(argv)
    if not args.input.is_dir():
        parser.error(f'Input directory does not exist: {args.input}')
    scores = sorted(p for p in args.input.iterdir() if p.is_file() and p.suffix.lower() == '.pdf')
    if not scores:
        parser.error('No PDFs found')
    args.output.mkdir(parents=True, exist_ok=True)
    work = args.work or args.output / 'work'
    failures = []
    for i, score in enumerate(scores, 1):
        print(f'[{i}/{len(scores)}] {score.name}', flush=True)
        command = [sys.executable, '-u', str(Path(__file__).with_name('jianpu.py')),
                   str(score.resolve()), '-o', str((args.output / score.stem).resolve()),
                   '--work', str((work / score.stem).resolve()), '--mode', args.mode,
                   '--engine', args.engine, '--gpu', args.gpu]
        log = args.output / f'{score.stem}.log'
        with log.open('w', encoding='utf-8') as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            failures.append(score.name)
            print(f'  FAILED: see {log}', flush=True)
        else:
            print(f'  OK: {args.output}', flush=True)
    print(f'Completed: {len(scores)-len(failures)}/{len(scores)}')
    if failures:
        print('Failed: ' + ', '.join(failures))
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
