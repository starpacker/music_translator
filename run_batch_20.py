#!/usr/bin/env python3
"""Batch-regenerate all 20 吹管 scores (OMR -> reconcile -> per-measure blank render).

Each 曲子 subdirectory is its own batch.py invocation, mirroring the existing
output layout output_fixed/吹管/<曲子>/.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/liuyutian/music_translator_refresh')
GROUPS = ['小河淌水', '渔歌', '雨夜花', '黑走马']

PYTHON = '/tmp/music_translator_env/bin/python'


def main() -> int:
    log_path = Path('/tmp/batch_20_run.log')
    log = open(log_path, 'w', encoding='utf-8')

    def emit(msg: str) -> None:
        line = f'[{time.strftime("%F %T")}] {msg}'
        print(line, flush=True)
        log.write(line + '\n')
        log.flush()

    emit('start batch (20 scores)')
    failed_groups = 0
    total_start = time.time()
    for group in GROUPS:
        in_dir = ROOT / 'input' / '吹管' / group
        out_dir = ROOT / 'output_fixed' / '吹管' / group
        emit(f'=== group {group} ===')
        cmd = [
            PYTHON, '-u', str(ROOT / 'batch.py'), str(in_dir),
            '-o', str(out_dir),
            '--mode', 'both', '--layout', 'single',
            '--pitch-mode', 'fixed', '--fixed-tonic', 'C', '--gpu', 'off',
        ]
        g0 = time.time()
        proc = subprocess.run(cmd, cwd=str(ROOT), stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True)
        log.write(proc.stdout)
        log.flush()
        emit(f'group {group} finished rc={proc.returncode} ({time.time()-g0:.0f}s)')
        if proc.returncode != 0:
            failed_groups += 1

    emit(f'DONE. groups_failed={failed_groups} total={time.time()-total_start:.0f}s')
    emit('--- generated insitu PDFs ---')
    for p in sorted((ROOT / 'output_fixed' / '吹管').glob('*/*.insitu.pdf')):
        emit(f'  {p.relative_to(ROOT)}  ({p.stat().st_size} bytes)')
    log.close()
    return int(bool(failed_groups))


if __name__ == '__main__':
    raise SystemExit(main())
