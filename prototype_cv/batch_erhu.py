"""
batch_erhu.py — translate every page of a score PDF and combine the output.

Defaults to 二胡.pdf (8 pages); pass another PDF path as first argument.
The detected time signature is carried across pages, exactly like a
performer reading on.

Usage:
    python batch_erhu.py [path/to/score.pdf] [--key X] [--dpi N]
"""
import io
import os
import re
import sys
import contextlib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

import main as cv_main
import pdf_tools


def main(pdf_path=None, key='auto', dpi=288):
    if pdf_path is None:
        pdf_path = os.path.join(ROOT, '二胡.pdf')
    stem = os.path.splitext(os.path.basename(pdf_path))[0]
    cv_main.KEY_OPTION = key

    pages = pdf_tools.render_pdf(pdf_path, dpi=dpi)
    all_jianpu, all_conf = [], []
    cumulative_line = 0
    last_bpm = None

    out_root = os.path.join(HERE, 'output', f'{stem}_batch')
    for page, png in enumerate(pages, 1):
        print(f"\n{'=' * 60}\n  Processing page {page}/{len(pages)}\n{'=' * 60}")
        outdir = os.path.join(out_root, f'p{page}')
        cv_main.set_output_dir(outdir)
        if last_bpm is not None:
            from config import CFG
            CFG.duration.beats_per_measure = last_bpm
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cv_main.main(png, bpm_override=last_bpm is not None)
        log = buf.getvalue()
        m = re.search(r'beats_per_measure=(\d+(?:\.\d+)?)', log)
        if m:
            last_bpm = float(m.group(1))

        jianpu_path = os.path.join(outdir, 'jianpu.txt')
        if os.path.exists(jianpu_path):
            with open(jianpu_path, encoding='utf-8') as f:
                lines = f.read().split('\n')
            body_start = next((i for i, l in enumerate(lines)
                               if l.startswith('---') or l.startswith('|')), 0)
            renumbered = []
            for l in lines[body_start:]:
                mm = re.match(r'---\s*第\s*(\d+)\s*行\s*---', l)
                if mm:
                    cumulative_line += 1
                    renumbered.append(f'--- 第{cumulative_line}行 ---')
                else:
                    renumbered.append(l)
            all_jianpu.append(f"\n{'#' * 60}\n# 第 {page} 页\n{'#' * 60}\n")
            all_jianpu.append('\n'.join(renumbered))

        conf_path = os.path.join(outdir, 'confidence.txt')
        if os.path.exists(conf_path):
            with open(conf_path, encoding='utf-8') as f:
                all_conf.append(f"\n{'#' * 60}\n# 第 {page} 页\n{'#' * 60}\n")
                all_conf.append(f.read())

    combined = os.path.join(out_root, f'{stem}_full_jianpu.txt')
    os.makedirs(out_root, exist_ok=True)
    with open(combined, 'w', encoding='utf-8') as f:
        f.write(f"{stem} — 全文简谱翻译\n" + '=' * 60 + '\n')
        f.write(''.join(all_jianpu))
    print(f"\nSaved: {combined}")
    with open(os.path.join(out_root, f'{stem}_full_confidence.txt'), 'w',
              encoding='utf-8') as f:
        f.write(f"{stem} — 全文置信度报告\n" + '=' * 60 + '\n')
        f.write(''.join(all_conf))


if __name__ == '__main__':
    args = [a for a in sys.argv[1:]]
    pdf = args[0] if args and not args[0].startswith('--') else None
    key = 'auto'
    if '--key' in args:
        key = args[args.index('--key') + 1]
    main(pdf, key=key)
