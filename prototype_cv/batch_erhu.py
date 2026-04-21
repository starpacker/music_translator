"""
batch_erhu.py — Translate all pages of 二胡.pdf and combine output.
"""
import subprocess
import sys
import os
import shutil
import datetime

def main():
    base_dir = r'C:\Users\30670\Desktop\music_translator'
    cv_dir = os.path.join(base_dir, 'prototype_cv')

    # Output collection
    all_jianpu = []
    all_confidence = []
    cumulative_line = 0  # global line counter across pages
    last_bpm = None  # carry detected time signature across pages

    for page in range(1, 9):
        img_path = os.path.join(base_dir, 'input', f'erhu_p{page}.png')
        if not os.path.exists(img_path):
            print(f"[SKIP] Page {page}: {img_path} not found")
            continue

        print(f"\n{'='*60}")
        print(f"  Processing page {page}/8: {img_path}")
        print(f"{'='*60}")

        cmd = [sys.executable, 'main.py', img_path]
        if last_bpm is not None:
            cmd += ['--bpm', str(last_bpm)]

        result = subprocess.run(
            cmd,
            cwd=cv_dir,
            capture_output=True,
            text=True,
            encoding='utf-8',
            timeout=300
        )

        if result.returncode != 0:
            print(f"[ERROR] Page {page} failed:")
            print(result.stderr[-500:] if result.stderr else "(no stderr)")
            continue

        # Extract detected time signature for carry-forward
        import re as _re
        for line in result.stdout.split('\n'):
            m = _re.search(r'beats_per_measure=(\d+(?:\.\d+)?)', line)
            if m:
                last_bpm = float(m.group(1))

        # Read output files
        jianpu_path = os.path.join(cv_dir, 'output', 'jianpu.txt')
        conf_path = os.path.join(cv_dir, 'output', 'confidence.txt')

        if os.path.exists(jianpu_path):
            with open(jianpu_path, encoding='utf-8') as f:
                text = f.read()
            all_jianpu.append(f"\n{'#'*60}\n# 第 {page} 页\n{'#'*60}\n")
            # Skip the header lines (first 3 lines)
            lines = text.split('\n')
            body_start = 0
            for i, l in enumerate(lines):
                if l.startswith('---') or l.startswith('|'):
                    body_start = i
                    break
            # Renumber line headers to be globally cumulative
            import re
            renumbered = []
            for l in lines[body_start:]:
                m = re.match(r'---\s*第\s*(\d+)\s*行\s*---', l)
                if m:
                    cumulative_line += 1
                    renumbered.append(f'--- 第{cumulative_line}行 ---')
                else:
                    renumbered.append(l)
            all_jianpu.append('\n'.join(renumbered))

        if os.path.exists(conf_path):
            with open(conf_path, encoding='utf-8') as f:
                text = f.read()
            all_confidence.append(f"\n{'#'*60}\n# 第 {page} 页\n{'#'*60}\n")
            all_confidence.append(text)

        # Save per-page visual outputs
        for fname in ['jianpu_on_staff.png', 'jianpu_visual.png']:
            src = os.path.join(cv_dir, 'output', fname)
            if os.path.exists(src):
                dst = os.path.join(base_dir, f'erhu_p{page}_{fname}')
                shutil.copy2(src, dst)

        print(f"  Page {page} done.")

    # Write combined output
    combined_jianpu = os.path.join(base_dir, 'erhu_full_jianpu.txt')
    with open(combined_jianpu, 'w', encoding='utf-8') as f:
        f.write("二胡 — 全文简谱翻译\n")
        f.write("=" * 60 + "\n")
        f.write(''.join(all_jianpu))
    print(f"\nSaved: {combined_jianpu}")

    combined_conf = os.path.join(base_dir, 'erhu_full_confidence.txt')
    with open(combined_conf, 'w', encoding='utf-8') as f:
        f.write("二胡 — 全文置信度报告\n")
        f.write("=" * 60 + "\n")
        f.write(''.join(all_confidence))
    print(f"Saved: {combined_conf}")

if __name__ == '__main__':
    main()
