"""
run_tests.py — regression harness: CV pipeline vs. vector-PDF ground truth.

For every page of every example PDF (二胡.pdf / 曲笛.pdf) this

  1. rasterises the page (cached, see pdf_tools),
  2. runs the CV pipeline (main.main) on the raster,
  3. extracts the exact ground truth from the PDF vectors (pdf_truth),
  4. compares measure by measure and reports measure / event / pitch
     accuracy plus a readable diff.

Usage:
    python run_tests.py                 # all pages of all pdfs
    python run_tests.py 二胡.pdf        # one pdf
    python run_tests.py 二胡.pdf 4      # one page (1-based)
    python run_tests.py --show          # print full outputs side by side
"""
import io
import os
import re
import sys
import contextlib
import difflib

import pdf_tools
import pdf_truth
from evaluate import parse_events, extract_notes_from_event

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_PDFS = ['二胡.pdf', '曲笛.pdf']


def _read_cv_measures(jianpu_path):
    """Parse output/jianpu.txt into {line_index: [measure strings]}."""
    with open(jianpu_path, encoding='utf-8') as f:
        text = f.read()
    lines = {}
    cur = None
    for raw in text.split('\n'):
        m = re.match(r'---\s*第\s*(\d+)\s*行\s*---', raw)
        if m:
            cur = int(m.group(1))
            lines[cur] = []
            continue
        if raw.strip().startswith('|') and cur is not None:
            lines[cur].append(raw.strip())
    if not lines:  # grand-staff format fallback
        for raw in text.split('\n'):
            if raw.startswith('|'):
                lines.setdefault(1, []).append(raw.strip())
    return lines


def _split_measures(line_str):
    return [p.strip() for p in line_str.split('|') if p.strip()]


def _norm(measure_str):
    return ' '.join(_split_measures(measure_str))


def _pitches(measure_str):
    out = []
    for ev in parse_events(measure_str):
        notes, _dur = extract_notes_from_event(ev)
        out.extend(notes)
    return out


def compare(truth_measures, cv_measures, verbose=False):
    """truth_measures / cv_measures: list of measure strings (no bars)."""
    n_meas = ok_meas = n_ev = ok_ev = n_pi = ok_pi = 0
    diffs = []
    sm = difflib.SequenceMatcher(None, truth_measures, cv_measures, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            for k in range(i2 - i1):
                t, c = truth_measures[i1 + k], cv_measures[j1 + k]
                n_meas += 1
                ok_meas += 1
                te, ce = parse_events(t), parse_events(c)
                n_ev += max(len(te), len(ce))
                ok_ev += min(len(te), len(ce))
                tp, cp = _pitches(t), _pitches(c)
                n_pi += max(len(tp), len(cp))
                ok_pi += min(len(tp), len(cp))
        else:
            for k in range(max(i2 - i1, j2 - j1)):
                t = truth_measures[i1 + k] if i1 + k < i2 else '<missing>'
                c = cv_measures[j1 + k] if j1 + k < j2 else '<missing>'
                n_meas += 1
                te, ce = parse_events(t), parse_events(c)
                n_ev += max(len(te), len(ce))
                ok_ev += sum(1 for a, b in zip(te, ce) if a == b)
                tp, cp = _pitches(t), _pitches(c)
                n_pi += max(len(tp), len(cp))
                ok_pi += sum(1 for a, b in zip(tp, cp) if a == b)
                diffs.append((i1 + k + 1, t, c))
            if verbose:
                pass
    return {'measures': (ok_meas, n_meas), 'events': (ok_ev, n_ev),
            'pitches': (ok_pi, n_pi), 'diffs': diffs}


def run_one(pdf_path, pno, offsets, show=False, out_root=None):
    pages = pdf_tools.render_pdf(pdf_path)
    png = pages[pno - 1]
    truth = pdf_truth.extract_page(pdf_path, pno - 1, offsets=offsets)
    if truth is None:
        return None
    t_measures = []
    for measures in pdf_truth.page_to_measures(truth):
        t_measures.append([' '.join(m) for m in measures])

    import main as cv_main
    outdir = os.path.join(out_root or os.path.join(HERE, 'output', 'tests'),
                          os.path.splitext(os.path.basename(pdf_path))[0],
                          f'p{pno}')
    cv_main.set_output_dir(outdir)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cv_main.main(png, bpm_override=False)
    cv_lines = _read_cv_measures(os.path.join(outdir, 'jianpu.txt'))

    result = {'page': pno, 'systems': []}
    n_sys = max(len(t_measures), len(cv_lines))
    for i in range(n_sys):
        t = t_measures[i] if i < len(t_measures) else []
        c = []
        for line in cv_lines.get(i + 1, []):
            c.extend(_split_measures(line))
        # truth keeps empty measures; cv may emit '|0 0|' placeholders – keep both
        res = compare(t, c, verbose=show)
        result['systems'].append(res)
        if show:
            print(f'   --- system {i + 1}')
            print('     TRUTH:', ' | '.join(t))
            print('     CV   :', ' | '.join(c))
    return result


def _pct(pair):
    ok, n = pair
    return 100.0 * ok / n if n else 100.0


def main(argv):
    show = '--show' in argv
    argv = [a for a in argv if not a.startswith('--')]
    pdfs = [a for a in argv if a.endswith('.pdf')] or DEFAULT_PDFS
    pages_want = [int(a) for a in argv if a.isdigit()]

    tot = {'measures': [0, 0], 'events': [0, 0], 'pitches': [0, 0]}
    for rel in pdfs:
        path = rel if os.path.isabs(rel) else os.path.join(ROOT, rel)
        if not os.path.exists(path):
            print(f'[skip] {rel} not found')
            continue
        offsets = pdf_truth._glyph_offsets(path)
        npages = pdf_tools.pdf_page_count(path)
        pages = pages_want or list(range(1, npages + 1))
        print(f'\n{"=" * 70}\n  {rel}  ({npages} pages)\n{"=" * 70}')
        for pno in pages:
            if pno > npages:
                continue
            res = run_one(path, pno, offsets, show=show)
            if not res:
                print(f'  page {pno}: no systems found')
                continue
            line = f'  page {pno}: '
            parts = []
            for key in ('measures', 'events', 'pitches'):
                ok = sum(s[key][0] for s in res['systems'])
                n = sum(s[key][1] for s in res['systems'])
                tot[key][0] += ok
                tot[key][1] += n
                parts.append(f'{key[:4]} {ok}/{n} ({_pct((ok, n)):.0f}%)')
            print(line + '  '.join(parts))
            for si, s in enumerate(res['systems']):
                if s['diffs']:
                    for mi, t, c in s['diffs']:
                        print(f'      sys{si + 1} M{mi}')
                        print(f'        TRUTH: {t}')
                        print(f'        CV   : {c}')
    print(f'\n{"=" * 70}\n  TOTAL: measures {tot["measures"][0]}/{tot["measures"][1]} '
          f'({_pct(tuple(tot["measures"])):.1f}%)   '
          f'events {tot["events"][0]}/{tot["events"][1]} ({_pct(tuple(tot["events"])):.1f}%)   '
          f'pitches {tot["pitches"][0]}/{tot["pitches"][1]} ({_pct(tuple(tot["pitches"])):.1f}%)')


if __name__ == '__main__':
    main(sys.argv[1:])
