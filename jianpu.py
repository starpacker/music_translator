#!/usr/bin/env python
"""五线谱 → 简谱：两种成品，一条命令。

    五线谱 PDF ──OMR──► MusicXML ──┬──► 整页简谱 PDF      (mode=jianpu)
                                   ├──► 原位简谱 PDF      (mode=insitu)
                                   └──► 两者都要          (mode=both)

用法
----
    python jianpu.py 曲笛.pdf                       # 默认 mode=both
    python jianpu.py 曲笛.pdf --mode insitu         # 只要原位简谱
    python jianpu.py 曲笛.pdf --mode jianpu -o 曲笛简谱.pdf
    python jianpu.py 曲笛.pdf --mxl 已有.musicxml   # 跳过 OMR，直接排版

OMR 结果会缓存为 ``<work>/<曲名>.musicxml``，第二次运行不再识别。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import importlib.metadata
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Optional

from _env import ensure_sumisora

HERE = Path(__file__).resolve().parent


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


# ── 步骤 1：OMR ──────────────────────────────────────────────────────────────

def omr_to_musicxml(score_pdf: Path, work: Path, use_gpu: Optional[bool] = None) -> Path:
    """PDF → MusicXML（HOMR 引擎）。已存在则直接复用。"""
    cached = work / f'{score_pdf.stem}.musicxml'
    metadata = work / f'{score_pdf.stem}.omr-cache.json'
    digest = hashlib.sha256()
    with score_pdf.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    try:
        version = importlib.metadata.version('homr')
    except importlib.metadata.PackageNotFoundError:
        version = None
    pipeline = hashlib.sha256()
    root = ensure_sumisora()
    pipeline.update((HERE / 'homr_shim.py').read_bytes())
    for name in ('core/config.py', 'core/utils.py', 'core/omr/homr_runner.py',
                 'core/omr/multirest_collapse.py', 'core/omr/vector_whole_chords.py',
                 'core/image/image_preprocess.py', 'core/image/sr_upscale.py',
                 'core/notation/tie_reconstruction.py'):
        pipeline.update((root / name).read_bytes())
    expected = {'source_sha256': digest.hexdigest(), 'gpu': use_gpu,
                'pipeline_sha256': pipeline.hexdigest(),
                'homr_version': version, 'cache_schema': 1}
    try:
        cached_metadata = json.loads(metadata.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        cached_metadata = None
    if cached.is_file() and cached.stat().st_size > 0 and cached_metadata == expected:
        print(f'[omr] 复用已有结果 {cached}')
        return cached
    # Do not leave a manifest authorizing an old cache during an interrupted
    # refresh of a different source with the same stem.
    metadata.unlink(missing_ok=True)

    import homr_shim
    homr_shim.install()
    from core.omr.homr_runner import run_homr_batch          # noqa: PLC0415
    from core.utils import find_first_musicxml_file          # noqa: PLC0415

    out_dir = work / f'{score_pdf.stem}.omr'
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / '.partial-omr').unlink(missing_ok=True)
    print(f'[omr] 识别 {score_pdf.name} …')
    res = run_homr_batch(score_pdf, out_dir, use_gpu_inference=use_gpu)
    if not res:
        raise RuntimeError('OMR 失败，请检查 homr 是否安装、模型权重能否下载')
    mxl = find_first_musicxml_file(Path(res), score_pdf.stem)
    if mxl is None:
        raise RuntimeError(f'OMR 没有产生 MusicXML（{out_dir}）')
    shutil.copy2(mxl, cached)
    ties_ok = True
    try:
        from core.notation.tie_reconstruction import reconstruct_ties_in_musicxml
        ties = reconstruct_ties_in_musicxml(cached, paired_slurs_only=True)
        if ties:
            print(f'[omr] 从成对同音曲线恢复 {ties} 对延音线（原始识谱文件仍保留在 work 中）')
    except Exception as exc:
        ties_ok = False
        print(f'[omr] 延音线恢复失败，保留原始识谱结果继续：{exc}')
    if (Path(res) / '.partial-omr').exists() or not ties_ok:
        print('[omr] 警告：识谱或后处理不完整，本次继续生成，但不作为完整结果缓存。')
    else:
        metadata.write_text(json.dumps(expected, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[omr] -> {cached}')
    return cached


# ── 步骤 2：整页简谱 ────────────────────────────────────────────────────────

def recover_pdf_tempos(score_pdf: Path, mxl: Path, work: Path, stem: str) -> Path:
    """Use verifiable PDF tempo marks in a derived XML, leaving the input intact."""
    report_path = work / f'{stem}.pdf-tempos.json'
    enriched = work / f'{stem}.pdf-tempos.musicxml'
    protected_paths = {mxl.resolve(), score_pdf.resolve()}
    while report_path.resolve() in protected_paths:
        report_path = report_path.with_name(report_path.stem + '.report.json')
    while enriched.resolve() in protected_paths:
        enriched = enriched.with_name(enriched.stem + '.derived.musicxml')
    try:
        from core.omr.vector_tempos import discover_pdf_tempos, write_enriched_tempos
        candidates, decisions = discover_pdf_tempos(score_pdf, mxl)
        inserted, skipped = (write_enriched_tempos(mxl, enriched, candidates)
                             if candidates else ([], []))
        report_path.write_text(json.dumps({'source_pdf': str(score_pdf),
                                          'source_pdf_sha256': _sha256_file(score_pdf),
                                          'source_musicxml': str(mxl),
                                          'source_musicxml_sha256': _sha256_file(mxl),
                                          'inserted': inserted, 'skipped': skipped,
                                          'decisions': decisions}, ensure_ascii=False, indent=2),
                               encoding='utf-8')
        if inserted:
            print(f'[jianpu] 从 PDF 矢量标记恢复 {len(inserted)} 个速度记号（推断结果，请核对；'
                  f'原始 MusicXML 未修改，详情：{report_path}）')
        unresolved = sum(1 for item in decisions if item.get('reason') != 'verified_candidate')
        if unresolved:
            print(f'[jianpu] {unresolved} 个 PDF 速度标记无法可靠定位，请参阅 {report_path} 校对。')
        if skipped:
            print(f'[jianpu] {len(skipped)} 个候选未写入（例如该小节已有速度），请参阅 {report_path}。')
        return enriched if inserted else mxl
    except Exception as exc:
        try:
            report_path.write_text(json.dumps({'source_pdf': str(score_pdf), 'source_musicxml': str(mxl),
                                               'inserted': [], 'error': str(exc)}, ensure_ascii=False, indent=2),
                                   encoding='utf-8')
        except Exception:
            pass
        print(f'[jianpu] PDF 速度标记恢复失败，保留原始识谱结果继续：{exc}')
        return mxl


def recover_pdf_dynamics(score_pdf: Path, source_mxl: Path, input_mxl: Path, work: Path, stem: str) -> Path:
    """Add font- and notehead-verified dynamics to another derived XML."""
    report_path = work / f'{stem}.pdf-dynamics.json'
    enriched = work / f'{stem}.pdf-dynamics.musicxml'
    protected = {score_pdf.resolve(), source_mxl.resolve(), input_mxl.resolve()}
    while report_path.resolve() in protected:
        report_path = report_path.with_name(report_path.stem + '.report.json')
    while enriched.resolve() in protected:
        enriched = enriched.with_name(enriched.stem + '.derived.musicxml')
    try:
        from core.omr.vector_dynamics import discover_pdf_dynamics, write_enriched_dynamics
        candidates, decisions = discover_pdf_dynamics(score_pdf, source_mxl)
        inserted, skipped = (write_enriched_dynamics(input_mxl, enriched, candidates)
                             if candidates else ([], []))
        report_path.write_text(json.dumps({'source_pdf': str(score_pdf),
                                           'source_pdf_sha256': _sha256_file(score_pdf),
                                           'source_musicxml': str(source_mxl),
                                           'source_musicxml_sha256': _sha256_file(source_mxl),
                                           'input_musicxml': str(input_mxl),
                                           'input_musicxml_sha256': _sha256_file(input_mxl),
                                           'inserted': inserted, 'skipped': skipped,
                                           'decisions': decisions}, ensure_ascii=False, indent=2),
                               encoding='utf-8')
        if inserted:
            print(f'[jianpu] 从 PDF 字体和音符位置恢复 {len(inserted)} 个力度记号'
                  f'（推断结果，请核对；原始 MusicXML 未修改，详情：{report_path}）')
        unresolved = sum(1 for item in decisions if item.get('reason') != 'verified_candidate')
        if unresolved:
            print(f'[jianpu] {unresolved} 个 PDF 力度记号无法可靠定位，请参阅 {report_path} 校对。')
        if skipped:
            print(f'[jianpu] {len(skipped)} 个力度候选未写入，请参阅 {report_path}。')
        return enriched if inserted else input_mxl
    except Exception as exc:
        try:
            report_path.write_text(json.dumps({'source_pdf': str(score_pdf), 'source_musicxml': str(source_mxl),
                                               'input_musicxml': str(input_mxl), 'inserted': [],
                                               'error': str(exc)}, ensure_ascii=False, indent=2), encoding='utf-8')
        except Exception:
            pass
        print(f'[jianpu] PDF 力度记号恢复失败，保留已有识谱结果继续：{exc}')
        return input_mxl


def recover_pdf_wedges(score_pdf: Path, source_mxl: Path, input_mxl: Path, work: Path, stem: str) -> Path:
    """Recover isolated vector hairpins only where both musical boundaries are verified."""
    report_path = work / f'{stem}.pdf-wedges.json'
    enriched = work / f'{stem}.pdf-wedges.musicxml'
    protected = {score_pdf.resolve(), source_mxl.resolve(), input_mxl.resolve()}
    while report_path.resolve() in protected:
        report_path = report_path.with_name(report_path.stem + '.report.json')
    while enriched.resolve() in protected:
        enriched = enriched.with_name(enriched.stem + '.derived.musicxml')
    try:
        from core.omr.vector_wedges import discover_pdf_wedges, write_enriched_wedges
        candidates, decisions = discover_pdf_wedges(score_pdf, source_mxl)
        inserted, skipped = (write_enriched_wedges(input_mxl, enriched, candidates)
                             if candidates else ([], []))
        report_path.write_text(json.dumps({'source_pdf': str(score_pdf),
                                           'source_pdf_sha256': _sha256_file(score_pdf),
                                           'source_musicxml': str(source_mxl),
                                           'source_musicxml_sha256': _sha256_file(source_mxl),
                                           'input_musicxml': str(input_mxl),
                                           'input_musicxml_sha256': _sha256_file(input_mxl),
                                           'inserted': inserted, 'skipped': skipped,
                                           'decisions': decisions}, ensure_ascii=False, indent=2),
                               encoding='utf-8')
        unresolved = sum(item.get('reason') != 'verified_candidate' for item in decisions)
        if inserted:
            print(f'[jianpu] 从 PDF 推断恢复 {len(inserted)} 条渐强渐弱线，请对照原谱；详情：{report_path}')
        if unresolved or skipped:
            print(f'[jianpu] {unresolved} 处线段待核对，{len(skipped)} 条候选未写入；详情：{report_path}')
        return enriched if inserted else input_mxl
    except Exception as exc:
        try:
            report_path.write_text(json.dumps({'source_pdf': str(score_pdf), 'source_musicxml': str(source_mxl),
                                               'input_musicxml': str(input_mxl), 'inserted': [],
                                               'error': str(exc)}, ensure_ascii=False, indent=2), encoding='utf-8')
        except Exception:
            pass
        print(f'[jianpu] PDF 渐强渐弱线恢复失败，保留已有识谱结果继续：{exc}')
        return input_mxl


def write_pdf_notehead_audit(score_pdf: Path, mxl: Path, work: Path, stem: str) -> Path:
    """Record limited vector evidence without modifying either source."""
    report = work / f'{stem}.pdf-noteheads.json'
    protected = {score_pdf.resolve(), mxl.resolve()}
    while report.resolve() in protected:
        report = report.with_name(report.stem + '.report.json')

    def write(data):
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=report.parent,
                                             prefix='.notehead-audit-', suffix='.json', delete=False) as f:
                temporary = Path(f.name)
                json.dump(data, f, ensure_ascii=False, indent=2)
            temporary.replace(report)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    try:
        write({'status': 'in_progress', 'source_pdf': str(score_pdf), 'source_musicxml': str(mxl)})
        from core.omr.vector_notehead_audit import audit_pdf_noteheads
        result = audit_pdf_noteheads(score_pdf, mxl)
        result.update({'source_pdf': str(score_pdf), 'source_pdf_sha256': _sha256_file(score_pdf),
                       'source_musicxml': str(mxl), 'source_musicxml_sha256': _sha256_file(mxl)})
        write(result)
        if result['status'] == 'unsupported':
            print(f'[audit] PDF 符头核对不支持此谱面，请人工核对：{report}')
        else:
            summary = result['summary']
            print(f"[audit] PDF 符头位置支持 {summary.get('supported_head_positions_agree', 0)} 个小节，"
                  f"发现 {summary.get('printed_multirest_count_detected', 0)} 处带数字的多小节休止；"
                  f'只核对部分证据，不代表识谱准确。详情：{report}')
            unresolved = sum(n for status, n in summary.items()
                             if status not in ('supported_head_positions_agree', 'printed_multirest_count_detected'))
            if unresolved or result['unmapped_musicxml_indices'] or result['duplicate_musicxml_indices']:
                print(f'[audit] {unresolved} 个谱面位置缺少充分符头证据，或存在未映射/重复小节，请参阅报告。')
    except Exception as exc:
        try:
            write({'status': 'error', 'source_pdf': str(score_pdf), 'source_musicxml': str(mxl),
                   'error': str(exc)})
        except Exception:
            pass
        print(f'[audit] PDF 符头核对失败，不影响生成，请人工核对：{exc}')
    return report


def musicxml_to_jianpu(mxl: Path, out_pdf: Path, title: Optional[str] = None) -> Path:
    """MusicXML → 整页简谱 PDF（music21 → jianpu-ly → LilyPond）。"""
    ensure_sumisora()
    from core.render.renderer import generate_jianpu_pdf_from_mxl   # noqa: PLC0415

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix='jianpu_'))
    try:
        ok = generate_jianpu_pdf_from_mxl(
            mxl, out_pdf, tmp, preferred_title=title or mxl.stem)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if not ok or not out_pdf.is_file():
        raise RuntimeError('简谱渲染失败（检查 LilyPond 与 jianpu-ly）')
    print(f'[jianpu] -> {out_pdf}  ({out_pdf.stat().st_size/1024:.0f} KB)')
    return out_pdf


# ── 步骤 3：原位简谱 ────────────────────────────────────────────────────────

def insitu_stamp(score_pdf: Path, mxl: Path, out_pdf: Path,
                 title: Optional[str] = None, expand: bool = False) -> Path:
    """矢量贴字：把每行简谱贴到对应谱表正下方，页码和原谱完全一致。"""
    sys.path.insert(0, str(HERE / 'insitu'))
    import render_insitu                                     # noqa: PLC0415

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    render_insitu.build(score_pdf, mxl, out_pdf, title=title or score_pdf.stem, expand=expand)
    print(f'[insitu] -> {out_pdf}  ({out_pdf.stat().st_size/1024:.0f} KB)')
    return out_pdf


def insitu_strip(score_pdf: Path, jianpu_pdf: Path, out_pdf: Path) -> Path:
    """插条式：在每个谱表系统后插入简谱条，页面相应加长。"""
    sys.path.insert(0, str(HERE / 'insitu' / 'vendor'))
    from score_align import Config as InSituConfig, build_output   # noqa: PLC0415

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    western, cells = build_output(score_pdf, jianpu_pdf, out_pdf, InSituConfig())
    print(f'[insitu-strip] {western} 个谱表 / {cells} 个简谱格 -> {out_pdf} '
          f'({out_pdf.stat().st_size/1024:.0f} KB)')
    return out_pdf


# ── CLI ─────────────────────────────────────────────────────────────────────

def main(argv: list[str]) -> int:
    # Some hosted environments export TMPDIR pointing to a removed directory.
    # Python silently falls back, but LilyPond uses the invalid value directly.
    configured_tmp = os.environ.get('TMPDIR')
    if configured_tmp and not Path(configured_tmp).is_dir():
        os.environ['TMPDIR'] = tempfile.gettempdir()
    ap = argparse.ArgumentParser(
        description='五线谱 PDF → 简谱：整页简谱（jianpu）或原位简谱（insitu）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    ap.add_argument('score', type=Path, help='源五线谱 PDF')
    ap.add_argument('-o', '--output', type=Path, default=None,
                    help='输出文件（mode=both 时作为前缀，默认与输入同名）')
    ap.add_argument('--mode', choices=('jianpu', 'insitu', 'both'), default='both')
    ap.add_argument('--engine', choices=('spaced', 'stamp', 'strip'), default='spaced',
                    help='原位简谱：spaced=矢量插入空白（默认）；stamp=矢量贴字；'
                         'strip=插条并加长页面')
    ap.add_argument('--mxl', type=Path, default=None, help='已有的 MusicXML（跳过 OMR）')
    ap.add_argument('--no-vector-tempos', action='store_true',
                    help='不从 PDF 矢量文字推断速度记号')
    ap.add_argument('--no-vector-dynamics', action='store_true',
                    help='不从 PDF 字体和音符位置推断力度记号')
    ap.add_argument('--no-vector-wedges', action='store_true',
                    help='不从 PDF 矢量线条推断渐强渐弱线')
    ap.add_argument('--no-pdf-audit', action='store_true',
                    help='不生成 PDF 符头位置和多小节休止数字核对报告')
    ap.add_argument('--work', type=Path, default=None,
                    help='OMR 中间结果目录（默认 <输出目录>/work）')
    ap.add_argument('--title', default=None, help='标题（默认取输入文件名）')
    ap.add_argument('--gpu', choices=('auto', 'on', 'off'), default='auto',
                    help='OMR 是否用 GPU（默认 auto）')
    args = ap.parse_args(argv)

    score: Path = args.score
    if not score.is_file():
        ap.error(f'找不到输入文件: {score}')

    if args.output is None:
        stem = args.output = score.with_suffix('')
        base = score.parent / score.stem
    else:
        base = args.output.with_suffix('') if args.output.suffix else args.output

    # Generated files must never replace either of the user's inputs, even
    # when an output prefix happens to end in .jianpu or .insitu.
    generated = []
    if args.mode in ('jianpu', 'both') or (args.mode == 'insitu' and args.engine == 'strip'):
        generated.append(base.parent / f'{base.name}.jianpu.pdf')
    if args.mode in ('insitu', 'both'):
        generated.append(base.parent / f'{base.name}.insitu.pdf')
    protected = {score.resolve()}
    if args.mxl is not None:
        protected.add(args.mxl.resolve())
    if any(path.resolve() in protected for path in generated):
        ap.error('输出路径与输入文件相同，请选择其他输出前缀')

    work = args.work or (base.parent / 'work')
    work.mkdir(parents=True, exist_ok=True)

    use_gpu = {'auto': None, 'on': True, 'off': False}[args.gpu]
    mxl = args.mxl or omr_to_musicxml(score, work, use_gpu=use_gpu)
    title = args.title or score.stem
    if not args.no_pdf_audit:
        write_pdf_notehead_audit(score, mxl, work, base.name)
    jianpu_mxl = mxl
    if not args.no_vector_tempos and (args.mode in ('jianpu', 'both') or args.engine == 'strip'):
        jianpu_mxl = recover_pdf_tempos(score, mxl, work, base.name)
    if not args.no_vector_dynamics and (args.mode in ('jianpu', 'both') or args.engine == 'strip'):
        jianpu_mxl = recover_pdf_dynamics(score, mxl, jianpu_mxl, work, base.name)
    if not args.no_vector_wedges and (args.mode in ('jianpu', 'both') or args.engine == 'strip'):
        jianpu_mxl = recover_pdf_wedges(score, mxl, jianpu_mxl, work, base.name)

    if args.mode in ('jianpu', 'both'):
        musicxml_to_jianpu(jianpu_mxl, base.parent / f'{base.name}.jianpu.pdf', title)

    if args.mode in ('insitu', 'both'):
        out = base.parent / f'{base.name}.insitu.pdf'
        if args.engine == 'strip':
            ref = base.parent / f'{base.name}.jianpu.pdf'
            # In insitu-only mode an existing PDF may belong to a previous
            # MusicXML or source score; it is not a validated cache.
            if args.mode == 'insitu':
                musicxml_to_jianpu(jianpu_mxl, ref, title)
            insitu_strip(score, ref, out)
        else:
            insitu_stamp(score, mxl, out, title, expand=args.engine == 'spaced')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
