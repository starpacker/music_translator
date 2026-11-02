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
    from core.omr.vector_tempos import discover_pdf_tempos, write_enriched_tempos

    report_path = work / f'{stem}.pdf-tempos.json'
    enriched = work / f'{stem}.pdf-tempos.musicxml'
    protected_paths = {mxl.resolve(), score_pdf.resolve()}
    while report_path.resolve() in protected_paths:
        report_path = report_path.with_name(report_path.stem + '.report.json')
    while enriched.resolve() in protected_paths:
        enriched = enriched.with_name(enriched.stem + '.derived.musicxml')
    try:
        candidates, decisions = discover_pdf_tempos(score_pdf, mxl)
        inserted, skipped = (write_enriched_tempos(mxl, enriched, candidates)
                             if candidates else ([], []))
        def digest(path):
            h = hashlib.sha256()
            with path.open('rb') as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b''):
                    h.update(chunk)
            return h.hexdigest()
        report_path.write_text(json.dumps({'source_pdf': str(score_pdf),
                                          'source_pdf_sha256': digest(score_pdf),
                                          'source_musicxml': str(mxl),
                                          'source_musicxml_sha256': digest(mxl),
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
        print(f'[jianpu] PDF 速度标记恢复失败，保留原始识谱结果继续：{exc}')
        return mxl


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

    work = args.work or (base.parent / 'work')
    work.mkdir(parents=True, exist_ok=True)

    use_gpu = {'auto': None, 'on': True, 'off': False}[args.gpu]
    mxl = args.mxl or omr_to_musicxml(score, work, use_gpu=use_gpu)
    title = args.title or score.stem
    jianpu_mxl = mxl
    if not args.no_vector_tempos and (args.mode in ('jianpu', 'both') or args.engine == 'strip'):
        jianpu_mxl = recover_pdf_tempos(score, mxl, work, base.name)

    if args.mode in ('jianpu', 'both'):
        musicxml_to_jianpu(jianpu_mxl, base.parent / f'{base.name}.jianpu.pdf', title)

    if args.mode in ('insitu', 'both'):
        out = base.parent / f'{base.name}.insitu.pdf'
        if args.engine == 'strip':
            ref = base.parent / f'{base.name}.jianpu.pdf'
            if not ref.is_file():
                musicxml_to_jianpu(jianpu_mxl, ref, title)
            insitu_strip(score, ref, out)
        else:
            insitu_stamp(score, mxl, out, title, expand=args.engine == 'spaced')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
