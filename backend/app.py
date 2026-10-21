"""SumisoraOMR web backend — PDF -> jianpu PDF, with real transposition.

Runs the project's OMR + jianpu pipeline inside the *sumisora* conda env and
exposes a tiny JSON/HTTP API for the GitHub Pages frontend.

Pipeline per request
--------------------
  1. OMR:   PDF -> MusicXML            (core.app.pipeline.process_single_input_to_jianpu,
                                         engine=HOMR by default for PDFs)
  2. Transpose (optional): MusicXML -> transposed MusicXML
                                         (core.notation.transposer.transpose_musicxml)
  3. Render: transposed MusicXML -> jianpu PDF
                                         (core.render.renderer.generate_jianpu_pdf_from_mxl)

The key selector is a REAL transposition: we detect the score's original key
from the OMR MusicXML, then shift every <pitch>/<fifths> so the requested tonic
becomes "1=".  "1=G" means the output reads in G.  Modes map to the same `to_key`
tonic plus a printed mode header handled downstream.

Endpoints
---------
GET  /api/health                 -> {"ok": true, "lilypond": bool, ...}
GET  /api/keys                   -> supported tonics + modes
POST /api/convert                -> multipart: file, tonic, mode, ...  => jianpu PDF

CORS is restricted to the configured ALLOWED_ORIGINS (your Pages domain).
"""
from __future__ import annotations

import os
import re
import shutil
import sys
import tempfile
import traceback
import uuid
import zipfile
from urllib.parse import quote
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

# ── make the SumisoraOMR repo importable ─────────────────────────────────────
SUMISORA_ROOT = Path(
    os.environ.get('SUMISORA_ROOT', '/liuyutian/SumisoraOMR-main/SumisoraOMR-main')
).resolve()
if str(SUMISORA_ROOT) not in sys.path:
    sys.path.insert(0, str(SUMISORA_ROOT))

from core.notation.transposer import (  # noqa: E402
    KEYS_BY_CIRCLE_OF_FIFTHS,
    detect_key_from_musicxml,
    transpose_musicxml,
)
from core.config import OMREngine  # noqa: E402
from core.app.pipeline import process_single_input_to_jianpu  # noqa: E402
from core.render.renderer import generate_jianpu_pdf_from_mxl  # noqa: E402

# ── the梆笛排版工具 in-situ aligner (vendored) ────────────────────────────────
# Pairs each visible jianpu cell with the corresponding engraved staff measure
# and lays the jianpu strip directly beneath every staff system.  This is the
# approach that superseded the earlier hand-rolled in-situ renderer.
sys.path.insert(0, str(Path(__file__).resolve().parent / 'vendor'))
from score_align import Config as InSituConfig, build_output as build_insitu  # noqa: E402

# ── supported tonics + modes ─────────────────────────────────────────────────
# 12 major tonics, spelled as jianpu editors expect (b for flat).
DEFAULT_TONICS = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
TONICS = os.environ.get('JIANPU_TONICS', '').split(',') if os.environ.get('JIANPU_TONICS') else DEFAULT_TONICS

# Modes. "major"/"minor" are the ones the pipeline can *print* today (jianpu
# header "1=X" / "6=X").  The rest are transposable but the printed header is
# still the parallel-major tonic until the mode-header renderer lands; they are
# listed so the UI can offer them, gated by MODES_ENABLED.
MODES = [
    {'id': 'major',      'label': '大调 (Major)',           'enabled': True},
    {'id': 'minor',      'label': '自然小调 (Minor)',        'enabled': True},
    {'id': 'harmonic',   'label': '和声小调 (Harmonic min)', 'enabled': False},
    {'id': 'melodic',    'label': '旋律小调 (Melodic min)',  'enabled': False},
    {'id': 'dorian',     'label': '多利亚 (Dorian)',         'enabled': False},
    {'id': 'phrygian',   'label': '弗里吉亚 (Phrygian)',     'enabled': False},
    {'id': 'lydian',     'label': '利底亚 (Lydian)',         'enabled': False},
    {'id': 'mixolydian', 'label': '混合利底亚 (Mixolydian)', 'enabled': False},
    {'id': 'locrian',    'label': '洛克里亚 (Locrian)',      'enabled': False},
]

ALLOWED_ORIGINS = [
    o.strip()
    for o in os.environ.get(
        'ALLOWED_ORIGINS',
        'http://localhost:5173,http://127.0.0.1:5173',
    ).split(',')
    if o.strip()
]

app = FastAPI(title='SumisoraOMR Jianpu API', version='1.0.0')
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=['GET', 'POST', 'OPTIONS'],
    allow_headers=['*'],
    expose_headers=['Content-Disposition', 'X-Detected-Key', 'X-Output-Key'],
)


def _lilypond_ok() -> bool:
    return shutil.which('lilypond') is not None


@app.get('/api/health')
def health() -> dict:
    return {
        'ok': True,
        'lilypond': _lilypond_ok(),
        'sumisora_root': str(SUMISORA_ROOT),
        'python': sys.version.split()[0],
    }


@app.get('/api/keys')
def keys() -> dict:
    return {'tonics': TONICS, 'modes': MODES}


@app.post('/api/convert')
async def convert(
    file: UploadFile = File(...),
    tonic: str = Form('C'),
    mode: str = Form('major'),
    engine: str = Form('homr'),
    transpose: bool = Form(True),
    format: str = Form('jianpu'),
) -> Response:
    """Upload a PDF (or image) and receive a jianpu PDF.

    - ``tonic``    target tonic for "1=" (e.g. ``G``)
    - ``mode``     mode id (see /api/keys)
    - ``transpose`` when True, real pitch transposition to the target key;
                    when False, the score keeps its own key and ``tonic`` is ignored.
    - ``format``   ``jianpu``  -> the standalone numbered-notation PDF
                   ``insitu``  -> numbered notation aligned under each staff
                   ``both``    -> a zip containing both
    """
    if file.filename is None:
        raise HTTPException(status_code=400, detail='缺少文件名')
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {'.pdf', '.png', '.jpg', '.jpeg'}:
        raise HTTPException(status_code=400, detail=f'不支持的文件类型: {suffix}')
    if not _lilypond_ok():
        raise HTTPException(status_code=503, detail='服务器缺少 LilyPond，无法渲染简谱')

    tonic = (tonic or 'C').strip()
    if tonic not in TONICS:
        raise HTTPException(status_code=400, detail=f'不支持的调号: {tonic}')
    mode_ids = {m['id'] for m in MODES}
    if mode not in mode_ids:
        raise HTTPException(status_code=400, detail=f'不支持的调式: {mode}')
    mode_enabled = next(m['enabled'] for m in MODES if m['id'] == mode)
    if not mode_enabled:
        raise HTTPException(
            status_code=400,
            detail=f'调式 {mode} 暂未支持（需要调式头渲染）。请先用大调/小调。',
        )

    job = Path(tempfile.gettempdir()) / f'jianpu-{uuid.uuid4().hex}'
    job.mkdir(parents=True, exist_ok=True)
    src = job / f'input{suffix}'
    try:
        src.write_bytes(await file.read())

        engine_enum = {
            'homr': OMREngine.HOMR,
            'audiveris': OMREngine.AUDIVERIS,
            'auto': OMREngine.AUTO,
        }.get(engine, OMREngine.HOMR)

        # 1) OMR -> MusicXML (also writes a first-pass jianpu PDF we discard)
        omr_pdf = job / 'omr_pass.pdf'
        ok = process_single_input_to_jianpu(
            src, job, omr_pdf, None, engine=engine_enum,
        )
        if not ok:
            raise HTTPException(status_code=422, detail='OMR 识别失败，请尝试更清晰的乐谱')

        mxl = next(iter(job.glob('*.musicxml')), None) or next(iter(job.glob('*.mxl')), None)
        if mxl is None:
            raise HTTPException(status_code=500, detail='OMR 未输出 MusicXML')

        detected = detect_key_from_musicxml(mxl)

        # 2) real transposition so "1=<tonic>" reads correctly
        final_mxl = mxl
        out_key = detected
        if transpose and tonic.upper() != detected.upper():
            shifted = job / 'transposed.musicxml'
            transpose_musicxml(mxl, shifted, from_key=detected, to_key=tonic,
                               direction='closest')
            final_mxl = shifted
            out_key = tonic

        # 3) render final jianpu PDF
        out_pdf = job / 'jianpu.pdf'
        render_ok = generate_jianpu_pdf_from_mxl(
            final_mxl, out_pdf, job, preferred_title=src.stem,
        )
        if not render_ok or not out_pdf.exists():
            raise HTTPException(status_code=500, detail='简谱渲染失败')

        headers = {
            'X-Detected-Key': detected,
            'X-Output-Key': out_key,
        }
        # Use the *uploaded* name, not the temp file's "input", so the download
        # is recognisable.  Keep it ASCII-safe: a raw UTF-8 filename in the
        # header would arrive mojibake'd, so send an ASCII fallback plus the
        # RFC 5987 UTF-8 form that real browsers prefer.
        orig = Path(file.filename).stem or 'jianpu'
        ascii_stem = re.sub(r'[^A-Za-z0-9._-]+', '_', orig).strip('._-') or 'jianpu'

        def _disposition(ext: str) -> str:
            quoted = quote(f'{orig}.{ext}')
            return f"attachment; filename=\"{ascii_stem}.{ext}\"; filename*=UTF-8''{quoted}"

        if format == 'jianpu':
            headers['Content-Disposition'] = _disposition('jianpu.pdf')
            return Response(content=out_pdf.read_bytes(),
                            media_type='application/pdf', headers=headers)

        # 4) in-situ: place each jianpu cell beneath its staff measure.
        #     Needs the *original* score PDF, which is exactly what the user
        #     uploaded, so transcription errors never move the notation.
        if suffix != '.pdf':
            raise HTTPException(
                status_code=400,
                detail='原位版需要 PDF 输入（图片没有五线谱矢量和分格）。',
            )
        insitu_pdf = job / 'insitu.pdf'
        try:
            build_insitu(src, out_pdf, insitu_pdf, InSituConfig())
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f'原位排版失败: {exc}') from exc
        if not insitu_pdf.exists():
            raise HTTPException(status_code=500, detail='原位排版未生成文件')

        if format == 'insitu':
            headers['Content-Disposition'] = _disposition('insitu.pdf')
            return Response(content=insitu_pdf.read_bytes(),
                            media_type='application/pdf', headers=headers)

        if format != 'both':
            raise HTTPException(status_code=400, detail=f'不支持的输出格式: {format}')

        bundle = job / f'{ascii_stem}.jianpu.zip'
        with zipfile.ZipFile(bundle, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.write(out_pdf, f'{orig}.jianpu.pdf')
            zf.write(insitu_pdf, f'{orig}.insitu.pdf')
        headers['Content-Disposition'] = _disposition('jianpu.zip')
        return Response(content=bundle.read_bytes(),
                        media_type='application/zip', headers=headers)

    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f'内部错误: {exc}') from exc
    finally:
        shutil.rmtree(job, ignore_errors=True)


@app.exception_handler(Exception)
async def _unhandled(_request, exc):  # pragma: no cover
    return JSONResponse(status_code=500, content={'detail': str(exc)})
