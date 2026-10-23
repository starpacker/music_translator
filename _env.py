"""Locate the SumisoraOMR checkout and the external tools this project needs.

Everything that used to be hard-coded to one machine lives here.  Resolution
order for the OMR library:

1. ``$SUMISORA_ROOT``
2. ``<repo>/vendor/SumisoraOMR``
3. a sibling directory next to this repo: ``../SumisoraOMR-main/SumisoraOMR-main``
   or ``../SumisoraOMR``

The jianpu renderer is invoked with the *same* interpreter that runs this
script, and LilyPond is looked up on ``PATH`` (``$LILYPOND`` overrides).
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent

_CANDIDATES = (
    HERE / 'vendor' / 'SumisoraOMR',
    HERE.parent / 'SumisoraOMR-main' / 'SumisoraOMR-main',
    HERE.parent / 'SumisoraOMR',
    HERE.parent / 'SumisoraOMR-main',
)


def sumisora_root(required: bool = True) -> Optional[Path]:
    """Return the SumisoraOMR checkout, or ``None`` when not found."""
    env = os.environ.get('SUMISORA_ROOT')
    candidates = ([Path(env).expanduser()] if env else []) + list(_CANDIDATES)
    for cand in candidates:
        if (cand / 'core' / 'notation' / 'jianpu' / '__init__.py').is_file():
            return cand.resolve()
    if required:
        raise RuntimeError(
            '找不到 SumisoraOMR。请设置环境变量 SUMISORA_ROOT 指向该仓库：\n'
            '  export SUMISORA_ROOT=/path/to/SumisoraOMR\n'
            '（或把仓库放在 <本仓库>/vendor/SumisoraOMR）'
        )
    return None


def ensure_sumisora() -> Path:
    """Put the OMR library on ``sys.path`` and return its root."""
    root = sumisora_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def jianpu_ly_script() -> Path:
    """The vendored ``jianpu-ly.py`` preprocessor shipped with SumisoraOMR."""
    path = ensure_sumisora() / 'scripts' / 'jianpu-ly.py'
    if not path.is_file():
        raise RuntimeError(f'缺少 jianpu-ly: {path}')
    return path


def python_exe() -> str:
    """Interpreter used for helper scripts (must be the one that has music21)."""
    return sys.executable


def lilypond_exe() -> str:
    exe = os.environ.get('LILYPOND') or shutil.which('lilypond')
    if not exe:
        raise RuntimeError('找不到 LilyPond。请安装 lilypond 2.22+ 或设置 $LILYPOND。')
    return exe
