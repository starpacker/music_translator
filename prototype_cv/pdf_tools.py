"""
pdf_tools.py — PDF input support for the staff→jianpu pipeline.

The example scores shipped with this repository (二胡.pdf / 曲笛.pdf) are
vector PDFs.  The computer-vision pipeline works on raster images, so the
first step for a PDF input is to rasterise every page at a fixed DPI.

Two helpers are provided:

* :func:`render_pdf`  — rasterise a PDF into PNG pages (cached on disk).
* :func:`pdf_page_count` — number of pages without rasterising.

Rendering is deterministic, so a cache directory keyed by
``<pdf-stem>_<dpi>`` lets repeated test runs skip the (slow) rasterisation.
The cache lives under ``output/pdf_cache/`` which is gitignored.
"""
import os
import hashlib

try:
    import pymupdf  # PyMuPDF (``fitz`` is the deprecated alias)
except ImportError:  # pragma: no cover - optional dependency
    pymupdf = None

# The PNG test images in ../input were produced at ~288 DPI; keep the same
# resolution so results stay comparable with the historical ground truth.
DEFAULT_DPI = 288


def _cache_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    d = os.path.join(here, "output", "pdf_cache")
    os.makedirs(d, exist_ok=True)
    return d


def pdf_page_count(pdf_path):
    """Return the number of pages of ``pdf_path``."""
    if pymupdf is None:
        raise RuntimeError("PyMuPDF is required for PDF input (pip install pymupdf)")
    doc = pymupdf.open(pdf_path)
    n = doc.page_count
    doc.close()
    return n


def render_pdf(pdf_path, dpi=DEFAULT_DPI, out_dir=None, force=False):
    """Rasterise every page of ``pdf_path`` to PNG.

    Returns the list of PNG paths (one per page, 1-based suffix ``_p1`` …).
    Results are cached: re-calling with the same pdf/dpi reuses the files
    unless ``force=True``.
    """
    if pymupdf is None:
        raise RuntimeError("PyMuPDF is required for PDF input (pip install pymupdf)")

    stem = os.path.splitext(os.path.basename(pdf_path))[0]
    if out_dir is None:
        out_dir = os.path.join(_cache_dir(), f"{stem}_{dpi}")
    os.makedirs(out_dir, exist_ok=True)

    doc = pymupdf.open(pdf_path)
    paths = []
    for i, page in enumerate(doc):
        out = os.path.join(out_dir, f"{stem}_p{i + 1}.png")
        if force or not os.path.exists(out):
            pix = page.get_pixmap(dpi=dpi)
            pix.save(out)
        paths.append(out)
    doc.close()
    return paths
