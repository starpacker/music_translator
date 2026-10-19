# music_translator — Staff Notation → Jianpu (Chinese Numbered Notation)

A computer-vision pipeline that takes an image of **Western staff notation**
and produces the corresponding **jianpu** (简谱, Chinese numbered musical notation).

Supports both **piano grand staff** (treble + bass) and **single-staff**
instruments (erhu, flute, etc.).

> **Accuracy** (evaluated test pieces):
> - Piano (Mozart K.331, hand-written ground truth): **90%** measures / **95%** events / **99%** pitches (20 bars)
> - Erhu + Qudi, **all 13 PDF pages** vs. exact vector-PDF ground truth
>   (`run_tests.py`): **38%** measures / **50%** events / **57%** pitches —
>   this whole-corpus number is the honest baseline the CV pipeline is now
>   being iterated against (see `prototype_cv/README.md` for per-page detail).
>
> **Movable do**: the jianpu tonic is selectable with `--key` (`1=C`, `1=G`,
> `1=bB`, … or `auto` = derive from the detected key signature).

---

## Repository layout

```
music_translator/
├── prototype_cv/          ← main implementation (OpenCV-based pipeline)
│   ├── main.py              ← end-to-end driver (PNG or PDF input, --key)
│   ├── config.py            ← all hyperparameters (dy-relative)
│   ├── staff_removal.py     ← staff line extraction & erasure
│   ├── pitch_detection.py   ← staff system detection + pitch mapping
│   ├── template_matching.py ← notehead detection (morphology + templates,
│   │                           incl. cross/x noteheads)
│   ├── symbol_detection.py  ← barlines, accidentals, rests, time/key sigs, slur arcs
│   ├── note_assignment.py   ← treble/bass assignment
│   ├── stem_tracking.py     ← stem direction & position
│   ├── note_unit.py         ← chord grouping + beam counting + duration estimation
│   ├── jianpu_formatter.py  ← text output formatting (key sig, accidental persistence)
│   ├── jianpu_transpose.py  ← movable-do renumbering (1=X) of formatted output
│   ├── jianpu_visual.py     ← PIL-rendered visual jianpu output
│   ├── confidence.py        ← detection confidence scoring
│   ├── evaluate.py          ← accuracy evaluation vs hand-written ground truth
│   ├── pdf_tools.py         ← rasterise PDF examples (cached) for the CV pipeline
│   ├── pdf_truth.py         ← exact ground truth from the PDF vector content
│   ├── run_tests.py         ← regression harness: CV output vs vector truth
│   ├── test_jianpu_transpose.py ← unit tests for the tonic renumbering
│   ├── batch_erhu.py        ← batch processor for multi-page scores (portable)
│   ├── output/              ← generated output files (gitignored)
│   ├── ground_truth.md      ← piano ground truth
│   ├── ground_truth_2.md    ← erhu ground truth
│   └── README.md            ← detailed algorithm walkthrough (Chinese)
├── template/              ← symbol templates (sharps, flats, rests, clefs, digits…)
├── 二胡.pdf / 曲笛.pdf      ← vector-PDF examples (test corpus + truth source)
└── input/                 ← test input images (renders of the PDFs + piano)
    ├── piano_p1.png         ← Mozart K.331
    ├── erhu_p[1-8].png      ← erhu (8 pages)
    └── qudi_p[1-5].png      ← qudi flute (5 pages)
```

## Quick start

```bash
cd prototype_cv
pip install opencv-python numpy Pillow pymupdf

# Run on the default test image (piano)
python main.py                          # uses ../input/piano_p1.png

# Run on a custom image
python main.py ../input/erhu_p1.png

# Run directly on a PDF (every page is rasterised and translated)
python main.py ../二胡.pdf
python main.py ../曲笛.pdf --pages 1,3 --dpi 300

# Choose the jianpu tonic (movable do): auto | C | G | D | bB | #F ...
python main.py ../input/erhu_p5.png --key auto     # 3 sharps detected → 1=A
python main.py ../input/piano_p1.png --key C       # force 1=C

# Override time signature (e.g. 3/4 = 3.0 beats)
python main.py score.png --bpm 3.0

# Run accuracy evaluation vs hand-written ground truth
python evaluate.py
python evaluate.py ground_truth_2.md output/jianpu.txt

# Regression harness: CV pipeline vs exact vector-PDF truth, all pages
python run_tests.py                    # both PDFs
python run_tests.py 二胡.pdf 4         # single page, with diffs

# Unit tests (tonic renumbering)
python test_jianpu_transpose.py

# Batch process a whole score into one combined file
python batch_erhu.py                   # 二胡.pdf
python batch_erhu.py ../曲笛.pdf
```

### Outputs (saved to `prototype_cv/output/`)

| File | Description |
|---|---|
| `jianpu.txt`           | Jianpu text (treble/bass lines for piano, line-by-line for single staff) |
| `jianpu_on_staff.png`  | Original staff with jianpu annotations overlaid |
| `jianpu_visual.png`    | Visual jianpu rendering (red digits + rhythm marks) |
| `jianpu_clean.png`     | Clean rendered jianpu image (piano only) |
| `confidence.txt`       | Per-measure confidence scores with `[!]` flags |

## Pipeline

1. **Staff removal** — detect & erase staff lines, producing a clean symbol image
2. **System detection** — find staff systems, detect layout (grand staff vs single staff)
3. **Slur/tie masking** — detect arc contours and subtract from symbol image
4. **Time & key signature** — template-match digits for time sig; detect accidental clusters for key sig
5. **Barline detection** — dual strategy (template + vertical morphology), adaptive threshold
6. **Notehead detection** — morphological opening + template matching + NMS
7. **Accidental detection** — multi-scale template matching for sharps/flats/naturals
8. **Rest detection** — quarter/eighth rest templates with false-positive filtering
9. **Stem tracking** — find stem direction, tip position, and beam count
10. **Note units & measures** — chord grouping, duration estimation (beams + proportional spacing), measure segmentation
11. **Formatting** — jianpu text with accidental persistence, key signature application, chord brackets

All pixel-level parameters scale with `dy` (staff-line spacing). No hardcoded pixel values.

## Acknowledgments

This project is inspired by and references [xuxiran/translate-staff-to-simple-musical-notation](https://github.com/xuxiran/translate-staff-to-simple-musical-notation). We would like to express our gratitude to the original author for their pioneering work in staff-to-jianpu translation.

## License

TBD.
