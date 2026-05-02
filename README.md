# music_translator — Staff Notation → Jianpu (Chinese Numbered Notation)

A computer-vision pipeline that takes an image of **Western staff notation**
and produces the corresponding **jianpu** (简谱, Chinese numbered musical notation).

Supports both **piano grand staff** (treble + bass) and **single-staff**
instruments (erhu, flute, etc.).

> **Accuracy** (evaluated test pieces):
> - Piano (Mozart K.331): **100%** measures / events / pitches (20 bars)
> - Erhu (二胡 8 pages): **93%** measures, **97%** events, **99%** pitches
> - Qudi (曲笛 5 pages): **81%** measures, **86%** events, **90%** pitches

---

## Repository layout

```
music_translator/
├── prototype_cv/          ← main implementation (OpenCV-based pipeline)
│   ├── main.py              ← end-to-end driver
│   ├── config.py            ← all hyperparameters (dy-relative)
│   ├── staff_removal.py     ← staff line extraction & erasure
│   ├── pitch_detection.py   ← staff system detection + pitch mapping
│   ├── template_matching.py ← notehead detection (morphology + templates)
│   ├── symbol_detection.py  ← barlines, accidentals, rests, time/key sigs, slur arcs
│   ├── note_assignment.py   ← treble/bass assignment
│   ├── stem_tracking.py     ← stem direction & position
│   ├── note_unit.py         ← chord grouping + beam counting + duration estimation
│   ├── jianpu_formatter.py  ← text output formatting (key sig, accidental persistence)
│   ├── jianpu_visual.py     ← PIL-rendered visual jianpu output
│   ├── confidence.py        ← detection confidence scoring
│   ├── evaluate.py          ← accuracy evaluation vs ground truth
│   ├── batch_erhu.py        ← batch processor for multi-page erhu scores
│   ├── output/              ← generated output files (gitignored)
│   ├── ground_truth.md      ← piano ground truth
│   ├── ground_truth_2.md    ← erhu ground truth
│   └── README.md            ← detailed algorithm walkthrough (Chinese)
├── template/              ← symbol templates (sharps, flats, rests, clefs, digits…)
└── input/                 ← test input images
    ├── piano_p1.png         ← Mozart K.331
    ├── erhu_p[1-8].png      ← erhu (8 pages)
    └── qudi_p[1-5].png      ← qudi flute (5 pages)
```

## Quick start

```bash
cd prototype_cv
pip install opencv-python numpy Pillow

# Run on the default test image (piano)
python main.py                          # uses ../input/piano_p1.png

# Run on a custom image
python main.py ../input/erhu_p1.png

# Override time signature (e.g. 3/4 = 3.0 beats)
python main.py score.png --bpm 3.0

# Run accuracy evaluation
python evaluate.py
python evaluate.py ground_truth_2.md output/jianpu.txt

# Batch process erhu (8 pages)
python batch_erhu.py
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
