# music_translator — Staff Notation → Jianpu (Chinese Numbered Notation)

A computer-vision pipeline that takes an image of **Western staff notation**
(specifically a piano Grand Staff) and produces the corresponding
**jianpu** (简谱, Chinese numbered musical notation) — useful for
musicians who read jianpu and want to play pieces that only exist as
Western sheet music.

> **Status:** the OpenCV prototype reaches **100% accuracy** on the test
> piece (Mozart *Turkish March*, K.331, Volodos arrangement —
> 20 bars / 111 events / 213 pitches all correct).

---

## Repository layout

```
music_translator/
├── prototype_cv/         ← main implementation (OpenCV-based pipeline)
│   ├── main.py             ← end-to-end driver
│   ├── pitch_detection.py  ← detect note heads and assign pitches from staff lines
│   ├── note_unit.py        ← Note datatype + duration/beam logic
│   ├── note_assignment.py  ← assign each note its measure / voice
│   ├── symbol_detection.py ← clefs, accidentals, rests, time-signature symbols
│   ├── stem_tracking.py    ← stem/beam tracking for grouped notes
│   ├── segmentation.py     ← measure / system segmentation
│   ├── staff_removal.py    ← remove staff lines so symbols can be found cleanly
│   ├── template_matching.py ← match small symbols against the template library
│   ├── jianpu_formatter.py ← lay out the result as jianpu text
│   ├── evaluate.py         ← compare against ground truth, report accuracy
│   ├── ground_truth.md     ← per-bar ground truth
│   ├── README.md           ← detailed pipeline-stage / algorithm walkthrough
│   └── test_duration_merge.py
├── template/             ← template images used by template_matching
│                            (clefs, time-signatures, rests, accidentals…)
├── input_page1.png       ← test input score (Mozart K.331, page 1)
├── ground_truth_2.md     ← extended ground truth (line-by-line jianpu)
├── 二胡.pdf              ← reference scores (erhu)
└── 曲笛.pdf              ← reference scores (Chinese flute)
```

## Quick start

```bash
cd prototype_cv

pip install opencv-python numpy Pillow

# Run on the default test image (../input_page1.png)
python main.py

# Or pass a custom image
python main.py path/to/score.png

# Run accuracy evaluation against the embedded ground truth
python evaluate.py
```

### Outputs

| File | Description |
|---|---|
| `output_jianpu.txt`           | Jianpu text, separated into treble / bass lines |
| `output_jianpu_on_staff.png`  | Original staff with jianpu annotations overlaid |
| `output_jianpu_clean.png`     | Clean rendered jianpu image |

## Pipeline at a glance

1. **Staff removal** — detect staff lines and erase them so symbols stand alone.
2. **Symbol detection** — clefs, accidentals (sharp/flat/natural), rests,
   time signatures, via template matching against `template/`.
3. **Note-head detection + pitch assignment** — find note heads, then map
   each one to a pitch using the staff position + active clef + key signature.
4. **Stem / beam tracking** — group flagged notes and recover their durations.
5. **Note assignment** — assign each note to its measure and voice.
6. **Jianpu formatting** — render the result as numbered notation, splitting
   into treble (high) and bass (low) lines.

See [`prototype_cv/README.md`](./prototype_cv/README.md) for the full
per-stage algorithm walkthrough, the jianpu notation conventions, the
accuracy report, and the known limitations / generalisation discussion.

## License

TBD.
