"""
confidence.py — per-note / per-event / per-measure confidence scoring.

The pipeline produces literal output. This module annotates it with
confidence so humans can focus review on low-confidence events rather
than line-by-line verification.

Signals used:
- notehead template match score (raw OpenCV match value, post-threshold)
- duration source tag (beam / proportional / rescued / pickup / tuplet / whole)
- measure beat-sum health (does the measure total match beats_per_measure?)

NOT a replacement for evaluation against ground truth. It only scores what
was detected — it cannot flag notes the detector missed entirely.
"""
from jianpu_formatter import format_note_unit, format_rest


LOW_CONF_THRESHOLD = 0.75

DURATION_SOURCE_CONF = {
    'beam':                          1.00,
    'whole':                         1.00,
    'tuplet':                        0.95,
    'pickup':                        0.90,
    'single_fill':                   0.85,
    'tuplet_leftover':               0.82,
    'proportional':                  0.80,
    'proportional_dotted_decomposed':0.80,
    'proportional_uniform':          0.65,
    'beam_rescue':                   0.78,
    'unknown':                       0.85,
}


def _notehead_conf(score):
    """Linear rescale of OpenCV template match score to confidence [0,1].

    Detection threshold is ~0.55, clean prints typically score ~0.85-0.95.
    Map 0.55 → 0.60, 0.90+ → 1.00.
    """
    if score <= 0.55:
        return 0.60
    if score >= 0.90:
        return 1.00
    return 0.60 + (score - 0.55) * (0.40 / 0.35)


def score_note_entry(note):
    """Return (confidence, reasons_list) for a single NoteEntry dict."""
    reasons = []
    nh_score = note.get('score', 0.70)
    nh_conf = _notehead_conf(nh_score)
    if nh_score < 0.70:
        reasons.append(f"notehead match={nh_score:.2f}")
    return nh_conf, reasons


def score_event(event):
    """Return (confidence, reasons_list) for a measure event (note_unit or rest)."""
    if event['type'] == 'rest':
        # Rests are post-adjusted to make measures sum correctly; they
        # carry little independent evidence. Default to high confidence
        # unless an explicit low-confidence source was set.
        dsource = event.get('duration_source', None)
        if dsource is None:
            return 0.95, []
        dur_conf = DURATION_SOURCE_CONF.get(dsource, 0.85)
        return dur_conf, ([] if dur_conf >= 0.9 else [f"rest dur: {dsource}"])

    unit = event['unit']
    notes = unit.get('notes', [])
    if not notes:
        return 0.5, ["empty unit"]

    note_confs = []
    all_reasons = []
    for n in notes:
        c, r = score_note_entry(n)
        note_confs.append(c)
        all_reasons.extend(r)
    note_conf = min(note_confs)

    dsource = event.get('duration_source', 'unknown')
    dur_conf = DURATION_SOURCE_CONF.get(dsource, 0.70)
    if dur_conf < 0.90:
        all_reasons.append(f"duration source: {dsource}")

    # Chord member score spread — if members differ by >0.15 in confidence,
    # the chord grouping may have swept in a stray note.
    if len(note_confs) >= 2 and (max(note_confs) - min(note_confs)) > 0.15:
        all_reasons.append("chord member score spread")

    overall = min(note_conf, dur_conf)
    return overall, all_reasons


def score_measure(measure, beats_per_measure=2.0):
    """Return per-event scores + reasons, plus measure-level signals.

    Returns
    -------
    (event_scores, event_reasons, measure_conf_adjustment, measure_reasons)
    measure_conf_adjustment is a multiplier (<= 1.0) applied to every event
    score in the measure when the measure itself is suspect.
    """
    event_scores = []
    event_reasons = []
    for e in measure:
        c, r = score_event(e)
        event_scores.append(c)
        event_reasons.append(r)

    # Beat-sum sanity
    total = 0.0
    for e in measure:
        if e['type'] == 'rest':
            total += e.get('duration', 1.0)
        elif e['type'] == 'note_unit':
            total += e['unit'].get('duration', 1.0)
    measure_reasons = []
    measure_mul = 1.0
    beat_err = abs(total - beats_per_measure)
    if beat_err > 0.1 and measure:
        measure_reasons.append(
            f"beat sum {total:.2f} != {beats_per_measure}")
        measure_mul = 0.70

    return event_scores, event_reasons, measure_mul, measure_reasons


def _format_event(event, accidentals_map, persistent_accs, dy):
    if event['type'] == 'rest':
        return format_rest(event)
    if event['type'] == 'note_unit':
        return format_note_unit(event['unit'], accidentals_map,
                                persistent_accs, dy=dy)
    return '?'


def _resolve_bpm(measure, default_bpm, anchors):
    if not anchors or not measure:
        return default_bpm
    min_x = min(e['x'] for e in measure)
    bpm = default_bpm
    for x_thr, b in anchors:
        if x_thr <= min_x + 1:
            bpm = b
        else:
            break
    return bpm


def format_confidence_report(staff_measures, accidentals_map,
                             beats_per_measure=2.0, dy=21.0,
                             staff_anchors=None):
    """Build a human-readable confidence report.

    Parameters
    ----------
    staff_measures : list of (staff_name, measures) tuples
    accidentals_map : global accidentals map.
    beats_per_measure : float, default bpm fallback.
    dy : float, staff line spacing.
    staff_anchors : optional list aligned with staff_measures; each entry
        is a list of (x_threshold, bpm) anchors for that staff's measures.
    """
    lines = []
    lines.append("简谱置信度报告")
    lines.append("=" * 60)
    lines.append(
        f"阈值: conf < {LOW_CONF_THRESHOLD:.2f} 标记为 [!] (建议人工复查)")
    lines.append(
        "评分来源: notehead 模板匹配分 / 时值来源 / 小节拍数合理性")
    lines.append("")

    grand_total = 0
    grand_low = 0

    for si_s, (staff_name, measures) in enumerate(staff_measures):
        anchors = None
        if staff_anchors and si_s < len(staff_anchors):
            anchors = staff_anchors[si_s]
        staff_total = 0
        staff_low = 0
        staff_lines = []

        for mi, measure in enumerate(measures):
            m_bpm = _resolve_bpm(measure, beats_per_measure, anchors)
            scores, reasons, mul, mreasons = score_measure(
                measure, m_bpm)
            # apply measure adjustment
            adj_scores = [s * mul for s in scores]

            persistent_accs = {}
            event_strs = []
            low_details = []
            for ei, e in enumerate(measure):
                est = _format_event(e, accidentals_map, persistent_accs, dy)
                c = adj_scores[ei]
                if c < LOW_CONF_THRESHOLD:
                    event_strs.append(f"[!]{est}")
                    low_details.append((ei + 1, est, c, reasons[ei]))
                    staff_low += 1
                else:
                    event_strs.append(est)
                staff_total += 1

            m_str = "|" + " ".join(event_strs) + "|"
            staff_lines.append(f"  M{mi + 1:02d}: {m_str}")
            if mreasons:
                staff_lines.append(
                    f"         measure: {'; '.join(mreasons)}")
            for pos, est, c, rlist in low_details:
                r_str = '; '.join(rlist) if rlist else '(aggregate low)'
                staff_lines.append(
                    f"         pos {pos} \"{est}\" conf={c:.2f}: {r_str}")

        pct = (100.0 * staff_low / staff_total) if staff_total else 0.0
        lines.append(
            f"--- {staff_name}: {staff_total} events, "
            f"{staff_low} low-conf ({pct:.1f}%) ---")
        lines.extend(staff_lines)
        lines.append("")

        grand_total += staff_total
        grand_low += staff_low

    overall_pct = (100.0 * grand_low / grand_total) if grand_total else 0.0
    lines.append("=" * 60)
    lines.append(
        f"汇总: {grand_total} 个事件, "
        f"{grand_low} 个低置信度 ({overall_pct:.1f}%)")
    lines.append(
        "注: 本报告仅对已检测到的音符给出置信度。"
        "检测器完全遗漏的音符无法被标记。")

    return "\n".join(lines)
