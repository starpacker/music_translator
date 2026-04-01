"""
evaluate.py
Compare converter output against ground truth to measure accuracy.

Metrics:
- Note accuracy: correct pitch (number + octave)
- Accidental accuracy: correct sharp/flat detection
- Duration accuracy: correct rhythmic value
- Measure-level comparison with diff output
"""
import re
import sys


def parse_jianpu_line(line):
    """
    Parse a jianpu line like "|0 [1 1']/2 [2 2']/2|" into a list of measures,
    each measure being a list of event strings.
    """
    line = line.strip()
    # Split by | and filter empty
    parts = re.split(r'\|', line)
    measures = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        measures.append(part)
    return measures


def parse_events(measure_str):
    """
    Parse a measure string into individual events.
    E.g. "0 [1 1']/2 [2 2']/2" → ["0", "[1 1']/2", "[2 2']/2"]
    """
    events = []
    i = 0
    s = measure_str.strip()

    while i < len(s):
        if s[i] == ' ':
            i += 1
            continue

        if s[i] == '[':
            # Chord: find matching ]
            j = s.index(']', i)
            token = s[i:j + 1]
            # Check for duration suffix after ]
            j += 1
            if j < len(s) and s[j] == '/':
                while j < len(s) and s[j] not in ' [':
                    j += 1
                token = s[i:j]
            events.append(token)
            i = j
        else:
            # Single note/rest
            j = i
            while j < len(s) and s[j] not in ' [':
                j += 1
            token = s[i:j]
            if token:
                events.append(token)
            i = j

    return events


def extract_notes_from_event(event_str):
    """
    Extract individual note strings from an event.
    "[#1 3' 5]/2" → ["#1", "3'", "5"], "/2"
    "0" → ["0"], ""
    "#5/4" → ["#5"], "/4"
    """
    duration = ""
    body = event_str

    # Extract duration suffix (outside brackets)
    if ']' in body:
        bracket_end = body.index(']')
        rest = body[bracket_end + 1:]
        if rest.startswith('/'):
            duration = rest
        body = body[:bracket_end + 1]
    else:
        match = re.search(r'/\d+$', body)
        if match:
            duration = match.group()
            body = body[:match.start()]

    # Remove brackets
    body = body.strip('[]').strip()

    if not body:
        return [], duration

    notes = body.split()
    return notes, duration


def compare_measures(gt_measures, out_measures, label=""):
    """
    Compare ground truth and output measure by measure.
    Returns dict with statistics.
    """
    stats = {
        'total_measures': max(len(gt_measures), len(out_measures)),
        'matching_measures': 0,
        'total_events': 0,
        'matching_events': 0,
        'total_notes': 0,
        'correct_pitch': 0,
        'correct_duration': 0,
        'diffs': [],
    }

    n = max(len(gt_measures), len(out_measures))
    for i in range(n):
        gt_m = gt_measures[i] if i < len(gt_measures) else ""
        out_m = out_measures[i] if i < len(out_measures) else ""

        if gt_m == out_m:
            stats['matching_measures'] += 1

        gt_events = parse_events(gt_m)
        out_events = parse_events(out_m)

        max_events = max(len(gt_events), len(out_events))
        stats['total_events'] += max_events

        for j in range(max_events):
            gt_e = gt_events[j] if j < len(gt_events) else ""
            out_e = out_events[j] if j < len(out_events) else ""

            if gt_e == out_e:
                stats['matching_events'] += 1

            gt_notes, gt_dur = extract_notes_from_event(gt_e) if gt_e else ([], "")
            out_notes, out_dur = extract_notes_from_event(out_e) if out_e else ([], "")

            max_notes = max(len(gt_notes), len(out_notes))
            stats['total_notes'] += max_notes

            for k in range(min(len(gt_notes), len(out_notes))):
                if gt_notes[k] == out_notes[k]:
                    stats['correct_pitch'] += 1

            if gt_dur == out_dur and gt_e and out_e:
                stats['correct_duration'] += 1

        if gt_m != out_m:
            stats['diffs'].append({
                'measure': i + 1,
                'gt': gt_m,
                'out': out_m,
            })

    return stats


def print_report(stats, label):
    """Print a comparison report."""
    print(f"\n{'─' * 50}")
    print(f"  {label}")
    print(f"{'─' * 50}")

    total_m = stats['total_measures']
    match_m = stats['matching_measures']
    print(f"  Measures:  {match_m}/{total_m} exact match "
          f"({match_m / total_m * 100:.0f}%)" if total_m else "  No measures")

    total_e = stats['total_events']
    match_e = stats['matching_events']
    if total_e:
        print(f"  Events:    {match_e}/{total_e} exact match "
              f"({match_e / total_e * 100:.0f}%)")

    total_n = stats['total_notes']
    correct_p = stats['correct_pitch']
    if total_n:
        print(f"  Pitch:     {correct_p}/{total_n} correct "
              f"({correct_p / total_n * 100:.0f}%)")

    if stats['diffs']:
        print(f"\n  Differences ({len(stats['diffs'])}):")
        for d in stats['diffs'][:10]:  # Show first 10
            print(f"    M{d['measure']:2d} GT:  {d['gt']}")
            print(f"         OUT: {d['out']}")


def load_gt_from_file(path):
    """Load ground truth from ground_truth.md format."""
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract treble and bass sections
    treble_lines = []
    bass_lines = []
    section = None

    for line in content.split('\n'):
        line = line.strip()
        if '高音' in line:
            section = 'treble'
            continue
        elif '低音' in line:
            section = 'bass'
            continue

        if line.startswith('|') and section:
            if section == 'treble':
                treble_lines.append(line)
            elif section == 'bass':
                bass_lines.append(line)

    treble_measures = []
    for line in treble_lines:
        treble_measures.extend(parse_jianpu_line(line))

    bass_measures = []
    for line in bass_lines:
        bass_measures.extend(parse_jianpu_line(line))

    return treble_measures, bass_measures


def load_output_from_file(path):
    """Load converter output from output_jianpu.txt format."""
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    treble_lines = []
    bass_lines = []
    section = None

    for line in content.split('\n'):
        line = line.strip()
        if '高音部分' in line or line.startswith('高音部分'):
            section = 'treble'
            continue
        elif '低音部分' in line or line.startswith('低音部分'):
            section = 'bass'
            continue
        elif line.startswith('=') or line.startswith('---'):
            continue

        if line.startswith('|') and section:
            if section == 'treble':
                treble_lines.append(line)
            elif section == 'bass':
                bass_lines.append(line)

    treble_measures = []
    for line in treble_lines:
        treble_measures.extend(parse_jianpu_line(line))

    bass_measures = []
    for line in bass_lines:
        bass_measures.extend(parse_jianpu_line(line))

    return treble_measures, bass_measures


def main():
    import os
    gt_path = os.path.join(os.path.dirname(__file__), "ground_truth.md")
    out_path = os.path.join(os.path.dirname(__file__), "output_jianpu.txt")

    if len(sys.argv) > 1:
        gt_path = sys.argv[1]
    if len(sys.argv) > 2:
        out_path = sys.argv[2]

    if not os.path.exists(gt_path):
        print(f"Error: Ground truth file not found: {gt_path}")
        sys.exit(1)
    if not os.path.exists(out_path):
        print(f"Error: Output file not found: {out_path}")
        print("Run main.py first to generate output_jianpu.txt")
        sys.exit(1)

    print("Loading ground truth and output...")
    gt_treble, gt_bass = load_gt_from_file(gt_path)
    out_treble, out_bass = load_output_from_file(out_path)

    print(f"GT:  {len(gt_treble)} treble measures, {len(gt_bass)} bass measures")
    print(f"Out: {len(out_treble)} treble measures, {len(out_bass)} bass measures")

    # Only compare up to the number of ground truth measures
    # (GT may only cover the first line of the score)
    out_treble_trimmed = out_treble[:len(gt_treble)]
    out_bass_trimmed = out_bass[:len(gt_bass)]

    treble_stats = compare_measures(gt_treble, out_treble_trimmed, "Treble (高音)")
    bass_stats = compare_measures(gt_bass, out_bass_trimmed, "Bass (低音)")

    print_report(treble_stats, "TREBLE (高音) Comparison")
    print_report(bass_stats, "BASS (低音) Comparison")

    # Overall
    total_notes = treble_stats['total_notes'] + bass_stats['total_notes']
    correct = treble_stats['correct_pitch'] + bass_stats['correct_pitch']
    total_measures = treble_stats['total_measures'] + bass_stats['total_measures']
    match_measures = treble_stats['matching_measures'] + bass_stats['matching_measures']

    print(f"\n{'=' * 50}")
    print(f"  OVERALL")
    print(f"{'=' * 50}")
    if total_measures:
        print(f"  Measure accuracy: {match_measures}/{total_measures} "
              f"({match_measures / total_measures * 100:.0f}%)")
    if total_notes:
        print(f"  Pitch accuracy:   {correct}/{total_notes} "
              f"({correct / total_notes * 100:.0f}%)")


if __name__ == "__main__":
    main()
