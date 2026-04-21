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
from config import CFG


def parse_jianpu_line(line):
    """Parse a jianpu line into a list of measure strings."""
    line = line.strip()
    parts = re.split(r'\|', line)
    return [p.strip() for p in parts if p.strip()]


def parse_events(measure_str):
    """Parse a measure string into individual events."""
    events = []
    i = 0
    s = measure_str.strip()
    while i < len(s):
        if s[i] == ' ':
            i += 1
            continue
        if s[i] == '[':
            j = s.index(']', i)
            token = s[i:j + 1]
            j += 1
            if j < len(s) and s[j] == '/':
                while j < len(s) and s[j] not in ' [':
                    j += 1
                token = s[i:j]
            events.append(token)
            i = j
        else:
            j = i
            while j < len(s) and s[j] not in ' [':
                j += 1
            token = s[i:j]
            if token:
                events.append(token)
            i = j
    return events


def extract_notes_from_event(event_str):
    """Extract individual note strings and duration suffix from an event."""
    duration = ""
    body = event_str
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
    body = body.strip('[]').strip()
    if not body:
        return [], duration
    return body.split(), duration


def compare_measures(gt_measures, out_measures, label=""):
    """Compare ground truth and output measure by measure."""
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
            stats['diffs'].append({'measure': i + 1, 'gt': gt_m, 'out': out_m})
    return stats


def print_report(stats, label):
    """Print a comparison report."""
    print(f"\n{'─' * 50}")
    print(f"  {label}")
    print(f"{'─' * 50}")
    total_m = stats['total_measures']
    match_m = stats['matching_measures']
    if total_m:
        print(f"  Measures:  {match_m}/{total_m} exact match "
              f"({match_m / total_m * 100:.0f}%)")
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
        for d in stats['diffs'][:10]:
            print(f"    M{d['measure']:2d} GT:  {d['gt']}")
            print(f"         OUT: {d['out']}")


def _normalize_gt_measure(measure_str):
    """Normalize ground truth measure string to standard format.

    Handles:
    - '5'---' → '5'' (whole note, strip dashes)
    - '6'(1.5拍)' → '6'.' (dotted note)
    - Other annotations in parentheses
    """
    s = measure_str
    # Normalize dotted annotations first: (1.5拍) → . suffix
    s = re.sub(r'\(1\.5拍\)', '.', s)
    s = re.sub(r'\(\d+\.?\d*拍\)', '', s)
    # Remove all remaining parenthetical comments (undone, Chinese annotations, etc.)
    s = re.sub(r'\([^)]*\)', '', s)
    return s.strip()


def load_gt_from_file(path):
    """Load ground truth from ground_truth.md.

    Supports two formats:
    1. Grand staff: "高音部分 (line=N):" / "低音部分 (line=N):"
       Returns: {(clef, line_num): [measure_strings]}
    2. Single staff: "line N:"
       Returns: {('solo', line_num): [measure_strings]}
    """
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    sections = {}
    current_key = None

    for line in content.split('\n'):
        stripped = line.strip()
        # Skip empty lines
        if not stripped:
            continue

        # Format 1: Grand staff headers
        if '高音' in stripped or '低音' in stripped:
            clef = 'treble' if '高音' in stripped else 'bass'
            m = re.search(r'line\s*=\s*(\d+)', stripped)
            line_num = int(m.group(1)) if m else 1
            current_key = (clef, line_num)
            if current_key not in sections:
                sections[current_key] = []
            continue

        # Format 2: Single staff headers ("line N:" or "line N:(undone)")
        m = re.match(r'line\s+(\d+)\s*:', stripped, re.IGNORECASE)
        if m:
            line_num = int(m.group(1))
            current_key = ('solo', line_num)
            # Check for (undone) marker — skip incomplete sections entirely
            if 'undone' in stripped.lower():
                current_key = None
                continue
            if current_key not in sections:
                sections[current_key] = []
            continue

        # Format 3: Chinese headers ("--- 第N行 ---")
        m = re.match(r'---\s*第\s*(\d+)\s*行\s*---', stripped)
        if m:
            line_num = int(m.group(1))
            current_key = ('solo', line_num)
            if current_key not in sections:
                sections[current_key] = []
            continue

        # Special: "空N拍" rest measures in GT (Chinese notation)
        # Treated as N full measures of rest. Format varies:
        # - Single measure: "0---" (whole rest) or "0 0 0 0"
        # - Multiple measures: each as "0 0 0 0"
        if current_key and re.match(r'^\|?空[一二两三四]拍\|?$', stripped):
            count_map = {'一': 1, '二': 2, '两': 2, '三': 3, '四': 4}
            m_rest = re.search(r'空([一二两三四])拍', stripped)
            if m_rest:
                n = count_map.get(m_rest.group(1), 1)
                if n == 1:
                    sections[current_key].append('0---')
                else:
                    for _ in range(n):
                        sections[current_key].append('0 0 0 0')
            continue

        if stripped.startswith('|') and current_key:
            measures = parse_jianpu_line(stripped)
            measures = [_normalize_gt_measure(m) for m in measures]
            sections[current_key].extend(m for m in measures if m)

    return sections


def load_output_from_file(path):
    """Load converter output from output/jianpu.txt.

    Returns dict: {(clef, line_num): [measure_strings]}
    Supports both grand-staff (高音/低音 labels) and single-staff output.
    """
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    sections = {}
    current_clef = None
    line_num = 0
    is_single_staff = False

    for line in content.split('\n'):
        stripped = line.strip()
        # Detect line headers: "--- 第N行 ---" or "--- System N ---"
        m = re.search(r'第\s*(\d+)\s*行|System\s+(\d+)', stripped)
        if m:
            line_num = int(m.group(1) or m.group(2))
            current_clef = None
            continue
        if stripped.startswith('='):
            line_num = 0
            current_clef = None
            continue
        if stripped.startswith('---') and '行' not in stripped:
            continue
        # Detect clef: "高音:" or "Treble:" / "低音:" or "Bass:"
        if stripped.startswith('高音') or stripped.startswith('Treble'):
            current_clef = 'treble'
        elif stripped.startswith('低音') or stripped.startswith('Bass'):
            current_clef = 'bass'

        if '|' in stripped and line_num > 0:
            measure_part = stripped
            # Remove leading labels
            for prefix in ['高音:', '低音:', 'Treble:', 'Bass:',
                           '高音: ', '低音: ']:
                idx = measure_part.find(prefix)
                if idx >= 0:
                    measure_part = measure_part[idx + len(prefix):]
                    current_clef = 'treble' if '高' in prefix else 'bass'
                    break

            measures = parse_jianpu_line(measure_part)
            if current_clef and line_num > 0:
                key = (current_clef, line_num)
            elif line_num > 0:
                # Single-staff mode: no clef label
                key = ('solo', line_num)
                is_single_staff = True
            else:
                continue
            if key not in sections:
                sections[key] = []
            sections[key].extend(measures)

    return sections


def main():
    import os
    gt_path = os.path.join(os.path.dirname(__file__), "ground_truth.md")
    out_path = os.path.join(os.path.dirname(__file__), "output", "jianpu.txt")

    if len(sys.argv) > 1:
        gt_path = sys.argv[1]
    if len(sys.argv) > 2:
        out_path = sys.argv[2]

    if not os.path.exists(gt_path):
        print(f"Error: Ground truth file not found: {gt_path}")
        sys.exit(1)
    if not os.path.exists(out_path):
        print(f"Error: Output file not found: {out_path}")
        print("Run main.py first to generate output/jianpu.txt")
        sys.exit(1)

    print("Loading ground truth and output...")
    gt_sections = load_gt_from_file(gt_path)
    out_sections = load_output_from_file(out_path)

    print(f"GT sections:  {sorted(gt_sections.keys())}")
    print(f"Out sections: {sorted(out_sections.keys())}")

    all_stats = []
    for key in sorted(gt_sections.keys()):
        clef, line_num = key
        gt_measures = gt_sections[key]
        out_measures = out_sections.get(key, [])
        if clef == 'solo':
            label = f"Line {line_num}"
        else:
            label = f"{'TREBLE' if clef == 'treble' else 'BASS'} Line {line_num}"
        if not out_measures:
            print(f"\n  WARNING: No output for {label}")
            continue
        stats = compare_measures(gt_measures, out_measures, label)
        print_report(stats, label)
        all_stats.append(stats)

    # Overall
    total_notes = sum(s['total_notes'] for s in all_stats)
    correct = sum(s['correct_pitch'] for s in all_stats)
    total_measures = sum(s['total_measures'] for s in all_stats)
    match_measures = sum(s['matching_measures'] for s in all_stats)
    total_events = sum(s['total_events'] for s in all_stats)
    match_events = sum(s['matching_events'] for s in all_stats)

    print(f"\n{'=' * 50}")
    print(f"  OVERALL")
    print(f"{'=' * 50}")
    if total_measures:
        print(f"  Measure accuracy: {match_measures}/{total_measures} "
              f"({match_measures / total_measures * 100:.0f}%)")
    if total_events:
        print(f"  Event accuracy:   {match_events}/{total_events} "
              f"({match_events / total_events * 100:.0f}%)")
    if total_notes:
        print(f"  Pitch accuracy:   {correct}/{total_notes} "
              f"({correct / total_notes * 100:.0f}%)")


if __name__ == "__main__":
    main()
