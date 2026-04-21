"""One-time script: extract digit templates from a score image.

Detects multi-measure rest bars (thick horizontal bar on middle staff line)
and crops connected components ABOVE the bar as candidate digit templates.

Usage:
    python extract_digit_templates.py ../input/qudi_p1.png

Saves: digit_candidate_<sysidx>_<segidx>_<compidx>.png in current dir.
You then manually inspect and rename the right ones to ../template/digit_<N>.png.
"""
import sys
import os
import cv2
import numpy as np

from staff_removal import extract_staff_lines
from pitch_detection import get_staff_systems


def main(image_path):
    staff_lines, music_symbols, binary = extract_staff_lines(image_path)
    systems = get_staff_systems(staff_lines)
    dy = float(np.mean([(s[4] - s[0]) / 4.0 for s in systems]))
    print(f"Found {len(systems)} systems, dy={dy:.1f}")

    out_dir = os.path.dirname(os.path.abspath(__file__))
    img_h, img_w = binary.shape
    saved = 0

    for si, sys in enumerate(systems):
        # Search for multi-rest bars on the middle staff line.
        # A multi-rest bar = thick horizontal black region across the
        # middle two staff lines, with row-fill ratio > 0.75 across
        # >= dy*0.3 vertical pixels.
        mid_y = int((sys[1] + sys[3]) / 2)
        bar_y1 = max(0, mid_y - max(3, int(dy * 0.5)))
        bar_y2 = min(img_h, mid_y + max(3, int(dy * 0.5)))

        # Slide horizontally to find thick-bar segments.
        # Use binary image (rest bar may be partially stripped from music_symbols).
        win = max(10, int(dy * 1.5))
        step = max(2, int(dy * 0.3))
        bar_segments = []
        for cx in range(win, img_w - win, step):
            region = binary[bar_y1:bar_y2, cx - win:cx + win]
            if region.size == 0:
                continue
            row_fills = np.mean(region > 127, axis=1)
            thick_rows = int(np.sum(row_fills > 0.75))
            if thick_rows >= max(4, int(dy * 0.3)):
                bar_segments.append(cx)

        # Cluster contiguous bar_segments
        clusters = []
        cur = []
        for x in bar_segments:
            if not cur or x - cur[-1] <= step * 2:
                cur.append(x)
            else:
                clusters.append(cur)
                cur = [x]
        if cur:
            clusters.append(cur)

        # Filter clusters wider than dy*3 (real multi-rest bars, not stems)
        bars = [(min(c), max(c)) for c in clusters
                if (max(c) - min(c)) > dy * 3]

        for bi, (bx1, bx2) in enumerate(bars):
            print(f"  Staff {si}: multi-rest bar at x=[{bx1}, {bx2}]"
                  f" width={bx2-bx1}")

            # Crop region above staff (where digit sits)
            digit_y1 = max(0, sys[0] - int(dy * 4))
            digit_y2 = sys[0] - int(dy * 0.2)
            if digit_y2 <= digit_y1:
                continue
            # X range slightly wider than the bar
            pad = int(dy * 1.0)
            digit_x1 = max(0, bx1 - pad)
            digit_x2 = min(img_w, bx2 + pad)
            crop = binary[digit_y1:digit_y2, digit_x1:digit_x2]
            if crop.size == 0:
                continue

            # Find connected components on the inverted binary
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                crop, connectivity=8)
            # Skip background (label 0)
            comps = []
            for li in range(1, num_labels):
                x, y, w, h, area = stats[li]
                if area < (dy * 0.5) ** 2:
                    continue  # too small (noise)
                if h < dy * 0.8:
                    continue  # too short
                comps.append((x, y, w, h))
            comps.sort(key=lambda c: c[0])  # left-to-right

            for ci, (cx, cy, cw, ch) in enumerate(comps):
                pad2 = 2
                cx1 = max(0, cx - pad2)
                cy1 = max(0, cy - pad2)
                cx2 = min(crop.shape[1], cx + cw + pad2)
                cy2 = min(crop.shape[0], cy + ch + pad2)
                comp_crop = crop[cy1:cy2, cx1:cx2]
                # Invert for saving (white background, black digit)
                save_img = 255 - comp_crop
                fname = f"digit_candidate_s{si}_b{bi}_c{ci}.png"
                cv2.imwrite(os.path.join(out_dir, fname), save_img)
                saved += 1
                print(f"    Saved {fname} (size {cw}x{ch})")

    print(f"\nTotal candidates saved: {saved}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_digit_templates.py <image_path>")
        sys.exit(1)
    main(sys.argv[1])
