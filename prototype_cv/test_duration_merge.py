"""Tests for duration-aligned event merging."""
import subprocess
import sys


def test_first_chord_detected():
    """The [3 1'] chord at x~284 on line 2 must not be filtered."""
    result = subprocess.run(
        [sys.executable, "main.py", "../input_page1.png"],
        capture_output=True, text=True, cwd="."
    )
    output = result.stdout
    lines = output.split('\n')
    for i, line in enumerate(lines):
        if 'System 2' in line:
            treble_line = lines[i + 1]
            assert "[3 1']" in treble_line, (
                f"First chord [3 1'] missing from System 2 treble: {treble_line}"
            )
            print(f"PASS: [3 1'] found in System 2 treble")
            return
    raise AssertionError("System 2 not found in output")


if __name__ == "__main__":
    test_first_chord_detected()
