"""
test_jianpu_transpose.py — unit tests for the movable-do (1=X) renumbering.

Run:  python test_jianpu_transpose.py
"""
import jianpu_transpose as jt


def test_parse_tonic():
    assert jt.parse_tonic('C') == 0
    assert jt.parse_tonic('G') == 7
    assert jt.parse_tonic('bB') == 10
    assert jt.parse_tonic('Bb') == 10
    assert jt.parse_tonic('#F') == 6
    assert jt.parse_tonic('F#') == 6
    assert jt.parse_tonic('auto') is None
    for bad in ('H', '1', 'bX', ''):
        try:
            if bad == '':
                assert jt.parse_tonic(bad) is None
            else:
                jt.parse_tonic(bad)
                raise AssertionError(f'should raise: {bad}')
        except ValueError:
            pass


def test_midi_round_trip():
    """Renumbering must preserve the absolute pitch exactly."""
    for tonic in range(12):
        for acc in ('', '#', 'b'):
            for deg in range(1, 8):
                for octv in (-2, -1, 0, 1, 2):
                    midi = jt.midi_of_token({'': 0, '#': 1, 'b': -1}[acc], deg, octv)
                    a2, d2, o2 = jt.token_from_midi(midi, tonic)
                    # reading the new token back in the *tonic* frame must
                    # land on the same absolute midi
                    assert jt.midi_of_token({'': 0, '#': 1, 'b': -1}[a2], d2, o2) + tonic == midi


def test_transpose_identity():
    s = "|1 2 3 0 5--- [1 3 5]/2 #4/4 b7/2. ×3|"
    assert jt.transpose_string(s, 0) == s


def test_transpose_g_major():
    # 1=C reading of a G-major fragment → 1=G reading
    src = "|1 2 3 #4 5|"
    assert jt.transpose_string(src, 7) == "|4, 5, 6, 7, 1|"


def test_transpose_preserves_structure():
    src = "|[1 3 5]--- 0/4 6'/2. b7/6 ×2|"
    out = jt.transpose_string(src, 9)
    assert out.count('[') == 1 and out.count(']') == 1
    assert '---' in out and '0/4' in out and '/2.' in out and '/6' in out and '×2' in out


def test_no_duration_corruption():
    # digits inside duration suffixes must not be treated as notes
    src = "|5/12 3/3*2 1/2 6/4|"
    out = jt.transpose_string(src, 7)
    assert out == "|1/12 6,/3*2 4,/2 2/4|", out


def test_tonic_from_key_sig():
    assert jt.tonic_from_key_sig(None) == 0
    assert jt.tonic_from_key_sig({'type': '#', 'count': 3}) == 9   # A major
    assert jt.tonic_from_key_sig({'type': 'b', 'count': 2}) == 10  # Bb major


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print(f'  ok  {name}')
    print('all jianpu_transpose tests passed')
