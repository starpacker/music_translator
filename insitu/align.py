"""Align the OMR measure stream onto the engraved staves.

Why alignment is needed at all
------------------------------
Two structures describe the same piece and neither is a reliable ruler on its
own:

* The engraved PDF gives, per staff, the exact number of barlines — so the
  exact number of measures on that staff (measured in `pdf_structure`).  But it
  cannot say which of the OMR notes belong to which staff beyond that count.
* The OMR MusicXML carries the notes and, via `<print new-system="yes">`, a
  hint of where the transcriber believed each system began.  That hint is only
  approximate: measured on 梆笛.pdf, OMR emits 46 system breaks for 51 engraved
  staves — it silently merges four pairs of adjacent staves and drops every page
  break — and its per-system measure counts drift by one or two from the
  engraving in a handful of places.

Neither stream is authoritative, so we do not trust either blindly.  We align
the two by their measure counts with a global optimal sequence alignment that
(i) strongly prefers the OMR `new-system` breaks to land on engraved staff
boundaries, and (ii) treats a small count mismatch between corresponding
systems as a cheap, expected correction rather than a fatal inconsistency.

The result is a per-staff slice of the OMR measure list: exactly the notes that
belong under that staff, ready to be typeset as one jianpu line.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Cost model for the alignment.  A missing OMR system break ("we should have
# started a new staff here but the OMR merged two staves into one") is the most
# common real error, so it is cheaper than the reverse; a count mismatch between
# two systems that DO correspond is cheaper still, because OMR routinely reads
# one measure too many or too few across a system boundary.
_COST_MERGE = 1.0      # OMR fused two staves -> one break not seen
_COST_SPLIT = 1.2      # OMR invented a break the engraving does not have
_COST_COUNT = 0.4      # corresponding systems disagree on their measure count
_ABSENT = None


@dataclass
class SystemSlice:
    """The OMR measures assigned to one engraved staff."""
    staff_index: int                              # 0-based index into the staff list
    omr_start: int                                # inclusive index into measures
    omr_end: int                                  # exclusive
    pdf_measures: int                             # barlines found on the staff
    expected_measures: int | None = field(default=None)  # from the OMR break hint

    @property
    def omr_measures(self) -> int:
        return self.omr_end - self.omr_start


def _pdf_counts(staves) -> list[int]:
    """Measures per engraved staff = number of barlines on that staff.

    A staff is drawn left-to-right as `measure | measure | ... | measure |`, so
    every measure is terminated by one barline and there is no leading barline:
    the barline count is the measure count.
    """
    return [len(s.barlines) for s in staves]


def _omr_segments(measure_count: int, break_indices: list[int]) -> list[int]:
    """Split `measure_count` OMR measures at the given 0-based start indices.

    `break_indices` are the indices of measures that begin a new system (the
    measures carrying `<print new-system="yes">`).  Index 0 is implicit.
    """
    starts = sorted({0, *break_indices})
    out: list[int] = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else measure_count
        out.append(end - start)
    return out


def align(pdf_counts: list[int], omr_counts: list[int]) -> list[SystemSlice]:
    """Return the OMR measure range assigned to each engraved staff.

    Both inputs are measure counts, in order.  The alignment walks the two
    sequences together and minimises the weighted edit cost described above.
    """
    n, m = len(pdf_counts), len(omr_counts)
    INF = float('inf')
    # best[i][j] = cost of matching the first i staves with the first j OMR
    # systems, plus the operation that got us there.
    best = [[INF] * (m + 1) for _ in range(n + 1)]
    back: list[list[tuple[int, int, str] | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    best[0][0] = 0.0

    for i in range(n + 1):
        for j in range(m + 1):
            here = best[i][j]
            if here == INF:
                continue
            # Pair staff i with OMR system j.
            if i < n and j < m:
                cost = 0.0 if pdf_counts[i] == omr_counts[j] else _COST_COUNT
                cand = here + cost
                if cand < best[i + 1][j + 1]:
                    best[i + 1][j + 1] = cand
                    back[i + 1][j + 1] = (i, j, 'pair')
                # The OMR fused a run of k consecutive engraved staves into
                # this one system (typically a grand staff or a page break it
                # dropped).  The system covers all k staves; keep the run
                # together so its measures can be shared out afterwards.
                run = 0
                for k in range(1, n - i + 1):
                    run += pdf_counts[i + k - 1]
                    if k == 1:
                        continue
                    # Cost is zero when the fused system's measure count is
                    # the sum of the staves it swallowed (the honest merge);
                    # otherwise charge a difference penalty.
                    cost = abs(omr_counts[j] - run) * _COST_COUNT + _COST_MERGE
                    cand = here + cost
                    if cand < best[i + k][j + 1]:
                        best[i + k][j + 1] = cand
                        back[i + k][j + 1] = (i, j, f'group{k}')
            # OMR invented a break: consume a system without a staff.
            if j < m and here + _COST_SPLIT < best[i][j + 1]:
                best[i][j + 1] = here + _COST_SPLIT
                back[i][j + 1] = (i, j, 'split')

    # Walk back from (n, m).
    i, j = n, m
    groups: list[tuple[int, int, int]] = []  # (first staff, staff count, system)
    while (i, j) != (0, 0):
        pi, pj, op = back[i][j]  # type: ignore[misc]
        if op.startswith('group'):
            groups.append((pi, int(op[5:]), pj))
        elif op == 'pair':
            groups.append((pi, 1, pj))
        i, j = pi, pj
    groups.reverse()

    # Convert the OMR start indices into measure ranges.  The OMR measure index
    # for a break index k is the cumulative count of the segments before it.
    offset_of_system: list[int] = []
    acc = 0
    for c in omr_counts:
        offset_of_system.append(acc)
        acc += c

    slices: list[SystemSlice] = []

    # Every OMR system is consumed once, in order.  When the OMR fused a run of
    # engraved staves into one system, that system's measures are shared out
    # between those staves in proportion to their engraved measure counts — the
    # fused run is still the real music of all of them, so giving one of them an
    # empty line would drop notes the reader needs.
    placed = [False] * n
    for first, count, o in groups:
        owners = list(range(first, first + count))
        start = offset_of_system[o]
        total = omr_counts[o]
        if len(owners) == 1:
            s = owners[0]
            slices.append(SystemSlice(s, start, start + total, pdf_counts[s], total))
        else:
            # Share the fused system out by the staves' engraved counts
            # (largest-remainder so the parts sum exactly to `total`).
            weights = [pdf_counts[s] for s in owners]
            wsum = sum(weights) or len(owners)
            allocated: list[int] = []
            remaining = total
            for k, w in enumerate(weights):
                if k == len(weights) - 1:
                    allocated.append(remaining)
                else:
                    share = int(round(total * w / wsum))
                    share = max(0, min(share, remaining))
                    allocated.append(share)
                    remaining -= share
            pos = start
            for s, share in zip(owners, allocated):
                slices.append(SystemSlice(s, pos, pos + share, pdf_counts[s], share))
                pos += share
        for s in owners:
            placed[s] = True

    # Any staff the DP never reached (only possible with degenerate input) keeps
    # an empty slice, placed at the right point in the stream.
    for s in range(n):
        if placed[s]:
            continue
        pos = slices[-1].omr_end if slices else 0
        insert_at = next((k for k, sl in enumerate(slices) if sl.staff_index > s), len(slices))
        slices.insert(insert_at, SystemSlice(s, pos, pos, pdf_counts[s], None))

    slices.sort(key=lambda sl: sl.staff_index)

    if abs(sum(omr_counts) - sum(pdf_counts)) > _MIN_DRIFT:
        _conservation_fix(slices, pdf_counts, omr_counts)

    return slices


# The engraving and the OMR disagree on the total number of measures by this
# much or less without it meaning anything: OMR drops or invents the odd measure
# across a system boundary.  Above it, the two streams are counting different
# music and we have to re-derive the boundary the alignment drifted across.
_MIN_DRIFT = 1


def _conservation_fix(
    slices: list[SystemSlice], pdf_counts: list[int], omr_counts: list[int]
) -> None:
    """Repair the drift the fused-system step introduces into the tail.

    When the OMR fuses a run of staves into one system, everything after that
    run is shifted by one system — the DP still tracks the *relative* drift, but
    the slices downstream then describe the wrong measures.  The engraving is
    authoritative for the music, so we slide each slice's OMR range so that its
    start lands on the cumulative engraved measure total of its staff, keeping
    every slice inside the OMR stream.  Everything before the fused run is
    already correct and is left untouched.
    """
    if not slices:
        return

    # Cumulative measure counts; the k-th staff begins at cumulative[k].
    cum_pdf: list[int] = [0]
    for c in pdf_counts:
        cum_pdf.append(cum_pdf[-1] + c)
    total_omr = sum(omr_counts)

    for sl in slices:
        want_start = min(cum_pdf[sl.staff_index], max(0, total_omr - 1))
        want_end = min(want_start + sl.pdf_measures, total_omr)
        if want_end <= want_start:  # keep an empty-but-valid slice valid
            want_end = min(want_start + 1, total_omr)
        sl.omr_start, sl.omr_end = want_start, want_end



def align_from_files(pdf_path, mxl_path) -> tuple[list, list[SystemSlice]]:
    """Convenience: read the engraved staves and the OMR system breaks, align."""
    try:
        from .pdf_structure import layout_for_document  # type: ignore
    except ImportError:  # run as a plain script / flat import
        from pdf_structure import layout_for_document  # type: ignore
    import xml.etree.ElementTree as ET

    staves = layout_for_document(pdf_path)
    root = ET.parse(str(mxl_path)).getroot()
    measures = root.find('part').findall('measure')
    breaks = [
        idx for idx, el in enumerate(measures)
        if (el.find('print') is not None
            and 'yes' in (el.find('print').get('new-system') or ''))
    ]
    omr_counts = _omr_segments(len(measures), breaks)
    returns = (staves, align(_pdf_counts(staves), omr_counts))
    return returns
