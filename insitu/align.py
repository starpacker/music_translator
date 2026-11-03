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
  approximate: OMR routinely merges adjacent staves, drops page breaks, and its
  per-system measure counts drift by one or two from the engraving in a handful
  of places.  Measured on the bundled 梆笛.pdf (18 engraved staves) it emits
  only 17 system breaks.

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
from pathlib import Path
import zipfile

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


def align(pdf_counts: list[int], omr_counts: list[int],
          index_offset: int = 0) -> list[SystemSlice]:
    """Return the OMR measure range assigned to each engraved staff.

    Both inputs are measure counts, in order.  The alignment walks the two
    sequences together and minimises the weighted edit cost described above.
    ``index_offset`` only shifts the staff numbers quoted in the console
    warnings; slices always use 0-based indices into ``pdf_counts``.
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

    # Split OMR systems not paired by the DP still contain real music. Assign
    # gaps to neighbouring staves without ever dropping or duplicating bars.
    if slices:
        total = sum(omr_counts)
        boundaries = [0]
        for left, right in zip(slices, slices[1:]):
            boundary = (left.omr_end + right.omr_start) // 2
            boundaries.append(max(boundaries[-1], min(boundary, total)))
        boundaries.append(total)
        for index, sl in enumerate(slices):
            sl.omr_start, sl.omr_end = boundaries[index:index + 2]
            if sl.omr_measures != sl.pdf_measures:
                print(f'  ! 谱表 {sl.staff_index + index_offset + 1}: '
                      f'五线谱 {sl.pdf_measures} 格，'
                      f'识谱分配 {sl.omr_measures} 小节，请校对')

    return slices


def align_from_files(pdf_path, mxl_path) -> tuple[list, list[SystemSlice]]:
    """Convenience: read the engraved staves and the OMR system breaks, align."""
    try:
        from .pdf_structure import layout_for_document  # type: ignore
    except ImportError:  # run as a plain script / flat import
        from pdf_structure import layout_for_document  # type: ignore
    import xml.etree.ElementTree as ET

    staves = layout_for_document(pdf_path)
    mxl_path = Path(mxl_path)
    if zipfile.is_zipfile(mxl_path):
        with zipfile.ZipFile(mxl_path) as archive:
            container = ET.fromstring(archive.read('META-INF/container.xml'))
            rootfile = next((entry.get('full-path') for entry in container.iter()
                             if entry.tag.rsplit('}', 1)[-1] == 'rootfile'
                             and entry.get('full-path')), None)
            if rootfile is None:
                raise ValueError(f'MXL 缺少 MusicXML 根文件: {mxl_path}')
            root = ET.fromstring(archive.read(rootfile))
    else:
        root = ET.parse(str(mxl_path)).getroot()
    def child(element, name):
        return next((item for item in element if item.tag.rsplit('}', 1)[-1] == name), None)

    part = child(root, 'part')
    if part is None:
        raise ValueError(f'MusicXML 没有可对齐的声部: {mxl_path}')
    measures = [item for item in part if item.tag.rsplit('}', 1)[-1] == 'measure']
    breaks: list[int] = []
    page_breaks: list[int] = []
    for idx, el in enumerate(measures):
        print_el = child(el, 'print')
        if print_el is None:
            continue
        if print_el.get('new-page') == 'yes':
            page_breaks.append(idx)
        if (print_el.get('new-system') == 'yes'
                or print_el.get('new-page') == 'yes'):
            breaks.append(idx)

    pdf_counts = _pdf_counts(staves)

    # Prefer aligning page by page.  A system can never legitimately span a
    # page turn, so letting the global DP move measures across it can hand one
    # page's bars to another page's staves — which happened on
    # 05_中音笙1-2: page 2's OMR was truncated and the aligner stole 20 of
    # page 1's measures to fill it.
    staff_pages = [getattr(getattr(layout, 'staff', None), 'page', None)
                   for layout in staves]
    if staves and None not in staff_pages:
        pdf_pages: list[tuple[int, list[int]]] = []
        for index, page in enumerate(staff_pages):
            if pdf_pages and pdf_pages[-1][0] == page:
                pdf_pages[-1][1].append(index)
            else:
                pdf_pages.append((page, [index]))
        page_starts = sorted({0, *page_breaks})
        omr_pages = [(start, page_starts[i + 1] if i + 1 < len(page_starts)
                      else len(measures))
                     for i, start in enumerate(page_starts)]
        if len(omr_pages) == len(pdf_pages):
            slices: list[SystemSlice] = []
            for (_, staff_ids), (start, end) in zip(pdf_pages, omr_pages):
                local_breaks = [b - start for b in breaks if start <= b < end]
                page_slices = align([pdf_counts[i] for i in staff_ids],
                                    _omr_segments(end - start, local_breaks),
                                    index_offset=staff_ids[0])
                for system in page_slices:
                    system.staff_index = staff_ids[system.staff_index]
                    system.omr_start += start
                    system.omr_end += start
                    slices.append(system)
            slices.sort(key=lambda system: system.staff_index)
            return staves, slices

    return staves, align(pdf_counts, _omr_segments(len(measures), breaks))
