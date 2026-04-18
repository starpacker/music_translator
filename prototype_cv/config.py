"""
config.py — Central configuration for all hyperparameters.

All tunable constants live here. No magic numbers in other modules.
Values are expressed as multipliers of dy (staff line spacing) where
possible so the pipeline scales automatically to different resolutions.

Usage:
    from config import CFG
    threshold = CFG.notehead.match_threshold
"""
from dataclasses import dataclass, field
from typing import List
import os

# ── Shared template paths ───────────────────────────────────
_BASE = os.path.dirname(__file__)
TEMPLATE_DIR = os.path.join(_BASE, "..", "template")
PICTURE_DIR = os.path.join(_BASE, "..", "repo",
    "translate-staff-to-simple-musical-notation-master",
    "score_recognition_v4", "picture")
PICTURE_EXPAND_DIR = os.path.join(_BASE, "..", "repo",
    "translate-staff-to-simple-musical-notation-master",
    "score_recognition_v4", "picture_expand")


# ── Staff & Image ────────────────────────────────────────────
@dataclass
class StaffConfig:
    horizontal_kernel_divisor: int = 30     # img_w // N for staff line morphology
    ledger_kernel_divisor: int = 100        # img_w // N for ledger line detection
    thick_line_kernel_height: int = 4       # vertical kernel for thick line removal
    peak_threshold_ratio: float = 0.3       # staff line peak detection threshold
    cluster_distance_px: int = 10           # pixel distance for line clustering
    spacing_tolerance: float = 0.8          # dy consistency check (fraction of avg_dy)
    grand_staff_gap_factor: float = 4.0     # max gap between staves = factor * staff_height


# ── Notehead Detection ───────────────────────────────────────
@dataclass
class NoteheadConfig:
    match_threshold: float = 0.55           # template correlation threshold
    ellipse_width_dy: float = 1.3           # notehead width = dy * N
    ellipse_height_dy: float = 0.85         # notehead height = dy * N
    canvas_size_dy: float = 2.0             # template canvas = dy * N
    ledger_line_thickness_dy: float = 0.12
    ledger_line_width_dy: float = 1.8
    morph_kernel_dy: float = 0.65           # morphological kernel for detection
    area_min_dy2: float = 0.2               # min area = dy^2 * N
    area_max_dy2: float = 6.0               # max area = dy^2 * N
    chord_split_height_dy: float = 1.6      # height trigger for chord splitting
    chord_split_width_dy: float = 2.5       # width constraint for chord splitting
    peak_cluster_dy: float = 0.3            # peak clustering distance
    peak_height_ratio: float = 0.65         # peak threshold = max * N
    weak_peak_ratio: float = 0.80           # weak peak multiplier
    centroid_half_window_dy: float = 0.45   # weighted centroid window
    ledger_search_dy: float = 5.0           # ledger line zone above/below staff
    on_staff_margin_dy: float = 1.0         # on-staff search margin
    ledger_match_threshold: float = 0.55    # template threshold for ledger notes
    nms_overlap: float = 0.3               # NMS overlap threshold
    dedup_distance_dy: float = 0.4          # center-distance deduplication
    exclusion_zone_margin_dy: float = 0.3   # exclusion zone margin
    exclusion_match_threshold: float = 0.55 # exclusion zone template threshold
    exclusion_nms_overlap: float = 0.3


# ── Clef & Key Signature ─────────────────────────────────────
@dataclass
class ClefConfig:
    area_ratio_first: float = 0.17          # clef area = img_w * N (first system)
    area_ratio_default: float = 0.18        # default clef boundary
    area_ratio_right: float = 0.96          # right boundary exclusion
    scan_margin_dy: float = 2.0             # search margin for clef detection
    scaling_range: List[float] = field(default_factory=lambda: [0.7, 0.85, 1.0, 1.15, 1.3])


# ── Barline Detection ────────────────────────────────────────
@dataclass
class BarlineConfig:
    template_threshold: float = 0.50        # barline template matching threshold
    morph_height_ratio: float = 0.8         # vertical kernel = staff_height * N
    thick_kernel_dy: float = 0.3            # horizontal kernel for thick lines
    projection_threshold_ratio: float = 0.5 # projection = staff_height * 255 * N
    dedup_distance_dy: float = 3.0          # min distance between raw barlines
    min_spacing_dy: float = 18.0            # minimum spacing between barlines
    search_margin_dy: float = 0.3           # above/below staff for template search
    clef_end_ratio: float = 0.25            # percentage of width for clef area
    right_boundary_ratio: float = 0.96      # right edge exclusion

    # Barline merge (treble + bass)
    match_tolerance_dy: float = 8.0         # max treble/bass distance for pairing
    verify_base_radius_dy: float = 4.0      # base search radius for verification
    verify_extra_radius_dy: float = 8.0     # extra radius for misaligned candidates
    verify_density_threshold: float = 0.55  # min density per staff for verification
    scan_step_px: int = 3                   # pixel step for barline scanning
    min_gap_median_ratio: float = 0.6       # min gap = median_spacing * N


# ── Stem Tracking ────────────────────────────────────────────
@dataclass
class StemConfig:
    half_width_dy: float = 0.15             # stem half-width for density
    search_offset_dy: float = 0.3           # search range offset
    notehead_offset_dy: float = 0.2         # alternative offset (w/2 + dy*N)
    scan_distance_dy: float = 2.0           # initial scan distance
    density_threshold: float = 0.3          # stem density threshold
    gap_tolerance_dy: float = 0.3           # allowed gap in tracing
    max_search_dy: float = 4.0              # maximum search distance
    far_scan_dy: float = 3.0                # extended scan for stem x
    min_length_dy: float = 1.5              # minimum valid stem length


# ── Chord Grouping & Note Units ──────────────────────────────
@dataclass
class ChordConfig:
    stem_proximity_dy: float = 0.5          # stem_x proximity for grouping
    x_proximity_dy: float = 0.9             # notehead x proximity (fallback)
    y_range_max_dy: float = 4.0             # max y-range for chord notes
    fill_check_pad_dy: float = 0.15         # padding for fill check
    fill_min_on_staff: float = 0.25         # min fill for on-staff notes
    fill_min_off_staff: float = 0.45        # min fill for off-staff notes
    off_staff_below_dy: float = 1.5         # distance below staff for stricter fill
    off_staff_above_dy: float = 3.0         # distance above staff for stricter fill


# ── Beam & Duration Detection ────────────────────────────────
@dataclass
class BeamConfig:
    roi_extent_dy: float = 1.5              # ROI height above/below tip
    roi_restrict_dy: float = 0.5            # restricted ROI for up-stems
    roi_width_dy: float = 1.0               # ROI half-width centered on stem
    binary_threshold: int = 127             # pixel value threshold
    density_threshold: float = 0.25         # beam density threshold
    min_thickness_dy: float = 0.08          # min beam thickness
    max_thickness_dy: float = 0.55          # max beam thickness
    flag_x_start_dy: float = 0.2            # flag search x offset
    flag_x_end_dy: float = 1.5              # flag search x range
    flag_y_extent_dy: float = 1.5           # flag search y range
    flag_y_narrow_dy: float = 0.2           # narrow flag bound
    flag_y_default_dy: float = 0.6          # default flag y range
    flag_density_threshold: float = 0.14    # flag presence threshold
    # Wide ROI for chords
    wide_roi_width_dy: float = 3.0          # chord beam ROI half-width
    wide_pad_dy: float = 0.5                # y-extent padding for chord beams
    wide_density_threshold: float = 0.35    # density for wide ROI
    wide_max_thickness_dy: float = 0.45     # max thickness for wide beams
    hollow_fill_ratio: float = 0.4          # hollow notehead detection threshold
    hollow_check_radius_dy: float = 0.2     # center region radius


# ── Duration & Rhythm ────────────────────────────────────────
@dataclass
class DurationConfig:
    standard_durations: List[float] = field(default_factory=lambda: [0.25, 0.5, 1.0, 2.0, 4.0])
    default_no_stem: float = 1.0            # duration when stem not found
    default_individual: float = 0.25        # default per-note duration
    pickup_barline_ratio: float = 0.4       # first barline position ratio
    pickup_duration: float = 0.5            # pickup note duration
    min_remaining_beats: float = 0.25       # min beats after rest deduction
    snap_tolerance: float = 0.1             # tolerance for duration snapping
    beats_per_measure: float = 2.0          # default; overridden by auto-detect
    use_legacy_bass: bool = True            # True = bass reads as treble (for GT compat)


# ── Two-Voice Merge ──────────────────────────────────────────
@dataclass
class MergeConfig:
    nearby_tolerance_dy: float = 1.5        # x-tolerance for voice detection
    trailing_dedup_y_dy: float = 0.25       # y-center dedup tolerance at barline
    trailing_dedup_x_dy: float = 5.0        # x-range for trailing dedup


# ── Accidental Detection ─────────────────────────────────────
@dataclass
class AccidentalConfig:
    search_margin_dy: float = 5.0           # extended margin above/below staff
    ideal_height_dy: float = 2.0            # ideal accidental height
    scale_min: float = 0.7                  # min scaling factor
    scale_max: float = 1.35                 # max scaling factor
    scale_steps: int = 7                    # number of scale steps
    match_threshold_global: float = 0.60    # global detection threshold
    match_threshold_sharp: float = 0.70     # sharps need higher confidence
    match_threshold_per_note: float = 0.45  # per-note search threshold
    nms_radius_dy: float = 0.8              # NMS dedup radius
    assign_left_dy: float = 0.5             # left offset for assignment
    assign_max_dist_dy: float = 3.5         # max horizontal distance
    assign_max_y_dy: float = 1.2            # max vertical distance
    search_left_dy: float = 3.5             # per-note search left
    search_right_dy: float = 0.3            # per-note search right
    search_y_dy: float = 2.0               # per-note search y


# ── Rest Detection ───────────────────────────────────────────
@dataclass
class RestConfig:
    quarter_height_ratio: float = 0.5       # ideal height = staff_height * N
    quarter_scale_factors: List[float] = field(default_factory=lambda: [0.8, 0.9, 1.0, 1.1, 1.2])
    quarter_threshold: float = 0.48         # template match threshold
    eighth_height_ratio: float = 0.35       # ideal height = staff_height * N
    eighth_scale_factors: List[float] = field(default_factory=lambda: [0.85, 1.0, 1.15])
    eighth_threshold: float = 0.60          # template match threshold
    half_threshold: float = 0.55            # stop_2 template threshold
    whole_threshold: float = 0.60           # stop_1 template threshold
    half_whole_scale_factors: List[float] = field(default_factory=lambda: [0.85, 1.0, 1.15])
    nms_distance_dy: float = 2.0            # NMS minimum distance
    min_x_ratio: float = 0.10              # minimum x = img_w * N


# ── Note Assignment & Filtering ──────────────────────────────
@dataclass
class AssignmentConfig:
    ledger_range_dy: float = 5.0            # ledger lines above/below staff
    # Treble false positive thresholds (below_dy, min_score)
    treble_below_1: tuple = (0.5, 0.90)
    treble_below_2: tuple = (2.0, 0.95)
    treble_min_score: float = 0.58
    treble_above_1: tuple = (3.0, 0.98)
    treble_above_2: tuple = (1.0, 0.65)     # modest above → require decent score
    # Bass false positive thresholds
    bass_above_1: tuple = (1.0, 0.80)
    bass_below_1: tuple = (4.0, 0.65)
    bass_min_score: float = 0.60


# ── Segment & Measure ────────────────────────────────────────
@dataclass
class SegmentConfig:
    rest_near_note_dy: float = 1.5          # remove rests within N*dy of notes
    rest_dedup_dy: float = 2.0              # dedup rests within N*dy
    barline_rest_proximity_dy: float = 3.0  # remove rests near barlines
    rest_note_x_dy: float = 1.5             # rest-note x proximity
    rest_note_y_dy: float = 2.0             # rest-note y proximity


# ── Visualization ────────────────────────────────────────────
@dataclass
class VisualizationConfig:
    max_image_width: int = 1200
    padding: int = 40
    font_size: int = 22
    title_font_size: int = 28
    label_font_size: int = 18
    min_font_scale: float = 0.22
    annotation_height_dy: float = 3.5
    staff_gap_dy: float = 2.0
    safety_margin_dy: float = 1.0


# ── Top-level config object ──────────────────────────────────
@dataclass
class Config:
    staff: StaffConfig = field(default_factory=StaffConfig)
    notehead: NoteheadConfig = field(default_factory=NoteheadConfig)
    clef: ClefConfig = field(default_factory=ClefConfig)
    barline: BarlineConfig = field(default_factory=BarlineConfig)
    stem: StemConfig = field(default_factory=StemConfig)
    chord: ChordConfig = field(default_factory=ChordConfig)
    beam: BeamConfig = field(default_factory=BeamConfig)
    duration: DurationConfig = field(default_factory=DurationConfig)
    merge: MergeConfig = field(default_factory=MergeConfig)
    accidental: AccidentalConfig = field(default_factory=AccidentalConfig)
    rest: RestConfig = field(default_factory=RestConfig)
    assignment: AssignmentConfig = field(default_factory=AssignmentConfig)
    segment: SegmentConfig = field(default_factory=SegmentConfig)
    viz: VisualizationConfig = field(default_factory=VisualizationConfig)


CFG = Config()
