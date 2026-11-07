"""Truth-free public contract for FSG6f persistent open-frontier validation.

FSG6e established persistent three-state frontier memory but exposed a final
aggregation gap: a gaze candidate could survive when only a small minority of
its aligned raw frontier surfels remained OPEN.  FSG6f changes only the
candidate-level interpretation of those already-existing states.

For each candidate direction, aligned raw frontier support is partitioned into
OPEN, MAP_RESOLVED and BOUNDARY_RESOLVED by the unchanged FSG6e classifier.
The candidate remains an exploration action only when OPEN support is a strict
majority of that aligned raw support, in addition to the already-frozen minimum
of eight OPEN surfels.  A tie is resolved, not open.  This is a state consensus
rule, not a fitted numerical threshold: OPEN must simply outnumber the two
resolved states together.  The FSG6d projected-frontier corridor, ranking,
measurement instrument, fusion, lattice, budget and numerical gates remain
unchanged.
"""
from __future__ import annotations
import hashlib
import json

SPEC_ID = "FSG6f-candidate-frontier-consensus-v1"
INSTRUMENT_ID = "FSG1-HDR-SGBM-one-original-update-original-validity-v1"
OBJECT_ID = 141
BACKGROUND_ID = 142
FIXTURES = ("consensus_up_right", "consensus_down_left")
SEEDS = (1237, 1291)
DEFAULT_SPP = {"small": 64, "full": 256}
SEED_GAZE_DEG = {
    "consensus_up_right": (-8.0, -7.0),
    "consensus_down_left": (+8.0, +7.0),
}
VERGENCE_DISTANCE_M = 2.10
MAX_BUDGET_FIXATIONS = 6

FUSION = {
    "association_radius_m": 0.012,
    "hash_cell_m": 0.012,
}

# Byte-for-byte numerical values inherited from FSG6a-d. Do not tune in FSG6f.
SURFACE_FRONTIER = {
    "component_step_deg": 5.0,
    "yaw_min_deg": -25.0,
    "yaw_max_deg": +25.0,
    "pitch_min_deg": -20.0,
    "pitch_max_deg": +20.0,
    "edge_band_fraction": 0.04,
    "edge_object_fraction_min": 0.15,
    "voxel_m": 0.025,
    "neighbour_radius_m": 0.065,
    "minimum_neighbours": 6,
    "tangent_asymmetry_min": 0.18,
    "lookahead_m": 0.12,
    "current_view_margin_deg": 1.0,
    "minimum_candidate_frontier_support": 8,
    "alignment_cos_min": 0.50,
    "map_extent_quantile": 0.01,
}

TARGETS = {
    "active_fixation_count_min": 4,
    "active_fixation_count_max": MAX_BUDGET_FIXATIONS,
    "patch_object_coverage_min": 0.90,
    "minimum_matched_points_each": 5000,
    "overlap_median_distance_max_m": 0.010,
    "overlap_p95_distance_max_m": 0.025,
    "map_surface_median_max_m": 0.010,
    "map_surface_p95_max_m": 0.030,
    "supported_signed_radial_bias_abs_max_m": 0.0075,
    "supported_surfels_min": 5000,
    "final_truth_coverage_min": 0.90,
    "truth_coverage_gain_over_seed_min": 0.35,
    "coverage_drop_tolerance": 0.005,
    "pitch_span_min_deg": 10.0,
}

PUBLIC_SPEC = {
    "id": SPEC_ID,
    "instrument": INSTRUMENT_ID,
    "object_id": OBJECT_ID,
    "background_id": BACKGROUND_ID,
    "fixtures": list(FIXTURES),
    "seeds": list(SEEDS),
    "spp": DEFAULT_SPP,
    "seed_gaze_deg": {k: list(v) for k, v in SEED_GAZE_DEG.items()},
    "prescribed_vergence_distance_m": VERGENCE_DISTANCE_M,
    "maximum_budget_fixations": MAX_BUDGET_FIXATIONS,
    "fusion": FUSION,
    "surface_frontier": SURFACE_FRONTIER,
    "targets": TARGETS,
    "fixed_head_frame": True,
    "oracle_instance_segmentation": True,
    "raw_frontier": "unchanged FSG6d local PCA/tangent asymmetry",
    "frontier_state": "unchanged FSG6e MAP_RESOLVED / BOUNDARY_RESOLVED / OPEN classification",
    "candidate_state_consensus": "candidate is exploration-open only when aligned OPEN support strictly outnumbers aligned MAP_RESOLVED plus BOUNDARY_RESOLVED support, while retaining the frozen >=8 OPEN support minimum",
    "strict_majority_has_no_fitted_threshold": True,
    "map_resolution_radius_is_frozen_fusion_association_radius": True,
    "boundary_resolution_uses_completed_binocular_history": True,
    "boundary_resolution_threshold_is_frozen_continuation_threshold": True,
    "boundary_patch_scale_derived_from_frozen_edge_band_fraction": True,
    "continuation_veto": "unchanged FSG6d binocular projected 3D-frontier exit corridor in the candidate forward image sector",
    "continuation_threshold_unchanged_from_fsg6d": True,
    "legacy_edge_band_fraction_reused_as_corridor_width": True,
    "no_new_frontier_state_threshold": True,
    "no_new_candidate_consensus_threshold": True,
    "no_icp": True,
    "no_hole_fill": True,
    "no_mesh_reconstruction": True,
    "fsg3_surface_map_unchanged": True,
    "frontier_truth_hidden_from_policy": True,
    "per_fixation_novelty_and_gain_are_descriptive_only": True,
}


def public_digest() -> str:
    return hashlib.sha256(json.dumps(PUBLIC_SPEC, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def fixture_index(fixture: str) -> int:
    if fixture not in FIXTURES:
        raise ValueError(f"unknown FSG6f fixture {fixture}")
    return FIXTURES.index(fixture)


def render_seed(fixture: str, seed: int, yaw_deg: float, pitch_deg: float, eye_id: int) -> int:
    """Deterministic physical-view seed independent of loop step."""
    if seed not in SEEDS:
        raise ValueError("seed not in frozen FSG6f schedule")
    if eye_id not in (0, 1):
        raise ValueError("eye_id must be 0 or 1")
    p = SURFACE_FRONTIER; y = float(yaw_deg); q = float(pitch_deg)
    if not (p["yaw_min_deg"] - 1e-9 <= y <= p["yaw_max_deg"] + 1e-9):
        raise ValueError("yaw outside frozen FSG6f range")
    if not (p["pitch_min_deg"] - 1e-9 <= q <= p["pitch_max_deg"] + 1e-9):
        raise ValueError("pitch outside frozen FSG6f range")
    iy = int(round((y + 30.0) * 10.0)); ip = int(round((q + 25.0) * 10.0))
    if abs(y - (iy / 10.0 - 30.0)) > 1e-8 or abs(q - (ip / 10.0 - 25.0)) > 1e-8:
        raise ValueError("gaze must lie on the 0.1-degree seed lattice")
    return 1_000_000 * int(seed) + 700_000 * fixture_index(fixture) + 1000 * iy + 2 * ip + eye_id
