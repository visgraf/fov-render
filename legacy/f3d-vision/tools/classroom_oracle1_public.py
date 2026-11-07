"""Public contract for Classroom-Oracle-1.

This experiment changes the measurement source, not the controller. Blender truth
replaces the local stereo matcher only for the current tangent observation.  It
may also enumerate controller-domain visible instances and provide one seed gaze
per instance.  No unobserved geometry, future visibility, completion label or
next-gaze choice is available to the control loop.
"""
from __future__ import annotations

import hashlib
import json

SPEC_ID = "Classroom-Oracle-1-v1"
PARENT_COMMIT = "2897d31"
SCENE = "scenes/classroom/classroom_eye.blend"
DEFAULT_PROFILE = "full"
DEFAULT_DEVICE = "OPTIX"
MAX_OBJECT_FIXATIONS = 24  # engineering watchdog only; never a scientific PASS/FAIL gate
SEED_SCAN_STEP_DEG = 0.25
CYCLOPEAN_GRID_DEG = 0.10
MIN_INITIAL_TARGET_POINTS = 100  # inherited fsg3_surface_map initialization precondition

# The metric association rule is inherited unchanged from FSG3/FSG6f.
FUSION = {
    "association_radius_m": 0.012,
    "hash_cell_m": 0.012,
}

ORACLE_CONTRACT = {
    "local_truth_only": True,
    "current_tangent_observation_only": True,
    "binocular_visibility_required": True,
    "right_eye_same_instance_required": True,
    "sgbm_range_bound_inherited": False,
    "global_geometry_hidden_from_control": True,
    "future_visibility_hidden_from_control": True,
    "blender_selects_autonomous_fixations": False,
    "blender_declares_completion": False,
    "hole_fill": False,
    "mesh_completion": False,
    "foreground_background_decomposition": False,
    "seed_oracle": "instance enumeration + exactly one seed direction per visible instance",
}

CONTROL_CONTRACT = {
    "local_controller": "frozen FSG6f via multiobject2c_policy adapter",
    "persistent_metric_memory": "fsg3_surface_map; fixed head frame",
    "epistemic_handoff": "cyclopean NEVER_OBSERVED exterior shoreline; deepest border-distance first",
    "empty_look_semantics": "record gaze and binocular evidence; fuse zero; continue",
    "scientific_completion": "FSG6f no_frontier followed by no eligible cyclopean epistemic fixation",
    "watchdog": MAX_OBJECT_FIXATIONS,
}

BENCHMARK_CONTRACT = {
    "save_every_controller_selected_pair": True,
    "save_raw_multilayer_exr": True,
    "save_rectified_rgb_pair": True,
    "save_oracle_patch": True,
    "open_loop_replay": "same gaze trajectory and same saved image pairs for every later matcher",
}

PUBLIC_SPEC = {
    "id": SPEC_ID,
    "parent_commit": PARENT_COMMIT,
    "scene": SCENE,
    "default_profile": DEFAULT_PROFILE,
    "max_object_fixations": MAX_OBJECT_FIXATIONS,
    "seed_scan_step_deg": SEED_SCAN_STEP_DEG,
    "cyclopean_grid_deg": CYCLOPEAN_GRID_DEG,
    "fusion": FUSION,
    "oracle": ORACLE_CONTRACT,
    "control": CONTROL_CONTRACT,
    "benchmark": BENCHMARK_CONTRACT,
}


def public_digest() -> str:
    payload = json.dumps(PUBLIC_SPEC, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def self_test() -> list[str]:
    fails: list[str] = []
    if FUSION != {"association_radius_m": 0.012, "hash_cell_m": 0.012}:
        fails.append("fusion rule is not the frozen 12 mm rule")
    if MAX_OBJECT_FIXATIONS != 24:
        fails.append("object watchdog is not 24")
    if ORACLE_CONTRACT["sgbm_range_bound_inherited"]:
        fails.append("oracle must not inherit the SGBM range bound")
    if not ORACLE_CONTRACT["global_geometry_hidden_from_control"]:
        fails.append("global geometry must be hidden from the controller")
    if ORACLE_CONTRACT["foreground_background_decomposition"]:
        fails.append("foreground/background decomposition is forbidden")
    return fails


if __name__ == "__main__":
    bad = self_test()
    for item in bad:
        print("[classroom-oracle1-public] FAIL", item)
    print("[classroom-oracle1-public] self-test", "FAILED" if bad else "PASS", public_digest())
    raise SystemExit(bool(bad))
