"""Active Bootstrap-1a: declared constants, the frozen NB1c action and the inherited binocular instrument.

Contract: docs/active-bootstrap/ab1a-first-natural-stereo-look-contract.md, sections 3-9 and 15.  Plain Python
(standard library only) so that both interpreters can import it: the Blender-side acquisition and the host-side
matcher, evaluation and figures.  The checker keeps its own literal copies.
"""
from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent
REPO = TOOLS.parent
SHARED = Path("/home/lvelho/rd/f3d-vision")
BASE_COMMIT = "509c341ff60994ff0bf726b081be1f5c5c107b73"   # accepted main after the NB1c closure
EXPERIMENT = "Active Bootstrap-1a: First Natural Stereo Look"
CONTRACT = "docs/active-bootstrap/ab1a-first-natural-stereo-look-contract.md"
MARKER = "ACTIVE_BOOTSTRAP1A_FIRST_NATURAL_STEREO_LOOK_COMPLETE"

TRUTH_ORACLE, TRUTH_DERIVED, TRUTH_REFERENCE = "ORACLE INPUT", "DERIVED", "REFERENCE / EVALUATION"

# ---- section 3: the frozen action (accepted NB1c, read-only)
NB1C_RUN = SHARED / "previews/natural-bootstrap-1c-rgb-candidate-gaze"
NB1C_FREEZE = (NB1C_RUN / "selection/rgb-gaze-freeze.json",
               "87a3bab01f55051316ac1eb45b48f394211f4c7a3a317fd0e61c0e52c36f9b37")
NB1C_CANDIDATES = (NB1C_RUN / "selection/candidate-gazes.json",
                   "8041b954f7b63d97f060a41aaca5db1ec6e0c2295381523026e98d11f8a5b621")
NB1C_MANIFEST = (NB1C_RUN / "manifest.json", "524f8fad71ccb8fd9359fa3221f4de6be3af40be8f3e5fa7cddeb482acedd87e")
GAZE_RANK, GAZE_ROW, GAZE_COL = 1, 164, 513
GAZE_YAW_DEG, GAZE_PITCH_DEG = 76.75, 7.75
ACTION_SOURCE = "frozen NB1c RGB gaze #1"

# The head frame of the gaze: the Classroom EYE, as recorded by the accepted Controller-01 bootstrap (Breadth-1 used it).
HEAD_POSE_SOURCE = (SHARED / "previews/controller-01-full/bootstrap/seeds.json",
                    "6ef833196de2462370ba4b2a2a8eed3fbde7f0a61e5aa6cbaf95f66947134c4f")
ACCEPTED_CATALOG = (SHARED / "previews/controller-01-full/bootstrap/instance_catalog.json",
                    "be26594254b03fbda7216ac1a942a5c800cf4bdd8f79751a2b8c22d6bfb4c7ae")
EYE_POSE_TOL = 1e-6
CALIBRATION_TOL = 1e-9     # Blender-built vs host-planned calibration (two interpreters' numpy builds)
BLEND = "scenes/classroom/classroom_eye.blend"
BLEND_SHA256 = "dca66a3257b909aef00b0331652c55f62a93a8dff0665d070fba7efa436953cc"
BLENDER = "blender"

# ---- section 4: the inherited instrument
PROFILE = "full"
CORE_FOV_DEG = 12.0
CORE_SIZE = 256
RAW_SIZE = 640
IPD_M = 0.063
VERGENCE_M = 2.10
TANGENT_FRAME = "baseline_projected"
Z_RECT_M = (0.75, 4.5)
DEVICE = "OPTIX"
SPP = 256
SEEDS = {"L": 2111, "R": 2112}
BASELINE_H = (1.0, 0.0, 0.0)
PASSES = ("Combined", "Position", "Object Index")

# Accepted sources reused read-only (sha256 at the AB1a base); a mismatch is a STOP.
SOURCE_PINS = {
    "tools/fsg_geometry.py": "ae3779bc5cbe8f9702227ae9f123c6be1abe394a0a3c617627f38f171c556c54",
    "tools/fsg_stereo.py": "faebf0f1b3acbfde60b08830a57213bf29c708c3aa274e493ade0042eb0545e9",
    "tools/classroom_oracle1_render.py": "ebeac7363a54a022c604161a3716a220fcf2aa7feea3792a68568699ed28a55a",
    "tools/classroom_oracle1_matcher.py": "d213c2c7da68a2969b54a095fc7d953d7d16b21f5690cf2fd152197265884afa",
    "tools/classroom_oracle1_public.py": "7b85e59a6e397ed69d3eaec8a6b3790639a6c6ffce596fd0d8d1664c738452bb",
    "tools/fsg6f_public.py": "c79f58c9b51f33d4463f2bcfcaa339d79cf20a962b44de9c6cc721cf157cd2fa",
    "tools/render_foveated.py": "6f4d6c20b3fa22a74cbf71552374410545b09aeb07ae0b5867cf95b7a264fedf",
    "tools/bl_common.py": "aa7a56e8cd4988cbe7fc92a57fde68e62740688b6ccfee718e01a8e52ba10370",
    "tools/exr_lite.py": "77286773ee52d17f45373a9c0a0358b197b18672c4cff4654e00bc69e828ef20",
    "tools/natural_bootstrap/nb1a_guard.py": "29ebca57be9e977def7a94ea45a7b0cff0f22a8b1d86b40f740f132491ebe99d",
}

# ---- section 7: the inherited matcher constants (fsg_stereo must agree; STOP otherwise)
MATCHER = {
    "name": "AB1a natural RGB-only SGBM_3WAY + bounded photometric refinement (accepted FSG mechanics, no identity guard)",
    "block_size": 5, "lr_tolerance_px": 1.0, "uniqueness_ratio": 10, "minimum_local_std_u8": 0.5,
    "sgbm": {"mode": "SGBM_3WAY", "P1": "8 * block_size^2", "P2": "32 * block_size^2", "disp12MaxDiff": -1,
             "preFilterCap": 31, "speckleWindowSize": 0, "speckleRange": 1, "minDisparity": 0,
             "numDisparities": "ceil((ceil(|P2[0,3]| / z_rect_min) + 4) / 16) * 16 (accepted rectification)"},
    "refinement": {"iterations": 3, "max_step_px": 0.5, "max_total_px": 0.75,
                   "statement": "accepted fsg_stereo.refine_disparity on SGBM-valid and calibration-supported pixels"},
    "z_rect_m": list(Z_RECT_M),
    "support": "fsg_stereo.support_mask(calibration, rectification, side) only (calibration-derived)",
    "identity_guard": None,
    "validity": ["term_sgbm_left", "term_right_valid", "term_support_left", "term_inside_raster", "term_lr",
                 "term_texture", "term_finite", "term_z_range", "term_roi"],
    "inputs": ["acquisition/calibration.json", "acquisition/rgb-observation.npz (exactly rgb_L, rgb_R)"],
}
TERMS = tuple(MATCHER["validity"])
RGB_KEYS = ("rgb_L", "rgb_R")
REFERENCE_KEYS = ("instance_L", "instance_R", "position_w_L", "position_w_R")

# ---- section 9: evaluation (declared before the canonical acquisition)
QUANTILES = {"median": 0.5, "p90": 0.90, "p95": 0.95, "p99": 0.99, "max": 1.0}
SUMMARY_QUANTILES = {"min": 0.0, "median": 0.5, "p90": 0.90, "p95": 0.95, "max": 1.0}
ERROR_FRACTIONS_M = (0.010, 0.025, 0.050)
MIN_INSTANCE_OVERLAP = 50
CAMERA_MODEL_TOL_PX = 0.5 + 1e-3

# ---- section 15: Blender rehearsal (synthetic factory-startup scene only)
REHEARSAL_SPP = 64
REHEARSAL_GAZES = {"forward": (0.0, 0.0), "gaze1-geometry": (GAZE_YAW_DEG, GAZE_PITCH_DEG)}
REHEARSAL_ROOM_HALF_M = 3.0
REHEARSAL_PLANE_M = 2.0
REHEARSAL_MIN_VALID = 0.50
REHEARSAL_MAX_MEDIAN_RADIAL_M = 0.010

# ---- section 15: host-side synthetic known-answer thresholds (synthetic data only)
SYN_PLANE_Z_M = 2.0
SYN_MIN_VALID = 0.90
SYN_MAX_MEDIAN_DISPARITY_PX = 0.05
SYN_MAX_P95_DISPARITY_PX = 0.50
SYN_MAX_MEDIAN_Z_REL = 0.005
SYN_MAX_MEDIAN_PLANE_M = 0.005
SYN_SUPPORT_VERGENCE_M = 0.15

# ---- run layout (section 13)
RUN_DEFAULT = SHARED / "previews/active-bootstrap/ab1a-first-natural-stereo-look"
VIS_DEFAULT = SHARED / "visuals/active-bootstrap/ab1a-first-natural-stereo-look"
FROZEN_MEASUREMENT = ["source/nb1c-action-manifest.json", "prelook/planned-calibration.json",
                      "prelook/prelook-geometry.json", "acquisition/calibration.json",
                      "acquisition/rgb-observation.npz", "acquisition/acquisition.json",
                      "measurement/stereo-result.npz", "measurement/stereo-summary.json",
                      "measurement/measurement-opened-files.json"]
MATCHER_CODE = ["tools/active_bootstrap/ab1a_stereo.py", "tools/active_bootstrap/ab1a_spec.py", "tools/fsg_stereo.py",
                "tools/fsg_geometry.py", "tools/natural_bootstrap/nb1a_guard.py"]


def gaze_cell_centre(row: int, col: int) -> tuple[float, float]:
    """The NB1c / NB1a cell-centre convention (0.5-degree cells)."""
    return -180.0 + 0.5 * (col + 0.5), 90.0 - 0.5 * (row + 0.5)
