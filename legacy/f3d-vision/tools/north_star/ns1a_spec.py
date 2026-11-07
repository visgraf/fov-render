"""North Star-1a: declared constants, pinned sources, the frozen NB1c action and the run layout.

Contract: docs/north-star/ns1a-perfect-bootstrap-round-contract.md.  Standard library only, so both interpreters
(Blender's and the host .venv) can import it.  The checker keeps its own literal copies.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent
REPO = TOOLS.parent
SHARED = Path("/home/lvelho/rd/f3d-vision")
BASE_COMMIT = "dfe16269eb4fcf71ee6a97bab0e8e6126970f1b0"      # accepted main: AB1d3 accepted + return to the North Star
AB1D3_ACCEPTANCE = "9e0bae487012164921343925b04569111bdc9296"
CONTRACT_COMMIT = "748411c01daaef010905104cf40ed82c917e5f3d"
CANONICAL_REMOTE = "visgraf/f3d-vision"
EXPERIMENT = "North Star-1a: RGB Bootstrap -> Perfect Local Measurement -> Persistent Entity Seeds"
CONTRACT = "docs/north-star/ns1a-perfect-bootstrap-round-contract.md"
MARKER = "NORTH_STAR1A_PERFECT_BOOTSTRAP_ROUND_COMPLETE"

TRUTH_ORACLE, TRUTH_DERIVED, TRUTH_REFERENCE = "ORACLE INPUT", "DERIVED", "REFERENCE / EVALUATION"
LABEL_RGB = "RGB-ONLY SELECTION"
LABEL_ORACLE_CORR = "ORACLE CORRESPONDENCE"
LABEL_GEOMETRY = "DERIVED SPHERICAL GEOMETRY"
LABEL_SEGMENTATION = "ORACLE SEGMENTATION AID"
UNASSIGNED = "UNASSIGNED / NON-CATALOG ORACLE GEOMETRY"

# ---- section 4a: the frozen action (accepted NB1c, read-only)
NB1C_RUN = SHARED / "previews/natural-bootstrap-1c-rgb-candidate-gaze"
NB1C_FREEZE = (NB1C_RUN / "selection/rgb-gaze-freeze.json",
               "87a3bab01f55051316ac1eb45b48f394211f4c7a3a317fd0e61c0e52c36f9b37")
NB1C_CANDIDATES = (NB1C_RUN / "selection/candidate-gazes.json",
                   "8041b954f7b63d97f060a41aaca5db1ec6e0c2295381523026e98d11f8a5b621")
NB1C_MANIFEST = (NB1C_RUN / "manifest.json", "524f8fad71ccb8fd9359fa3221f4de6be3af40be8f3e5fa7cddeb482acedd87e")
GAZES = ((1, 164, 513, 76.75, 7.75), (2, 28, 354, -2.75, 75.75), (3, 231, 510, 75.25, -25.75),
         (4, 303, 436, 38.25, -61.75), (5, 110, 719, 179.75, 34.75), (6, 122, 47, -156.25, 28.75))
RANKS = tuple(g[0] for g in GAZES)
GRID_CONVENTION = ("yaw = -180 + 0.5 (col + 0.5), pitch = 90 - 0.5 (row + 0.5); "
                   "d = (sin yaw cos pitch, sin pitch, -cos yaw cos pitch)")
ACTION_SOURCE = "frozen NB1c RGB gaze (rank {rank} of 6, frozen order)"
# the coarse 360 RGB attention input of NB1c (display only, panel A)
NB1A_RGB = (SHARED / "previews/natural-bootstrap-1a-range-connectivity/input/rgb-sensory.npz",
            "cd2600e678fe7f33471e50bee5443128dd5470cc2b380eb7b34a22d3e4ad2c3e")

# ---- section 4b: the fixed head (the accepted AB1a calibration; NOT the Controller-01 seeds file)
AB1A_RUN = SHARED / "previews/active-bootstrap/ab1a-first-natural-stereo-look"
HEAD_POSE_SOURCE = (AB1A_RUN / "acquisition/calibration.json",
                    "9960c86e0fda1c002ae67e702d34ea3399bff3180d61bf71ac535563f8fc3913")
AB1A_ACQUISITION = (AB1A_RUN / "acquisition/acquisition.json",
                    "ba316b11edc400b47a4cab467312ecf717551c11b8cb75b27a6fe7c997a16a41")
AB1D2_SETTINGS_SOURCE = (SHARED / "previews/active-bootstrap/ab1d2-4096spp-observation-quality/observations/gaze-1/"
                                  "acquisition/acquisition.json",
                         "7c023b1f382514716893ca1a4d3009a6ae1102e327b3de00312206e3d9f37eff")
EYE_POSE_TOL = 1e-6             # accepted (ab1a)
CALIBRATION_TOL = 1e-9          # accepted (ab1a / ab1c)
BLEND = "scenes/classroom/classroom_eye.blend"
BLEND_SHA256 = "dca66a3257b909aef00b0331652c55f62a93a8dff0665d070fba7efa436953cc"
BLENDER = "blender"

# ---- section 5: the inherited instrument and the observation quality
PROFILE = "full"
CORE_FOV_DEG = 12.0
RAW_SIZE = 640
CORE_SIZE = 256
CORE_ORIGIN = (RAW_SIZE - CORE_SIZE) // 2     # 192: raw x, y = 192..447
IPD_M = 0.063
VERGENCE_M = 2.10
TANGENT_FRAME = "baseline_projected"
DEVICE = "OPTIX"
SPP = 4096
SEEDS = {"L": 2111, "R": 2112}
RENDER_BUDGET_S = 1800.0
EXR_SAMPLES_ATTR = "cycles.interior.samples"   # the Classroom view layer is `interior`
RGB_KEYS = ("rgb_L", "rgb_R")
REFERENCE_KEYS = ("instance_L", "instance_R", "position_w_L", "position_w_R")
SEGMENTATION_MEMBERS = ("instance_L",)         # the ONLY reference member the segmentation aid reads

# ---- section 11: the accepted persistent-map machinery (verified against the accepted sources by `source`)
MIN_INITIAL_TARGET_POINTS = 100
ASSOCIATION_RADIUS_M = 0.012
HASH_CELL_M = 0.012
IDEMPOTENCE_ATOL = 1e-10                       # the Controller-01 replay test (recorded beside the exact test)
PATCH_ID = "nb1c_gaze_{rank:02d}"
ACTIONS = ("INITIALIZED", "FUSED", "SEEN_BUT_NOT_INITIALIZED", "RETAINED_NOT_FUSED")

# ---- section 8: the core class map (ORACLE AID; never read by geometry)
CLASS_NO_HIT, CLASS_INSTANCE0, CLASS_NOT_VISIBLE, CLASS_CORRESPONDENCE = 0, 1, 2, 3
CLASS_NAMES = {0: "no finite geometric hit", 1: "hit with instance 0 (unassigned / non-catalog)",
               2: "positive instance, not right-visible", 3: "perfect correspondence"}

# ---- section 13: post-freeze reference (REFERENCE / EVALUATION; opened only by `evaluate`)
ACCEPTED_CATALOG = (SHARED / "previews/controller-01-full/bootstrap/instance_catalog.json",
                    "be26594254b03fbda7216ac1a942a5c800cf4bdd8f79751a2b8c22d6bfb4c7ae")
C01_SEEDS = (SHARED / "previews/controller-01-full/bootstrap/seeds.json",
             "6ef833196de2462370ba4b2a2a8eed3fbde7f0a61e5aa6cbaf95f66947134c4f")
BREADTH1_OBJECTS = (SHARED / "previews/breadth-1-classroom-234-spherical-glance/object-stats.json",
                    "2f8aa7bee3abf88bd4a93316ad13babd6419f551536d99e71db6b63dce175a02")
AB1B_PRODUCT = (SHARED / "previews/active-bootstrap/ab1b-spherical-epipolar-geometry/oracle/oracle-correspondences.npz",
                "4d26ff328cafb9ff1b9f54e3a1a7309528de87de1e26f41884516d52465e8fc0")
FORBIDDEN_BEFORE_FREEZE = (
    "instance-catalog.json", "instance_catalog.json", "seeds.json", "object-stats.json",
    "natural-bootstrap-1a-range-connectivity/discovery", "natural-bootstrap-1b-foveal-serviceability",
    "breadth-1-classroom-234-spherical-glance", "controller-01-full", "controller-02-classroom-replay",
)
QUANTILES = {"min": 0.0, "p05": 0.05, "median": 0.5, "p95": 0.95, "p99": 0.99, "max": 1.0}

# ---- section 15: synthetic known-answer tolerances (analytic data only)
SYN_PROJECTION_PX = 1e-3       # analytic uv_R vs the oracle's (float32 Position in the synthetic reference)
SYN_E2E_MEDIAN_M = 1e-4
SYN_E2E_MAX_M = 5e-4

# ---- accepted code reused read-only (sha256 at the base); a mismatch is a STOP
SOURCE_PINS = {
    "tools/active_bootstrap/ab1b_oracle.py": "51e2ee556a4a3576d00cd99255793212ad5efcb50b1e15ed292740d8c24aa231",
    "tools/active_bootstrap/ab1b_geometry.py": "a125dd9cc666120219e0421a09a380353d2304ed0f09681a1655962444572538",
    "tools/active_bootstrap/ab1b_spec.py": "1634e455a61879934545a1fa634a72cbb2b8ca74fd83a6ba3a977412be0e497d",
    "tools/fsg_geometry.py": "ae3779bc5cbe8f9702227ae9f123c6be1abe394a0a3c617627f38f171c556c54",
    "tools/natural_bootstrap/nb1a_guard.py": "29ebca57be9e977def7a94ea45a7b0cff0f22a8b1d86b40f740f132491ebe99d",
    "tools/fsg3_surface_map.py": "1b9dbeb873105ec9bd9aef9680f67ed18db9a7387d6871d6ed8e6ab58bcb1e01",
    "fov3d/reconstruction/surface_map.py": "51e5da5262a74db782930562b2a8026513996e5512a178b05600d770a915f983",
    "fov3d/reconstruction/association.py": "fe8cf0e22f2a0b7d252f9110bf42e07f168c6ba85b341ab9cef20853308ef135",
    "fov3d/_compat.py": "014d1fb8c9cfc4c80fbf015c698e8fc706a16b79a0eda2b916e8727d9e61a870",
    "tools/classroom_oracle1_public.py": "7b85e59a6e397ed69d3eaec8a6b3790639a6c6ffce596fd0d8d1664c738452bb",
    "fov3d/experiments/classroom_oracle/config.py": "8231c7f3b7a1b19108eaeb1fa6294cad0dbcf47f61b12025250e3ef7ac24aa07",
    "tools/active_bootstrap/ab1a_render.py": "ae96164abae9a5241cee3e496293879be43ac96e77697d44d4e2bc956d51e01b",
    "tools/active_bootstrap/ab1a_spec.py": "24ab5df947b6aaabc46737d3fd015d43802405f94ff1b01595c650bf5ab24a62",
    "tools/classroom_oracle1_render.py": "ebeac7363a54a022c604161a3716a220fcf2aa7feea3792a68568699ed28a55a",
    "tools/render_foveated.py": "6f4d6c20b3fa22a74cbf71552374410545b09aeb07ae0b5867cf95b7a264fedf",
    "tools/bl_common.py": "aa7a56e8cd4988cbe7fc92a57fde68e62740688b6ccfee718e01a8e52ba10370",
    "tools/exr_lite.py": "77286773ee52d17f45373a9c0a0358b197b18672c4cff4654e00bc69e828ef20",
    "tools/visual_language/style.py": "c379a2a31c9112d22bbe8989324d2ff74608c669fe84d58905509b4d62264053",
}
NS1A_TOOLS = ("ns1a_spec.py", "ns1a_render.py", "ns1a_core.py", "ns1a_run.py", "ns1a_synthetic.py", "ns1a_visuals.py",
              "check_ns1a.py", "check_ns1a_corruptions.py")

# ---- section 19: run layout and commands
RUN_DEFAULT = SHARED / "previews/north-star/ns1a-perfect-bootstrap-round"
VIS_DEFAULT = SHARED / "visuals/north-star/ns1a-perfect-bootstrap-round"
COMMANDS = ("source", "synthetic", "preflight", "acquire", "freeze-observations", "perfect-correspondence",
            "freeze-correspondence", "spherical-geometry", "freeze-geometry", "local-oracle-segmentation",
            "persistent-seed-construction", "freeze-seed-set", "evaluate", "visualize")
CANONICAL = COMMANDS[:-1]
FIGURES = ("overview.png", "bootstrap-progression.png", "entity-seeds-3d.png")


def rank_dir(rank: int) -> str:
    return f"rank-{int(rank):02d}"


def acq_rel(rank: int, name: str) -> str:
    return f"observations/{rank_dir(rank)}/acquisition/{name}"


def aid_rel(rank: int, name: str) -> str:
    return f"observations/{rank_dir(rank)}/oracle_aid/{name}"


CATALOG_REL = "observations/evaluation_only/instance-catalog.json"
ACQ_RUN_REL = "observations/acquisition-run.json"
OBS_ACQ_FILES = ("calibration.json", "rgb-observation.npz", "acquisition.json")
OBS_AID_FILES = ("raw_L.exr", "raw_R.exr", "reference-observation.npz")


def gaze_yaw_pitch(row: int, col: int) -> tuple[float, float]:
    """The NB1c / NB1a cell-centre convention (0.5-degree cells)."""
    return -180.0 + 0.5 * (col + 0.5), 90.0 - 0.5 * (row + 0.5)


def patch_id(rank: int) -> str:
    return PATCH_ID.format(rank=int(rank))


def config_sha256(cfg) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


PERSISTENCE = {
    "map": "fov3d.reconstruction.surface_map (facade of the sealed tools/fsg3_surface_map.py): initialize, fuse",
    "min_points": MIN_INITIAL_TARGET_POINTS, "association_radius_m": ASSOCIATION_RADIUS_M, "hash_cell_m": HASH_CELL_M,
    "patch_id": PATCH_ID, "order": "frozen gaze rank 1..6; within a gaze, positive ids ascending",
    "patch": "xyz_h = frozen spherical P_epi (valid_epi); rgb = raw left-core linear RGB at uv_L; "
             "instance_id = local oracle id at uv_L",
    "rule": "no map: n >= min -> initialize, else SEEN_BUT_NOT_INITIALIZED; map: n >= min -> fuse + exact replay "
            "idempotence, else RETAINED_NOT_FUSED",
    "identity": "ORACLE SEGMENTATION AID: positive Blender instance id (local Object Index at uv_L); cross-look "
                "persistence by the same id is also an oracle aid",
    "instance_0": "never an entity; reported as " + UNASSIGNED,
}
