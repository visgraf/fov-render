"""The two configurations every sensor and check is built on.

`small` is for building and debugging; `full` is for the number that goes in a report.
Geometry, record and checks are identical between them; only spacing and sample counts
differ. Values carried over from fov-3d-vision `tools/bl_common.PROFILES` and f3d-vision
`tools/fsg_geometry.CORE_SIZE`.
"""
from __future__ import annotations

PROFILES: dict[str, dict] = {
    "small": {"s0_deg": 0.1, "core_px": 128, "fix_spp": 64, "ref_width": 3600, "ref_spp": 1024},
    "full": {"s0_deg": 0.05, "core_px": 256, "fix_spp": 256, "ref_width": 7200, "ref_spp": 8192},
}

DEFAULT_IPD_M = 0.063
DEFAULT_SEEDS = {"L": 2111, "R": 2112, "C": 0}   # AB1a declared constants; C is the cyclopean eye
EYES = ("L", "R", "C")


def profile(name: str) -> dict:
    if name not in PROFILES:
        raise ValueError(f"unknown profile {name!r}; expected one of {sorted(PROFILES)}")
    return PROFILES[name]
