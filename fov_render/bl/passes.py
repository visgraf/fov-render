"""Render passes: which are on, and trimming what a .blend file ships with."""
from __future__ import annotations

PASS_ATTR = {
    "combined": "use_pass_combined",
    "z": "use_pass_z",
    "normal": "use_pass_normal",
    "position": "use_pass_position",
    "object_index": "use_pass_object_index",
}


def set_passes(view_layer, passes) -> None:
    """Turn on exactly the named passes among PASS_ATTR (others in PASS_ATTR off)."""
    unknown = set(passes) - set(PASS_ATTR)
    if unknown:
        raise ValueError(f"unknown passes {sorted(unknown)}; expected among {sorted(PASS_ATTR)}")
    for name, attr in PASS_ATTR.items():
        setattr(view_layer, attr, name in passes)


def add_passes(view_layer, passes) -> None:
    """Turn the named passes on and leave the others as they are."""
    for name in passes:
        setattr(view_layer, PASS_ATTR[name], True)


def trim_passes(view_layer, keep, remove_aovs: bool = False) -> dict:
    """Switch off every other pass and light pass the file enables (f3d-vision render_server).
    AOVs are listed; they are removed only with remove_aovs=True (the legacy server left them)."""
    keep_attr = {PASS_ATTR[k] for k in keep}
    off = []
    for name in dir(view_layer):
        if name.startswith("use_pass_") and name not in keep_attr:
            try:
                if getattr(view_layer, name) is True:
                    setattr(view_layer, name, False)
                    off.append(name)
            except (AttributeError, TypeError):
                pass
    cyc = getattr(view_layer, "cycles", None)
    for name in (dir(cyc) if cyc is not None else []):
        if name.startswith("use_pass_") or name == "denoising_store_passes":
            try:
                if getattr(cyc, name) is True:
                    setattr(cyc, name, False)
                    off.append("cycles." + name)
            except (AttributeError, TypeError):
                pass
    aovs = list(getattr(view_layer, "aovs", []))
    for aov in aovs:
        off.append("aov:" + aov.name)
    if remove_aovs:
        for aov in aovs:
            view_layer.aovs.remove(aov)
    view_layer.update()
    return {"disabled": off, "kept": sorted(k for k in keep if getattr(view_layer, PASS_ATTR[k]))}


def enabled(view_layer) -> list[str]:
    return sorted(k for k, a in PASS_ATTR.items() if getattr(view_layer, a))
