# Fixtures

Byte-identical copies of four fov-3d-vision files the checks use as tools, not as code fov_render
replaced (issue #16): `make_calib_room.py` and `make_hdri_scene.py` build the test scenes,
`bl_common.py` is their helper, and `check_foveated.py` is fov-3d-vision's independent checker of a
foveated render (gates 1.1, 1.C and the scene gate's S3). They come from fov-3d-vision at the commit
in `../known/pins.json`, and gate 0 checks their sha256 against it. Never edit them.
