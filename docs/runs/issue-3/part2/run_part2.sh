#!/bin/bash
# Issue #3 part 2, Code's runs: Chat's steps 1 and 2 (issue comment of 2026-11-08 01:03), with wall times.
#
#     bash docs/runs/issue-3/part2/run_part2.sh            (from the fov-render checkout; batch, ~30 min)
#
# Step 1, exact, CPU, calibration room, 50 fixations: the legacy loop (fov-3d-vision's active_loop.py,
# unchanged), the loop on fov_render (--mode exact), and the control (IPD + 1 ULP); then
# compare_loop_runs on legacy/exact (must be IDENTICAL) and legacy/control (must be DIFFERENT).
# Step 2, statistical, OPTIX, native, 3 runs each of class_small and class_full; then loop_spread_gate
# against part 1's legacy runs (previews/issue-3/<c>-{a,b,c}) and spread.json.
# The runs are serial, so their wall times compare (walltimes.txt). fov-3d-vision is only read.
set -u
FR=$(cd "$(dirname "$0")/../../../.." && pwd)
FOV=/home/lvelho/rd/fov-3d-vision
S=$FOV/scenes
P=$FR/previews/issue-3b
D=$FR/docs/runs/issue-3/part2
mkdir -p "$P" "$D"
cd "$FR" || exit 1
T=$P/walltimes.txt

run() {   # name, then the command
  local name=$1; shift
  if [ -e "$P/$name" ]; then echo "exists: $P/$name" >&2; exit 1; fi
  local t0; t0=$(date +%s.%N)
  "$@" > "$P/$name.log" 2>&1
  local rc=$?
  echo "$name exit=$rc wall_s=$(echo "$(date +%s.%N) - $t0" | bc) start=$(date -d @${t0%.*} +%H:%M:%S)" | tee -a "$T"
}

A="--profile small --policy coverage --fixations 50 --device CPU"
run calib-cpu-legacy blender -b $S/calib_room/calib_room.blend --python-exit-code 1 -P $FOV/tools/active_loop.py -- --out $P/calib-cpu-legacy $A
run calib-cpu-exact  blender -b $S/calib_room/calib_room.blend --python-exit-code 1 -P tools/active_loop_session.py -- --fov3d $FOV --mode exact --out $P/calib-cpu-exact $A
run calib-cpu-ctrl   blender -b $S/calib_room/calib_room.blend --python-exit-code 1 -P tools/active_loop_session.py -- --fov3d $FOV --mode exact --ipd 0.06300000000000001 --out $P/calib-cpu-ctrl $A
$FOV/.venv/bin/python tools/compare_loop_runs.py $P/calib-cpu-legacy $P/calib-cpu-exact --fov3d $FOV --json $D/exact.json | tee -a "$T"
echo "compare exact exit=${PIPESTATUS[0]}" | tee -a "$T"
$FOV/.venv/bin/python tools/compare_loop_runs.py $P/calib-cpu-legacy $P/calib-cpu-ctrl --fov3d $FOV --json $D/exact-control.json | tee -a "$T"
echo "compare control exit=${PIPESTATUS[0]}" | tee -a "$T"

for r in 1 2 3; do
  run class_small-native-$r blender -b $S/classroom/classroom_eye.blend --python-exit-code 1 -P tools/active_loop_session.py -- --fov3d $FOV --mode native --out $P/class_small-native-$r --profile small --policy coverage --fixations 50
  run class_full-native-$r  blender -b $S/classroom/classroom_eye.blend --python-exit-code 1 -P tools/active_loop_session.py -- --fov3d $FOV --mode native --out $P/class_full-native-$r --profile full --kappa 3.8 --policy coverage --fixations 50
done
for c in class_small class_full; do
  $FOV/.venv/bin/python tools/loop_spread_gate.py --spread docs/runs/issue-3/spread.json --config $c \
    --runs $P/$c-native-1 $P/$c-native-2 $P/$c-native-3 --legacy previews/issue-3/$c-a previews/issue-3/$c-b previews/issue-3/$c-c \
    --fov3d $FOV --json $D/gate-$c.json | tee -a "$T"
  echo "gate $c exit=${PIPESTATUS[0]}" | tee -a "$T"
done
cp "$T" "$D/walltimes.txt"
