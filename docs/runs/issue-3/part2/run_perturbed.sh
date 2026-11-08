#!/bin/bash
# Issue #3 part 2, a diagnosis (not a gate): how far does the legacy loop itself move when fixation 0's
# pose changes by about as much as fov_render's native pose differs from it?
#
#     bash docs/runs/issue-3/part2/run_perturbed.sh       (from the fov-render checkout; batch, ~6 min)
#
# fov-3d-vision's active_loop.py, unchanged, OPTIX, with the IPD 0.2, 0.4 and 0.6 um larger than
# 0.063 m (each eye 0.1-0.3 um further out: one to three float32 steps at the head's 1.6 m), in both
# Classroom configurations; then loop_spread_gate with these runs in place of fov_render's, against
# part 1's legacy runs. If they fail like the native runs, the gate's reference cannot tell a correct
# loop from a wrong one. The runs are serial (walltimes-perturbed.txt). fov-3d-vision is only read.
set -u
FR=$(cd "$(dirname "$0")/../../../.." && pwd)
FOV=/home/lvelho/rd/fov-3d-vision
S=$FOV/scenes
P=$FR/previews/issue-3b
D=$FR/docs/runs/issue-3/part2
mkdir -p "$P" "$D"
cd "$FR" || exit 1
T=$P/walltimes-perturbed.txt
declare -A IPD=([1]=0.0630002 [2]=0.0630004 [3]=0.0630006)

run() {   # name, then the command
  local name=$1; shift
  if [ -e "$P/$name" ]; then echo "exists: $P/$name" >&2; exit 1; fi
  local t0; t0=$(date +%s.%N)
  "$@" > "$P/$name.log" 2>&1
  local rc=$?
  echo "$name exit=$rc wall_s=$(echo "$(date +%s.%N) - $t0" | bc) start=$(date -d @${t0%.*} +%H:%M:%S)" | tee -a "$T"
}

for r in 1 2 3; do
  run class_small-legacy-ipd$r blender -b $S/classroom/classroom_eye.blend --python-exit-code 1 -P $FOV/tools/active_loop.py -- --out $P/class_small-legacy-ipd$r --ipd ${IPD[$r]} --profile small --policy coverage --fixations 50
  run class_full-legacy-ipd$r  blender -b $S/classroom/classroom_eye.blend --python-exit-code 1 -P $FOV/tools/active_loop.py -- --out $P/class_full-legacy-ipd$r --ipd ${IPD[$r]} --profile full --kappa 3.8 --policy coverage --fixations 50
done
for c in class_small class_full; do
  $FOV/.venv/bin/python tools/loop_spread_gate.py --spread docs/runs/issue-3/spread.json --config $c \
    --runs $P/$c-legacy-ipd1 $P/$c-legacy-ipd2 $P/$c-legacy-ipd3 --legacy previews/issue-3/$c-a previews/issue-3/$c-b previews/issue-3/$c-c \
    --fov3d $FOV --json $D/diag-perturbed-$c.json | tee -a "$T"
  echo "perturbed gate $c exit=${PIPESTATUS[0]}" | tee -a "$T"
done
cp "$T" "$D/walltimes-perturbed.txt"
