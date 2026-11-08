#!/bin/bash
# Issue #3 part 1: the legacy spread. fov-3d-vision's active loop, unchanged, policy coverage, OPTIX,
# run twice with identical arguments in each of Phase C's three coverage configurations, then judged
# by its own active_eval.py. fov-3d-vision is only read; the runs go to fov-render's previews/.
#
#     bash docs/runs/issue-3/run_spread.sh            (from the fov-render checkout; batch, ~5 min)
#
# The arguments are those recorded in the accepted runs' loop.json (fov-3d-vision previews/loop2/
# calib_coverage, loop3/class_coverage, loop3/class_coverage_full): 50 fixations, seed 0, OPTIX,
# profile small (64 spp) or full (256 spp, kappa 3.8).
set -u
FR=$(cd "$(dirname "$0")/../../.." && pwd)
FOV=/home/lvelho/rd/fov-3d-vision
W=$FR/previews/issue-3
mkdir -p "$W"
cd "$FOV" || exit 1
declare -A SCENE=([calib_small]=scenes/calib_room/calib_room.blend [class_small]=scenes/classroom/classroom_eye.blend
                  [class_full]=scenes/classroom/classroom_eye.blend)
declare -A ARGS=([calib_small]="--profile small" [class_small]="--profile small" [class_full]="--profile full --kappa 3.8")
# Three runs per configuration (a, b, c), judged together by active_eval (<c>-compare3). On
# 2026-11-07 a and b were run first with an eval of the pair (<c>-compare), then c with the
# three-run eval; walltimes.txt has the order and start times.
for c in calib_small class_small class_full; do
  for r in a b c; do
    out=$W/$c-$r
    if [ -e "$out" ]; then echo "exists: $out" >&2; exit 1; fi
    t0=$(date +%s.%N)
    blender -b "${SCENE[$c]}" --python-exit-code 1 -P tools/active_loop.py -- --out "$out" ${ARGS[$c]} \
      --policy coverage --fixations 50 > "$W/$c-$r.log" 2>&1
    rc=$?
    echo "$c-$r loop exit=$rc wall_s=$(echo "$(date +%s.%N) - $t0" | bc) start=$(date -d @${t0%.*} +%H:%M:%S)" | tee -a "$W/walltimes.txt"
  done
  .venv/bin/python tools/active_eval.py "$W/$c-a" "$W/$c-b" "$W/$c-c" --out "$W/$c-compare3" > "$W/$c-eval3.log" 2>&1
  echo "$c eval3 exit=$?" | tee -a "$W/walltimes.txt"
done
