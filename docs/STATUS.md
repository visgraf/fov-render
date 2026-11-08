# Status

Rewritten at each milestone; earlier versions are in git.

**Now: the migration is done** (pending the merge of #3's PR). Steps 1–7 of the plan are built and
gated on the workstation:

- **Steps 1–4:** #1 and #10. The Phase 1 gates on OPTIX, plus the Classroom gates; `complete` is the
  default Object Index numbering (D13).
- **Step 5:** #2. f3d-vision's greedy explorer runs unchanged on a Session and reproduces the frozen
  Playground-2 run (`check_equivalence` 23/23).
- **Step 6:** #3. fov-3d-vision's foveated active loop runs unchanged on a Session
  (`tools/active_loop_session.py --mode exact`). Over 50 fixations on CPU it equals the legacy loop
  bit for bit in every file and every Phase C number; the IPD + 1 ULP control differs.
- **Step 7:** #4. The Barbershop passes the scene gate (`python -m fov_render.checks.scene`). The
  Pavilion's triage led to the fixes behind gates 4.7–4.11.
- **#9:** the one-off darkening of a wheel run is ignored (Luiz); the wheel is not the cause.

Gates: 32 in the Chat sandbox, plus the 4 Classroom gates and the scene gate on the workstation.

| Issue | Step | State |
|---|---|---|
| #1, #10 | Phase 1 gates on the workstation; numbering default | done |
| #2 | Step 5: Session, server, client | done |
| #4 | Step 7: the Barbershop | done |
| #9 | the wheel's darker Classroom | done (ignored) |
| #3 | Step 6: the active loop on a Session | PR for Luiz |

**Next:** Luiz merges #3. Then the clean-up discussed in Chat: tag the end of the migration, turn
the legacy comparisons into stored known answers, remove `legacy/`. Chat opens it as a Decision.

**Known limits:**
- **OptiX:** not bit-reproducible. Position and Object Index repeat exactly; Combined does not.
  CPU repeats bit for bit on the calibration room, but not on the Classroom (#9).
- **The active loop is chaotic.** A one-float32-step difference at fixation 0 (fov_render's float64
  rig against fov-3d-vision's float32 composition) gives a different 50-fixation run. The unchanged
  legacy loop does the same when its IPD moves by 0.2 µm (#3).
- **Native mode is not validated statistically** (Luiz, #3). In `tools/active_loop_session.py
  --mode native`, fov_render's own rig and records drive the loop. Its fixation-0 records agree
  with legacy's (gate 6.3), but whether its Phase C numbers are unbiased is not established: on the
  full Classroom its fine-level errors leaned ~3 % high, in too few runs to tell.
- **Object Index:** instances of one object share its number; geometry-nodes instancers are listed,
  not numbered.
- **Not built:** the `fov-render` command line of report §8.6, beyond `python -m fov_render.checks.scene`.
