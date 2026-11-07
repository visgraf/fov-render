# How we work on fov-render

This is the N-Surface method as used in f3d-vision, made lighter. One file of rules instead of
three, GitHub Issues instead of hand-carried prompts. If something here gets in the way, say so
in an issue: the method serves the work.

## Who does what

- **Luiz** decides: direction, what gets built, what is accepted. Only Luiz changes this file.
- **Chat** (Claude in claude.ai) plans, writes code that runs without a GPU, runs what its sandbox
  can, opens issues for Code and reviews what comes back. The sandbox has the `bpy` 5.2.1 wheel, so
  it can render on CPU, but it has no GPU and no Classroom assets.
- **Code** (Claude Code on the RTX 4090 workstation) runs what needs the GPU or the assets,
  measures, fixes what the issue allows, and reports.

## Issues are the channel

Every piece of work is an issue. Whoever needs something opens it, usually Chat.

1. **Open** with the *Task* template: goal, branch, what to run, what "done" means, what not to touch.
2. **Whose turn** is a label: `for-code`, `for-chat` or `for-luiz`. When you finish your part,
   comment and switch the label. One label at a time.
3. **Report** in a comment: the commit, the commands you ran, the numbers (say where each comes
   from, rule 1), what failed. Long output goes in a file in the repo; link it.
4. **Decisions** go to Luiz with the *Decision* template (or a `for-luiz` comment): the question,
   the options, a recommendation. Luiz answers in the issue; that comment is the record.
5. **Done** when Luiz closes the issue. Commits and PRs say `Refs #n`, or `Fixes #n` to close it on merge.

Luiz starts Code with "work on #n"; nobody has to copy prompts around.

## Six rules

1. **Say where a number comes from.** Measured (command, commit, machine), estimated, or proposed.
   A Chat sandbox number is labelled "Chat sandbox". GPU, timing and Classroom numbers count only
   when Code measures them on the workstation.
2. **Every piece of code ships a check that can fail**, with a control that must fail. Never weaken
   a check or move a threshold to make something pass; find out why first.
3. **If it isn't in the repo or an issue, it didn't happen.** Conversations are not memory.
   Decisions that matter end up in `docs/PLAN.md` or `docs/STATUS.md`.
4. **Fix forward.** Don't rewrite pushed history; a correction is a new commit.
5. **When unsure, ask** (`for-luiz`) instead of guessing.
6. **Behaviour before structure.** When legacy code moves here, show it behaves the same, by a
   check against the copies in `legacy/`, before improving it.

## Branches

`main` moves only when Luiz accepts. Work on `issue-<n>-<slug>`. One writer per branch: pull before
you start, push before you hand over, and put the commit in your issue comment.

## Project facts

- Metres. The head is fixed; eyes rotate about their own centres. `EYE` is a camera object: local
  −Z is the gaze, local +Y is up.
- Equirect `(u, v)` → direction: `lon = (u − 0.5)·2π`, `lat = (0.5 − v)·π`, `v = 0` at the top.
- Epipolar `(θ, φ)`: θ from +X (the baseline), `φ = atan2(d_y, −d_z)`.
- Two ways to run Blender-side code, not interchangeable: `blender -b -P script.py -- …` (the
  workstation) and `python script.py -- …` with the `bpy` wheel (sandboxes). Blender-side code
  imports only `bpy`, `mathutils`, `numpy` and the standard library.
- `blender -b -P` exits 0 even when the script failed: a tool exits nonzero itself.
- Cost: interactive under ~10 s, batch under ~5 min, overnight only when the issue says so.
- Renders and other generated output are not committed (`previews/` is ignored); checksums and
  the command that regenerates them are.

## Files

- `CLAUDE.md`: this file. `docs/STATUS.md`: where the project is, a page at most, rewritten at
  each milestone. `docs/PLAN.md`: the plan and the decisions.
- `legacy/`: byte-identical copies of the code fov-render replaces, used only by the checks.
  Never edit them.
