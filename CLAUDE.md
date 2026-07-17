# CLAUDE.md — Kiosk Analytics Pipeline

## What this is
Person counting + dwell-time analytics for one fixed-camera retail kiosk.
Input: one .mp4. Outputs: (1) count of distinct people who queue for/use the kiosk,
(2) per-person dwell time, (3) an annotated overlay video. Everything is measured
against hand-labelled ground truth.

## Non-negotiable constraints
- Peak VRAM <= 16 GB (edge target; develop and report against a T4-class GPU).
- Runs on an arbitrary .mp4 via a single command.
- Report throughput (FPS) and peak VRAM for every benchmarked configuration.

## How we work here
- Evaluation comes first. Nothing lands without a before/after number from the
  eval harness. If a change can't be measured, it doesn't go in.
- Small, single-purpose commits with real messages. One change, one commit.
- Baseline first, then justified improvements.
- Config-driven, never hardcoded. ROI, thresholds, model choices live in configs/.
  We are tested on a different video from the same camera; nothing may be tuned
  to the dev clip.

## Architecture (keep this shape)
src/
  detect/   detector wrappers behind one swappable interface
  track/    tracker wrappers behind one swappable interface
  zones/    ROI definition + enter/exit event logic
  dwell/    track -> per-person dwell aggregation + fragment stitching
  overlay/  annotated video renderer
  run.py    single entrypoint: video + config -> outputs
eval/
  label/    ground-truth format + loading
  metrics.py  count accuracy, dwell MAE/MAPE, temporal matching
  report.py   comparison table + plots
configs/    per-camera yaml (roi polygon, thresholds, model)

## Style
- Python, ruff + black, type hints on public functions.
- Comments explain *why* and non-obvious decisions; never restate the code.
- Domain names: track, dwell_s, kiosk_roi, enter_frame. Not data/result/tmp.
- No emoji, no banner comments, no defensive try/except around things that
  shouldn't fail. In dev, let it crash loudly.
- No premature abstraction. One implementation is a function, not a factory.

## Don't
- Don't add unrequested features.
- Don't swallow exceptions or hardcode paths/thresholds/ROI.
- Don't write a module without a matching entry in the decision log.

## Git workflow
- Commit after each completed phase/step, not continuously mid-work.
- Write commit messages the way a person would after finishing a task:
  present tense, specific, no filler ("add ROI stationarity gate to
  filter pass-throughs" not "Update files" or "Implement changes").
- Never add a "Co-Authored-By: Claude" trailer, "Generated with Claude Code"
  line, or any AI-attribution footer to any commit message. Plain
  `git commit -m "..."` with nothing appended.
- Stage only files relevant to the phase just completed. Don't bundle
  unrelated work-in-progress into one commit.