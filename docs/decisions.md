# Decision log

One entry per decision, kept lean for the final write-up. Eval-first: nothing
lands without a before/after number from the harness.

## Architecture & eval harness

- `src/detect`, `src/track` — concrete models behind `Protocol` interfaces
  (`Detection`/`Detector`, `Track`/`Tracker`), swappable. Baseline is YOLOv8n +
  ByteTrack (ultralytics).
- `eval/metrics.py` — predicted tracks matched to GT people by temporal IoU under
  a Hungarian (optimal one-to-one) assignment with a `min_iou=0.5` floor, so
  spurious tracks surface as count error rather than corrupting dwell. Reports
  count error (pred − gt), dwell MAE, dwell MAPE (over matched pairs only).
- `eval/label/schema.py` — `PersonInterval` vs `TrackInterval` kept distinct so a
  person/track mix-up can't slip through matching.
- Config-driven (`configs/cam1.yaml`): ROI polygon, thresholds, models. Nothing
  tuned to the dev clip — we are tested on a different video from the same camera.

## GT labelling protocol

- **Population:** only people who queue for/use the kiosk. Walk-throughs, staff,
  and non-interacting companions are excluded entirely.
- **Inclusion is behavioural, not geometric.** The kiosk sits at the shop
  entrance, so its queue lane overlaps the entrance walking path; ROI membership
  alone cannot separate customers from pass-through shoppers. Included only if the
  person stopped / queued / interacted with a screen — even if a walker's path
  clipped the ROI.
- **Dwell = queue wait + active use combined** (enter = ROI entry / join queue,
  exit = ROI exit after finishing).
- **Returning people keep the same person_id** — count is distinct *people*, not
  visits. Multiple (enter, exit) segments per id; brief step-outs (<~2s) not
  split; durations summed per id via `collapse_segments`, shared by GT and the
  pipeline so both sum the same way.
- **Result:** 18 people over ~216,300 frames (~2h @ ~30.08 fps); 8 returned. Long
  dwells (up to ~500s) manually verified genuine (slow self-service + queue wait).

## Coordinate space

ROI gating runs in native 1280×720 (not the letterboxed detector input). Anchor =
box bottom-center `((x1+x2)/2, y2)`. `roi_polygon` is authored in native 1280×720
pixels; verified a test slice returns non-empty in-zone tracks, only possible
post `scale_boxes` mapping to the native frame.

## Baseline results (full video)

YOLOv8n + ByteTrack, no ReID/gating: **354 tracks vs 18 GT, count_error +336,
2/18 matched at IoU≥0.5, dwell MAE 121.8s.** 105.5 FPS on MPS (~34 min). Peak
VRAM read 0.0 — `torch.cuda.max_memory_allocated` only reports on CUDA, so VRAM
must be measured on the T4 target.

Per-person diagnosis:
- **Fragmentation ≈ 85% of the overcount.** 285/354 tracks overlap a real GT
  person — fragments, not false tracks. ByteTrack is motion-only, so any occlusion
  longer than the track buffer (~1s) returns a person as a new id. Severity tracks
  crowd density: P1/P2/P3 (dense window ~26700–41960) shatter into 41/35/21 tracks
  each; P16–18 (quiet stretch) fragment far less. One track can be the best match
  for several simultaneous people (track 305 ↔ P7/P8/P9), which one-to-one matching
  can't resolve → crowding depresses match rate independently of fragmentation.
- **69/354 tracks have zero GT overlap**, classified: 42 walk-throughs, 11 staff,
  ~8 phantom (no person present — likely kiosk-screen signage/reflections), 5
  candidate missed customers (mostly children, pending accompanying-person policy),
  rest ambiguous.
- **Staff** wear a consistent black uniform + light hygiene headcover; appear both
  alone and while assisting at the kiosk, so timing can't proxy for them. Confirmed
  staff track ids: **680, 38, 37, 915, 719, 618, 665, 711, 828, 829.**
- **ROI gate too permissive:** a single feet-anchor lets edge-clipping walkers and
  boundary-grazing phantom boxes count as in-zone.

Deferred (out of Phase 3 scope): phantom/ghost detections; GT reconciliation of the
5 candidates; P6 segment 3 (frames 75471–75650) — a genuine detector recall miss
with no overlapping track.

## Phase 3 — improvements (measured on the dense slice)

Priority from the diagnosis: ReID (biggest lever) > zone hardening > stationarity
gate > staff filter.

**Slice + scoring.** `eval_slice: [26700, 71000]` is the dense window (P1–P9) where
fragmentation and staff are worst. `run.py --slice` runs only that range (full
video is the default). `evaluate_baseline.py --slice` restricts GT to the window: a
segment is included if it overlaps (`enter<end and exit>start`) and is **clipped**
to the bounds before dwell metrics, so boundary-crossing visits (P6 runs 338 frames
past a 71000 end) don't inflate error. Slice-run predictions need no clipping.
Window keeps exactly P1–P9 (9 people).

### Step 1 — appearance ReID re-association

- **Backbone: torchvision `mobilenet_v3_small`, head removed, ImageNet weights.**
  Chosen over OSNet because torchreid's OSNet weights download from Google Drive
  hung >2 min (blocks the local sanity check, risks the remote run); torchvision
  weights host reliably on download.pytorch.org. 0.93M params, a few MB VRAM on top
  of YOLOv8n — inside the 16 GB edge budget. Config-driven (`reid.model`) so OSNet
  can be swapped in later. Trade-off: ImageNet features are less person-specific and
  non-negative (cosine sits high), so appearance alone is not trusted.
- **Stitch (post-process, ByteTrack untouched):** union-find; merge a track that
  ends with a later one starting within `gap_frames` (90, ~3s), near where it left
  off (`max_anchor_dist` 250px), matching in appearance (`min_similarity` 0.8 cosine
  on mean embeddings). All three gates required, so two people side-by-side aren't
  merged. Canonical id = earliest track.
- **Scope: within-visit occlusion breaks only.** Multi-visit returns (minutes)
  exceed `gap_frames` and stay separate; `segment_gap_frames` still re-splits a
  merged id that has a genuine long internal gap.
- Thresholds are principled defaults, not tuned to the clip — revisited in the final
  tuning pass.
- **Result (slice, 9 GT):** pred 172→**99** (−42%), count_error +163→**+90**,
  matched 1→**2**, dwell MAE 40→**81s**. Overcount (ReID's target) cut sharply;
  match rate still crowding-limited; dwell MAE is noisy over 1–2 matched pairs and is
  the signal to watch for over-merging. Baseline-slice row is *derived* (full
  baseline tracks filtered+clipped to the window; the baseline pipeline predates
  `--slice`).

### Step 2 — zone-membership hardening

`in_zone` now requires the lower `box_depth_frac` (0.4, config) of the box's central
axis inside the ROI — both the feet point and the point 40% up — instead of the feet
point alone. Targets two confirmed zero-overlap failure modes: edge-clipping walkers
(feet inside, body out) and phantom boxes grazing the boundary. Result: pending
slice-eval.

## Tooling

`evaluate_baseline.py` scores `outputs/tracks.yaml` against `kiosk_gt.yaml`,
collapsing repeat visits with the pipeline's own `collapse_segments` and calling the
untouched eval harness; regenerates `outputs/eval_report.csv`. `--slice` adds the
windowed scoring above. Labelling/diagnostic scripts (label_gt, validate_gt,
review_frames, diagnose_baseline, classify_tracks, define_roi) are gitignored — not
part of the deliverable.
