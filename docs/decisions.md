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
(feet inside, body out) and phantom boxes grazing the boundary.
- **Result (slice, 9 GT):** pred 99→100, `num_matched` 2 and dwell MAE 81s
  byte-identical to Step 1 — a no-op on this customer-dense window (the ±1 is ReID
  re-stitch jitter from the gate change). Its target population (edge-clip walkers,
  phantoms) is sparse in the dense window and concentrated in the sparser rest of
  the video, so the payoff is expected in the full-video pass. Crucially removed no
  real customers, so `box_depth_frac` 0.4 is safe.

### Step 3 — stationarity / min-dwell gate

New `src/zones/stationarity.py`, applied to merged tracks before aggregation. A
track counts as a visit only if it lingers: in-ROI dwell ≥ `min_dwell_s` (3.0), OR a
run of ≥ `min_still_frames` (15) consecutive frames with anchor step ≤ `max_step_px`
(4.0 px/frame). Targets people crossing the ROI *interior* at walking pace — the
walk-throughs zone hardening (edge-clip only) can't catch, unavoidable given the
kiosk sits on the entrance path.
- **Result (slice, 9 GT):** pred 100→**80** (−20, −20%), count_error +91→**+71**,
  `num_matched` 2 and dwell MAE 81s unchanged. First gate to bite on the slice:
  removed 20 walk-throughs and zero real customers.

### Step 4 — staff-exclusion filter

New `src/staff/filter.py`. Staff wear a horizontal green-over-red-over-white chest
stripe on an all-black outfit + dark head covering. A frame reads as staff only if a
chest band (`stripe_band`, fractions of box height) holds **both** a saturated-green
and a saturated-red cluster **and** green sits above red (the uniform's layout); a
track is flagged only if that holds across ≥ `min_staff_frame_frac` (0.7) of its
frames. The stripe is the load-bearing signal — plain black clothing is common on
customers, so it alone can't decide.

Two calibration findings from real staff crops changed the design from the first
placeholder version:
- **Darkness is not a usable confirming signal here.** The black outfit renders as
  mid-gray on this camera (lower-torso V median 77–121), so a low-V test
  false-negatives real staff. Replaced it with the **green-above-red ordering**,
  which is lighting-invariant and more specific to the uniform.
- **Red is matched on the high hue side only (155–179).** The staff are dark-skinned
  and skin sits at hue ~0–15 — inside pure red's low side — which inflated the red
  cluster with neck/arm skin and inverted the ordering. The stripe red actually sits
  at hue ~167–175, so excluding the low side keeps skin out.

Validated: both reference crops flag as staff; all-black / bright / skin-tone crops
do not; and on the raw video the confirmed staff-680 window (frames ~115468+) is
flagged, with `run.py --staff-debug N` dumping annotated frames that show the box
on a uniformed staff member. HSV ranges stay config-driven for full-video re-tuning.

Flagged tracks are removed from `tracks.yaml` and written to `outputs/staff.yaml`;
`evaluate_baseline.py` prints a false-positive check — how many flagged tracks
temporally match a GT customer (must be 0). Confirmed staff ids (680, 38, 37, 915,
719, 618, 665, 711, 828, 829) are a floor reference only; most sit outside the
slice, so full true-positive validation is on the full-video pass. Result (slice):
0 flagged, 0 false positives — no staff in the window, customers untouched.

## Final results

Full pipeline = baseline + ReID + zone hardening + stationarity + staff filter.
Accuracy is from the full-video run (trailbreak; the pipeline is deterministic, so
accuracy is hardware-independent); FPS + peak VRAM are from a T4 slice (the 16 GB
edge target — throughput and VRAM don't depend on frame count).

Full video vs baseline (18 GT):

| config   | pred | count_err | matched | dwell MAE | dwell MAPE |
| -------- | ---- | --------- | ------- | --------- | ---------- |
| baseline | 354  | +336      | 2/18    | 121.8s    | 46.8%      |
| final    | 157  | +139      | 4/18    | 46.5s     | 13.75%     |

Every metric improved: overcount −56%, matched doubled, dwell MAE −62%, MAPE −71%.
Staff filter on the full video: **7 flagged, 0 false positives** (no GT customer
flagged) — the deferred true-positive + false-positive validation, passed.

Slice progression (frames 26700–71000, 9 GT; fast-iteration count):
baseline 172 → +ReID 99 → +zone 100 → +stationarity 80 → +staff 80. ReID is the
dominant lever (−42%); stationarity the next (−20%); zone hardening and staff are
near-no-ops on this customer-dense window (their targets — edge-clip walkers,
phantoms, staff — mostly live outside it) but act on the full video.

Reported perf (T4): final pipeline **43.4 FPS** (above the ~30 fps source rate, so
real-time capable) and **peak VRAM 0.039 GB** (torch max_memory_allocated); even
with CUDA context + reserved cache the footprint is a fraction of the 16 GB budget.
The full-video run (trailbreak RTX 4070 Ti Super, all 216k frames) independently
reported the same **0.039 GB** peak VRAM at **311 FPS** — confirming VRAM is
hardware-independent and the pipeline is real-time on both the edge target and
modern hardware. `outputs/{tracks,staff,perf}.yaml` + `eval_report.csv` are that
full-video run, committed as the reproducible result.

Remaining overcount (157 vs 18) is residual fragmentation in dense crowds, phantom
detections (deferred), and untuned thresholds; ReID/stationarity thresholds are the
levers for a further pass.

## Tooling

`evaluate_baseline.py` scores `outputs/tracks.yaml` against `kiosk_gt.yaml`,
collapsing repeat visits with the pipeline's own `collapse_segments` and calling the
untouched eval harness; regenerates `outputs/eval_report.csv`. `--slice` adds the
windowed scoring above; `--name` labels the report row (e.g. `final`) so a
non-baseline run isn't mislabelled. Labelling/diagnostic scripts (label_gt, validate_gt,
review_frames, diagnose_baseline, classify_tracks, define_roi) are gitignored — not
part of the deliverable.

## Phase 5 — architecture comparison: built-in vs post-hoc ReID

The Phase 3 diagnostic pinned the dominant remaining failure to long-term identity
loss: 7/8 multi-segment GT people get a fresh track_id on each re-entry, and crowding
collapses several simultaneous people onto one track. Our fix is a post-hoc ReID
embed + gap-stitch bolted onto motion-only ByteTrack. Phase 5 tests whether a tracker
with appearance association built *into* the data-association step does meaningfully
better, and which one — StrongSORT / BoT-SORT / OC-SORT / DeepOCSORT via boxmot.

### Step 1 — boxmot integration

- **`src/track/boxmot_tracker.py`** wraps boxmot's `create_tracker` behind the same
  `Tracker` Protocol as `ByteTrackTracker`, so run.py swaps trackers on `tracker.type`
  (strongsort / botsort / ocsort / deepocsort / bytetrack). Tracker type and ReID
  backbone are config-driven; hyperparameters are boxmot's own per-tracker defaults —
  nothing tuned to this camera, so the bake-off compares each tracker out of the box.
- **Protocol change:** `Tracker.update` now takes the frame
  (`update(detections, frame, frame_index)`). Appearance trackers crop ReID features
  from it; ByteTrack and OC-SORT are motion-only and ignore it. One unified interface,
  both paths intact.
- **Post-hoc ReID is bypassed for boxmot trackers** (`reid.post_hoc_stitch: false` in
  `configs/cam1_*.yaml`) — they associate on appearance internally, so running our
  stitch too would double-apply appearance matching. The mobilenet embedder isn't even
  constructed on that path. `configs/cam1.yaml` keeps `post_hoc_stitch: true` for the
  ByteTrack baseline. Both modules (`src/reid/embed.py`, `src/reid/stitch.py`) stay for
  that comparison baseline. Zone hardening, stationarity gate and staff filter are
  tracker-independent and run on *every* config, so the bake-off varies only the
  identity-association mechanism.
- **OC-SORT is the control:** motion-only, no ReID and no post-hoc stitch, to isolate
  how much appearance association actually buys over pure motion inside boxmot.
- **Wrapper detail:** boxmot crops every detection for ReID and its `cv2.resize` raises
  on a zero-area crop, so the wrapper drops detections that collapse to nothing once
  clamped to the frame (fully past an edge / sub-pixel slivers); all other coordinates
  pass through untouched, leaving boxmot's motion model unaffected. Not needed on the
  ByteTrack path, which only embeds in-zone (well inside the frame) boxes.

### Step 1 — OSNet weights resolution (Phase 3 Step 1 deviation closed)

Phase 3 Step 1 fell back to a torchvision ImageNet backbone because torchreid's OSNet
weights hung >2 min on a Google-Drive download. **Resolved: boxmot's OSNet downloads
and loads cleanly.** `weights/osnet_x0_25_msmt17.pt` (OSNet x0.25 / MSMT17 — smallest
OSNet, largest ReID dataset) fetched in ~3s (3.06 MB) and produced 512-d features on a
smoke test. Honest caveat: in boxmot 12.0.2 this weight is *still* hosted on Google
Drive (via `gdown`), not boxmot's GitHub releases as first assumed — but at 3 MB the
file downloads without the virus-scan confirmation dance that hangs large Drive files,
which is what actually broke Phase 3. It auto-downloads to `weights/` on first run and
is gitignored, exactly like `yolov8n.pt`.

**Dependency pin:** `boxmot==12.0.2`, the last classic-API release. It exposes
`create_tracker`/`get_tracker_config` and pins `numpy==1.26.4` (matching the rest of
the stack), so it coexists with the ByteTrack baseline; the 20/21/22 redesign forces
numpy 2.2 and ships a broken high-level API. Needs `setuptools<81` (boxmot 12.x imports
`pkg_resources`). Local sanity checks run in a gitignored `.venv-boxmot` (Python 3.12,
since boxmot 12.x's torchvision 0.17.x pin has no 3.13 wheel); the remote targets run
3.10/3.11 where a fresh `requirements.txt` install resolves the whole stack.

### Step 1 — VRAM measurement fix

`perf.yaml` logged only `torch.cuda.max_memory_allocated()`, which counts live tensor
bytes and understates real device footprint — PyTorch's caching allocator keeps freed
blocks reserved rather than returning them to the driver. Now logs **both**
`peak_vram_allocated_gb` and `peak_vram_reserved_gb`, clearly labelled. **We report
`peak_vram_reserved_gb` against the 16 GB budget** — it's the closer proxy to what
`nvidia-smi` shows live. `reset_peak_memory_stats()` runs before the loop starts and
`torch.cuda.synchronize()` runs before either stat is read. (Both read 0.0 on mps/cpu,
so the meaningful numbers still come from the T4 target.) This means the Phase 3 final
row's 0.039 GB was an *allocated* figure; the Phase 5 tables report reserved, so those
numbers aren't directly comparable to the old VRAM column — the full-pipeline winner
gets a fresh reserved measurement in Step 3.

### Step 2 — bake-off (pending remote runs)

Five configs, same slice as Phase 3 (frames 26700–71000, 9 GT people), everything held
constant except the tracker + post-hoc-stitch flag: `cam1.yaml` (ByteTrack + post-hoc
stitch, the comparison baseline), `cam1_strongsort.yaml`, `cam1_botsort.yaml`,
`cam1_ocsort.yaml`, `cam1_deepocsort.yaml`. Accuracy (count_error, matched, dwell
MAE/MAPE) from trailbreak; FPS + peak reserved VRAM from a Colab T4, plus a longer-slice
VRAM stability check. Table + read to follow; no winner picked until the numbers land.
