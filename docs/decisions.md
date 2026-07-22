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

### Step 2 — bake-off results

Five configs, same slice as Phase 3 (frames 26700–71000, 9 GT people), everything held
constant except the tracker + post-hoc-stitch flag. Accuracy (count/dwell) from
trailbreak (RTX 4070 Ti Super); FPS + peak *reserved* VRAM from a Colab T4 on a 3k-frame
dense sub-slice (the worst case — most concurrent people → slowest, highest VRAM):

| config                          | pred | count_err | matched | dwell MAE | dwell MAPE | FPS (T4) | VRAM_resv |
| ------------------------------- | ---- | --------- | ------- | --------- | ---------- | -------- | --------- |
| bytetrack (ByteTrack + post-hoc)| 79   | +70       | 2/9     | 22.14s    | 12.32%     | 35.9     | 0.201 GB  |
| strongsort + OSNet              | 131  | +122      | 1/9     | 41.36s\*  | 28.01%\*   | 27.2     | 0.331 GB  |
| botsort + OSNet                 | 79   | +70       | 2/9     | 18.35s    | 10.15%     | 28.7     | 0.331 GB  |
| ocsort (motion-only control)    | 116  | +107      | 1/9     | 41.36s\*  | 28.01%\*   | 57.5     | 0.197 GB  |
| deepocsort + OSNet              | 138  | +129      | 2/9     | 24.55s    | 16.69%     | 21.5     | 0.331 GB  |

\* single matched pair — dwell is noise at matched=1; only the matched=2 rows compare.

Read:
- **VRAM is a non-issue.** Peak reserved tops out at **0.331 GB (~2% of the 16 GB
  budget)** and is flat across trackers. Stability check: StrongSORT reserved VRAM is
  0.331 GB at both 3k and 15k frames — identical, so the per-track feature bank
  plateaus, no leak.
- **Built-in ReID did not beat the bolted-on stitch.** Three of four boxmot trackers
  are *worse* on count than ByteTrack + post-hoc (+122 / +107 / +129 vs +70). Only
  **BoT-SORT ties the baseline count (+70) and edges its dwell** (MAE 18.35 vs 22.14,
  MAPE 10.15% vs 12.32%) — out of the box, no custom stitch. The differentiator is the
  base association engine, not appearance: ByteTrack-lineage (ByteTrack, BoT-SORT) both
  land at 79; observation-centric SORT (OC-SORT/DeepOCSORT) and classic StrongSORT
  fragment more here. The control confirms it — bolting OSNet onto OC-SORT (→ DeepOCSORT)
  makes *count* worse (+107 → +129), not better.
- **The core problem is unmoved.** Match rate stays at **2/9 at best across every
  tracker**, baseline included — on this dense slice the crowding-induced one-to-one
  matching ceiling caps identity recovery, exactly as Phase 3 predicted, and neither
  post-hoc nor built-in appearance breaks it.
- **Throughput (worst-case window):** OC-SORT (57.5) and ByteTrack (35.9) clear the ~30
  FPS source rate; BoT-SORT (28.7) sits just under (full-video average would be higher);
  StrongSORT (27.2) and DeepOCSORT (21.5) are below. The cost is boxmot's ECC
  camera-motion compensation — wasted work on a fixed camera.

Bottom line: not a decisive win for built-in over bolted-on — roughly a tie, with
BoT-SORT trading ~20% throughput for a modest dwell gain and a simpler pipeline (no
stitch module). Tracker selection (Step 3) deferred pending the overlay + diagnostic
review below.

### Step A — overlay video renderer

`src/overlay/renderer.py` draws the full-pipeline result on the source video: per-track
boxes coloured consistently by id (stable golden-angle hue per id), the id + running
dwell-so-far, a live "Current count: N" of in-zone non-staff people, and staff tracks in
red with a STAFF tag (rendered, not hidden, so where the staff filter fires is visible
for review). The ROI polygon is outlined for context. Output path, codec, and the
rendered span (inherited from the artifact's frame range, i.e. run.py `--slice`) are
config-driven (`overlay:` block).

`tracks.yaml` is per-visit intervals with no per-frame boxes, so run.py gained an opt-in
per-frame render artifact (`render_frames.yaml`, gated by `overlay.emit_render_frames`,
written into `--out-dir`): each surviving in-zone box per frame resolved to its canonical
id + kind (customer / staff), boxes the stationarity gate dropped excluded. The renderer
consumes that + the video — same full-pipeline provenance as tracks.yaml — so one slice
pass feeds both the overlay and the diagnostic. Off by default so the full-video
analytics run isn't burdened with a 216k-frame artifact. Rendered once on the dense slice
→ `outputs/overlay_slice.mp4`.

**Robustness fix uncovered while rendering:** a sub-pixel-wide in-zone detection clamps to
a zero-area crop that `cv2.resize` rejects, aborting the run. It surfaced re-running the
slice locally (MPS); trailbreak's CUDA detector floats never produced the degenerate box,
so the committed numbers are unaffected. `src/reid/embed.py` now clamps every crop to a
>=1px extent within frame bounds — same class of fix as the boxmot wrapper's degenerate-box
drop. The staff classifier already guarded empty crops.

### Step B — diagnostic against the current best pipeline (dense slice)

Re-ran `diagnose_baseline.py` against the current best pipeline's slice `tracks.yaml`
(ByteTrack + post-hoc stitch + zone hardening + stationarity + staff; local MPS run,
86 predicted tracks vs trailbreak's 79 — a detection-float difference that doesn't change
the structure), not the Phase 2 baseline. The question: are the ~77 excess predicted
tracks fixable model errors or structural?

- **Every excess track is real-person fragmentation — 0 / 86 have zero GT overlap.** The
  Phase 3 filters (zone hardening + stationarity + staff) removed 100% of walk-throughs,
  phantoms, and staff on this slice: not one surviving predicted track fails to overlap a
  real GT customer. The overcount is entirely the same person counted many times, never a
  spurious detection. (Phase 2 baseline had 69/354 zero-overlap tracks; that class is now
  empty.)
- **Fragmentation is concentrated in two structural regimes:**
  - *Long dwellers under crowd occlusion* (P1/P2/P3, frames ~26.7k–42k): shatter into
    **38 / 32 / 16** overlapping tracks. Crucially many of these tracks *overlap in time*
    (e.g. tracks 5071 / 5076 / 5081 all span ~32.2k–33.5k) — they are simultaneous, not
    sequential, so the gap-stitch cannot merge them by construction (it links a track's end
    to a later track's start). This is detector/tracker churn under sustained occlusion in a
    tight standing cluster over 300–500s dwells.
  - *Multi-minute re-entries* (P4–P9): each return gets a **different** predicted id
    (P4 → 12038, 15623, 18097 across its three visits), because the stitch is deliberately
    scoped to ~3s gaps and these gaps are minutes. Compounded by *crowding collapse*: single
    tracks are the best match for several people at once (track 15623 best-matches P4, P6 and
    P8; tracks 15756 / 16401 are shared between P7 and P9), which one-to-one matching can't
    resolve — so match rate stays at 2/9 regardless of tracker (confirmed by the Step 2
    bake-off).
- **Read: the remaining error is structural, not spurious-detection cleanup.** The filters
  are saturated (no false tracks left to remove), so further count gains must come from
  *merging* real fragments, which splits into: (a) a model-fixable slice — long-gap ReID to
  re-link multi-minute re-entries — bounded by precision risk (a looser gap/appearance gate
  starts merging distinct people, and the ImageNet/OSNet appearance signal is already the
  weak link), and (b) a largely inherent slice — simultaneous overlapping tracks in dense
  occlusion, which no sequential stitch and no built-in-ReID tracker in the bake-off
  resolved. This is the case for evaluating a segmentation/instance approach (queued
  separately) against the crowd-occlusion fragments, and for documenting the multi-minute
  re-entry collapse as a precision-bounded limitation rather than chasing it with looser
  thresholds. Frame ranges per fragment are in the diagnostic output for review against
  `overlay_slice.mp4`.

## Phase 5 (cont.) — crowding-collapse fix, coverage diagnostics, OSNet long-gap merge

### Crowding-collapse correctness fix

The overlay exposed one canonical id drawn on several people at once. Cause: the post-hoc
stitch's pairwise gap check blocked *directly* overlapping merges, but union-find is
transitive — A can link to B and to C (each disjoint from A) while B and C overlap, so a
whole queue collapses into one id. Traced id 15756: it absorbed **18 raw tracks with 27
simultaneously-active pairs**. Fix (`src/reid/stitch.py`): a hard invariant — two identities
never merge if their segments were ever simultaneously active, checked across whole merged
groups before any spatial/appearance gate. Not a threshold. Pinned by `eval/tests/test_stitch.py`.
Result on the slice: **2472 simultaneous-id collapse instances → 0**, purity 100%. Later
optimized from an O(N^3) member rescan to an incremental per-group interval list
(~O(N^2), verified identical id_maps, ~2200x faster on 355 tracks) so it scales to the full
video.

The fix *worsens* the naive metrics (count_error +77→+95, matched 1→0) because the collapse
was masking fragmentation and its long spurious tracks were accidentally satisfying the
IoU≥0.5 matcher — which motivated better diagnostics.

### Coverage/purity metrics + oracle ceiling (extends eval, strict metric kept)

`eval/metrics.py` gains `coverage_report`: per-GT-person **coverage** (fraction of real frames
with any predicted box) and **fragment count**, and per-track **purity** (largest share of a
track's frames on one GT person). Needs per-frame boxes, so run.py emits them via the render
artifact / `stitch_state.pkl` dump. Current best pipeline on the slice: mean coverage **76%**
(P4/P5 only 54–61% — a real recall gap), purity **100%** (confirms the invariant).

Oracle stitcher (diagnostic only, GT used purely as a ceiling): perfectly merging the
fragments that already exist gives **count_error +0, 9/9 matched** (any-overlap) — proving the
raw fragments are sufficient and **count is a merge-algorithm bottleneck, not a detection
one**. Argmax-assignment oracle (each fragment to one person) collapses to 4 people / 15s dwell,
exposing that temporal-only GT cannot separate co-present people — so dwell has a co-presence
ceiling stitching can't cross.

### OSNet long-gap merge (the merge improvement)

Merge-blocker analysis on the raw fragments: mobilenet-ImageNet appearance was the dominant
blocker (116/212 same-person pairs) and its same/different cosine distributions **fully
overlap** (same 10th pct 0.169 vs different 90th pct 0.824) — no threshold works, so widening
the gap alone would over-merge. Swapped the post-hoc embedder to **OSNet** (the boxmot
person-ReID backbone; `src/reid/embed.py`, config-driven — a `.pt` name selects it), which
separates far better (same 10th pct 0.492 vs different 90th pct 0.693). With OSNet, widened
`configs/cam1.yaml` reid to **gap 3000 / anchor 400 / sim 0.6** (offline sweep on the dumped
state, exact replay of the pipeline):

| config | pred | count_err | matched | purity |
| ------ | ---- | --------- | ------- | ------ |
| fixed stitch (mobilenet, gap 90) | 104 | +95 | 0/9 | 1.00 |
| OSNet gap 3000 / 400 / 0.6       | 12  | **+3** | **4/9** | 0.90 |

count_error **+95 → +3** (near the +0 oracle ceiling), match rate 0→4/9, at a modest precision
cost (purity 1.00→0.90: OSNet's 0.49–0.69 overlap band lets a few look-alike different people
merge across a gap). Dwell stays ~40–67s, the co-presence ceiling. Slice numbers are local MPS;
full-video accuracy re-runs on GPU next.

### Staff re-check under the long-gap config

The long-gap merge is a new mechanism (sequential, appearance-gated) that could dilute a staff
track into a customer differently from the simultaneous-merge case. On the slice: **0 staff
flagged, 0 false positives**, and the merged tracks' staff-frame fractions max at 0.026 (vs the
0.7 threshold) — no spurious flags, no near-flips. Caveat: the slice has no real staff (they sit
outside 26700–71000), so the dilution of a *real* staff track is only testable on the full video.

### Children — diagnosis (no filter built)

The pipeline has no child concept, like it originally had no staff concept; GT excludes
accompanying children (counted only if they independently interact). The 5 Phase-3
`missed_customer` flags were single-frame (0.0s) blips already dropped by the stationarity gate —
not missed customers. Measuring the current pipeline: short-box tracks exist and box height is
only weakly explained by distance (corr 0.16), but height **conflates children with
kiosk-occluded adults** — of the three shortest tracks, two are genuine children (14301, 12006)
and one is a real customer occluded by the counter (14635). So children do contribute to the
residual count (≈2 of the slice's +3), but a naive height/aspect filter would false-positive
occluded-adult customers. Decision: size the child impact on the full video first, and design a
confound-aware signal, before building any filter.

## Phase 5 (cont. 2) — staff-dilution bug + fix, and detector diagnostics

### Staff-dilution / phantom-sliver bug (Section-1 diagnostic) + fix

A diagnostic on the staff window (113000–119000, confirmed staff-680) found the staff filter
miscounting staff as a customer under the long-gap OSNet config — the dilution the earlier
"Staff re-check" flagged as untested, now confirmed and fixed. Trace: the detector boxes thin
vertical slivers of kiosk signage as "people" (~4px wide, w/h ~1:60); these phantom tracks pass
the zone gate, cluster at OSNet cosine ~0.95 (similar background), and merge into the real staff
track via a marginal 0.633 bridge. The staff person's own high-signal track (raw 904, 71.5%
staff frames) is pooled with 0%-staff phantom frames, dropping the merged staff fraction to
46.9% < 0.7 → staff reclassified as customer (rendered as a customer, counted wrong).

Fix (two parts, plus tests):
- **Aspect gate** (`src/zones/roi.py`, `kiosk_roi.min_box_aspect: 0.1`): `in_zone` rejects boxes
  with width/height below 0.1. Reasoned from the bimodal data — slivers ~0.02, real people
  ≥0.15, empty gap 0.08–0.15 — so it drops phantom slivers before they form tracks.
- **Dominant-segment staff verdict** (`src/staff/filter.py::is_staff_track`): a merged track is
  staff if its largest constituent segment is majority-uniform, not the pooled fraction, so a
  real staff track isn't diluted below threshold by merged non-uniform segments. Identical to the
  old rule for an unmerged track.
- Regression tests (`eval/tests/test_staff.py`): the 500fr@72% + 300fr@0% dilution flags staff;
  a 4×234 sliver is rejected by `in_zone`. `ultralytics==8.4.102` pinned (Section-6 drift).

Verified on the staff window: raw tracks 18→8 (slivers gone), staff fraction 46.9%→**72.1%**,
staff-680 now flagged STAFF (0 false positives), rendered with STAFF styling, count corrected.

### Other diagnostic findings (Sections 2–6)

- **Stitch O(N²) optimization is correct** — identical id_maps to the O(N³) version on the staff,
  crowd, and a never-tuned 140000–150000 window.
- **Generalization (untouched 103000–113000, 4 GT): no overfit** — coverage 97%, purity 0.98,
  count_error +2, better than the denser tuning slice.
- **Detector recall in Slice B is not the bottleneck** — both GT customers detected ≥0.5 in 15/16
  sampled frames; the missing boxes are coverage/proximity gaps, not detector misses.
- **Env drift:** ultralytics was unpinned (now pinned); local runs are MPS, remote CUDA, so
  detection floats differ (86 vs 79 tracks on the same slice) — local slice metrics aren't
  bit-identical to the CUDA full-video run.

### Detector merge (two adjacent people → one box) — sizing only, no fix

Frame 26800: YOLOv8n emits one box over a close couple. Pre-NMS (near-off, iou 0.99) it proposes
both people (man 0.78, woman 0.70, mutual IoU 0.175) plus a wide encompassing box; standard NMS
(0.45–0.95) always collapses to one, a *lower* threshold makes it worse, and only near-disabling
NMS recovers both — at the cost of duplicate boxes. So NMS tuning is not a clean fix. RT-DETR
(NMS-free) separates them cleanly (0.92 / 0.91) but costs ~14× latency (142 vs ~10 ms/frame) and
~11× params (33M vs 3M). Frequency on Slice B: the automatic classifier flags 36.8% of frames,
but validation shows it over-fires on isolated frames; genuine sustained merges (runs ≥15
frames) are **22% of the slice**, concentrated in a few multi-second couple-proximity windows
(peaks 27030–27360, 28020–29010), not random — one recurring close-couple situation, buffered by
the tracker's re-association. RT-DETR decision deferred; benefit is bounded and this slice is a
dense worst case.

## Phase 5 (cont. 3) — detector decision: RT-DETR-R18 replaces YOLOv8n

Time-boxed 3-step investigation into the two-people-one-box merge before the full-video run:

- **Frequency scales with crowd density**, so Slice B's ~17–22% is moderate, not a spike:
  sparse 85000–91000 (1.1 ROI people/frame) **2%** (≈ the classifier's FP floor) → Slice B
  couple **~22%** → dense 140000–150000 **26%** → crowded 103000–109000 (3.4 people/frame)
  **56%**. The merge is a general, density-driven failure that worsens as the kiosk gets busy.
- **RT-DETR-R18 vs the alternatives on the merge frame (26800), MPS proxy:** YOLOv8n merges the
  couple into one box (3M params); **RT-DETR-R18 separates them (2 boxes 0.91/0.90), 20.2M
  params, 48 ms, 121 MB** — ~3× cheaper than rtdetr-l (33M, 142 ms) and NMS-free (no threshold to
  tune). Only rtdetr-l/x ship in ultralytics; R18 came from `PekingU/rtdetr_r18vd` (transformers,
  needs a one-line `torch.compiler.is_compiling` shim on torch 2.2.2).
- **SAM-lite ruled out:** FastSAM-s (11.8M, 65 ms) is class-agnostic — on frame 26800 it returned
  162 instances and over-segments people into parts + furniture; a naive person-shape filter gave
  18 "people" for 2. A fair count needs a semantic person layer (CLIP prompting per instance),
  far beyond a single comparison, so it is not a drop-in and does not pay off at similar cost.

RT-DETR-R18 was wired in behind the Detector Protocol (`src/detect/rtdetr.py`, config-driven via
`detector.type: rtdetr`), and downstream verified: it emits the same xyxy pixel boxes, produces no
slivers (min box width 47px), and flows through zone/staff/stitch/ReID unchanged.

**Before/after (YOLOv8n vs RT-DETR-R18), local MPS, three windows — a genuine trade-off, not a
clean win:**

| window  | detector | count_err | match | coverage | FPS (MPS) |
| ------- | -------- | --------- | ----- | -------- | --------- |
| sparse  | YOLO     | +0        | 0/1   | 18.5%    | 56.8      |
| sparse  | RT-DETR  | −1        | 0/1   | 0.0%\*   | 9.2       |
| Slice B | YOLO     | +0        | 1/2   | 67.6%    | 40.4      |
| Slice B | RT-DETR  | +2        | 2/2   | 100%     | 9.5       |
| crowded | YOLO     | +2        | 2/4   | 95.1%    | 18.9      |
| crowded | RT-DETR  | +5        | 4/4   | 100%     | 7.0       |

\* a marginal P10 track was swallowed by a spurious 1-frame staff flag (dominant-segment rule can
flag a 1-frame 100%-staff track; harmless, FP=0). Purity ~1.0 and staff false-positives 0 for both.

- **Win:** RT-DETR recovers every identity on the dense/crowded cases — match 1/2→2/2 and 2/4→4/4,
  coverage 67→100% and 95→100% — doing exactly what it was chosen for (separating adjacent people).
- **Cost:** count_error regresses everywhere (+0→+2, +2→+5: better per-frame separation makes more
  tracks the stitch doesn't fully re-merge), and throughput drops ~4–6× (7–9 FPS MPS vs 18–56) —
  even a 2–3× T4 speedup is likely below the 30 FPS real-time target.

**Decision pending real T4 throughput** (MPS is only a proxy, and the call rests on it): lock
RT-DETR-R18 if it clears ~30 FPS on the T4 (the identity-recovery win justifies the count_error
cost); otherwise revert to YOLOv8n (better count_error, comfortably real-time, at the cost of
merging crowded pairs). Config currently left on RT-DETR, uncommitted, awaiting that number.
