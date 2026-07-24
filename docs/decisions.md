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

## Phase 6 — detector+tracker v2 (branch `detector-tracker-v2`)

Goal: a properly-tuned pipeline on a stronger detector+tracker pair, judged against
main's proven baseline (YOLOv8n + ByteTrack + custom crowding-invariant stitch + OSNet
long-gap merge). Three validation windows throughout, so results stay comparable to the
Phase 5 findings and no conclusion rests on one window:
sparse 85000–91000 (1 person) · Slice B 26700–30000 (2, close pair) · crowded
103000–109000 (4+).

**Starting state note:** the RT-DETR-vs-YOLO decision at the end of Phase 5 was left
"pending real T4 throughput". That number has *not* landed in this log, so Phase 6
treats throughput as unresolved and reports MPS figures as an explicit proxy, flagging
every place the conclusion depends on it.

### Step 0 — infrastructure: detection cache (the enabler)

Detection dominates cost (RT-DETR ~8 FPS on MPS) and both the detector bake-off and the
tracker sweep re-run the same frames many times. `dump_detections.py` writes one
detector's raw output for a window to an `.npz`; `src/detect/cached.py` replays it behind
the same Detector Protocol (`detector.type: cached`). This is the Phase 5 `stitch_state.pkl`
idea moved one stage earlier — replay a fixed upstream so only the stage under test varies.

- Verified **bit-exact**: cached replay reproduced live YOLOv8n detections over 30 frames,
  161 detections, 0 mismatches.
- Makes a sweep point cost tracking only (~2.2 min on Slice B instead of ~11 min), and it
  is what makes RF-DETR testable at all — see the dependency conflict below.
- `src/detect/build.py` centralises detector construction with **deferred imports per
  branch**, because the backends' dependency pins are mutually exclusive.
- `resolve_device` moved from `run.py` to `src/device.py`: the detector-only scripts must
  not have to import the tracker stack (boxmot is absent from the RF-DETR environment).
- **FPS caveat:** a cached run's FPS measures tracking only. Every throughput number in
  the Step 5 table comes from a **live-detector** run, never a cache replay.

### Step 0 — RF-DETR dependency conflict (isolated, not worked around)

`pip install rfdetr` resolves `transformers>=5.1.0`, but the RT-DETR wrapper is pinned to
`transformers==4.45.2` (newer transformers needs torch>=2.4; boxmot pins torch 2.2.x).
Installing RF-DETR into the working venv would have silently broken the RT-DETR path —
the detector it is being compared against.

Resolved by isolation, not by loosening a pin: RF-DETR lives in its own `.venv-rfdetr`
(torch 2.13, transformers 5.14, numpy 2.5) and meets the pipeline at the detection cache.
`src/detect/rfdetr.py` implements the same Protocol. Note RF-DETR's checkpoints keep
COCO's original 91-class ids (person = 1, not 0), so the wrapper translates from the
config's YOLO-space class ids rather than making the config carry a per-detector id.

### Step 1 — detector separation on the couple-merge frame (26800)

Region under test x∈[740,915], y∈[300,615] — the close couple YOLOv8n merges. All at the
config's feed threshold (conf 0.1), MPS:

| detector | total boxes | boxes on the couple | couple confidences | params |
| -------- | ----------- | ------------------- | ------------------ | ------ |
| YOLOv8n  | 7           | **1 (merged)**      | 0.70               | 3.2M   |
| RT-DETR-R18 | 44       | **5**               | 0.91, 0.90, + 0.22/0.14/0.13 | 20.2M |
| RF-DETR-nano | 23      | **2 (clean)**       | 0.91, 0.89         | — |

- Both NMS-free detectors separate the couple; YOLOv8n does not, reconfirming Phase 5.
- **RF-DETR's output is markedly cleaner**: exactly the two real people in the region,
  and 23 boxes on the frame against RT-DETR's 44. RT-DETR adds three sub-0.25 boxes on
  the couple (partial-body/duplicate queries). Spurious low-confidence boxes are not free
  — they are what the tracker spawns junk tracks from, and they are the mechanism behind
  the count_error regression RT-DETR showed in Phase 5.
- RF-DETR logs that it is *not* inference-optimized and offers `optimize_for_inference(fp16)`
  for "~8x on T4 via FP16 Tensor Cores" — directly relevant to the throughput question
  that blocked RT-DETR. Untested here; flagged, not claimed.

Single-frame evidence only decides separation, not counting. Window-level results follow.

### Step 1 — ROI bottom-edge bug (found via RF-DETR, affects the baseline too)

Scoring RF-DETR on the sparse window returned **count_error −1 with 0% coverage**: not one
customer track, though GT P10 spans the whole window. Not a detector recall failure —
RF-DETR produced 400 in-zone detections there, but **zero above conf 0.6**, so BoT-SORT
(whose `track_high_thresh` defaults to 0.6) could never spawn a track: only high-confidence
detections create tracks, the low band can merely extend existing ones.

Tracing why those detections were both rare and low-confidence exposed the real cause, a
**box-geometry/ROI interaction, not a confidence one**:

| frame | YOLOv8n box bottom | RF-DETR box bottom | in-zone? |
| ----- | ------------------ | ------------------ | -------- |
| 85186 | y2 = 715           | y2 = 720           | YOLO yes / RF-DETR no |
| 88273 | y2 = 715           | y2 = 720           | YOLO yes / RF-DETR no |

`roi_polygon`'s bottom edge was authored at **y = 716** on a **720**-row frame. A person at
the near edge is cut off by the frame, so their box bottom clamps to the image boundary —
below the polygon — and the anchor test rejects them. It only ever worked because YOLOv8n
under-extends edge-cut boxes by a few pixels. That is a detector quirk standing in for scene
geometry, exactly the kind of hidden coupling this project's config-driven rule exists to avoid.

**This is a latent bug on main, not something RF-DETR introduced.** In-zone detections at
conf ≥ 0.6, polygon bottom 716 → 721:

| detector | window  | bottom 716 | bottom 721 |
| -------- | ------- | ---------- | ---------- |
| YOLOv8n  | sparse  | 1248       | **6000**   |
| RF-DETR  | sparse  | 0          | **6002**   |
| YOLOv8n  | crowded | 6924       | **10903+** |
| RF-DETR  | crowded | 14292      | **19653+** |

It plausibly explains the Phase 5 observation that P4/P5 sat at 54–61% coverage and the
sparse window at 18.5% — a "recall gap" that is substantially a zone-gate artifact.

**Fix: bottom edge → y = 721**, chosen on evidence, not taste. At 720 the edge is exactly
coincident with the clamped box bottom and ray-casting is boundary-ambiguous (RF-DETR sparse
recovers only 3372 of 6002); at 721 and 725 the count is identical and stable. The floor
genuinely continues past the visible frame, so the ROI should too.

Applied to `configs/cam1_v2.yaml`. `configs/cam1.yaml` is left **byte-identical to main**, and
`configs/cam1_roifix.yaml` (baseline + this fix only) is added so the Step 5 table can separate
the ROI gain from the detector/tracker gain instead of confounding them.

### Step 1 — detector decision: RF-DETR-nano over RT-DETR-R18

Window-level comparison with the tracker held constant at boxmot's BoT-SORT defaults and
the corrected ROI, so only the detector varies. Local MPS; detections replayed from cache,
so all three saw byte-identical frames.

| window  | detector     | count_err | matched | coverage | purity | staff FP |
| ------- | ------------ | --------- | ------- | -------- | ------ | -------- |
| sparse  | YOLOv8n      | +0        | 1/1     | 100%     | 1.000  | 0        |
| sparse  | **RF-DETR**  | **+0**    | **1/1** | **100%** | 1.000  | 0        |
| sparse  | RT-DETR-R18  | +1        | 1/1     | 100%     | 1.000  | 0        |
| Slice B | YOLOv8n      | +5        | 0/2     | 42.7%    | 1.000  | 0        |
| Slice B | **RF-DETR**  | **+2**    | **2/2** | **100%** | 1.000  | 0        |
| Slice B | RT-DETR-R18  | +4        | 2/2     | 100%     | 1.000  | 0        |
| crowded | YOLOv8n      | +27       | 1/4     | 96.8%    | 1.000  | 0        |
| crowded | **RF-DETR**  | **+18**   | **4/4** | **100%** | 0.998  | 0        |
| crowded | RT-DETR-R18  | +25       | 3/4     | 100%     | 0.998  | 0        |

**RF-DETR-nano wins on every window** — never worse on count_error, and strictly better on
identity recovery in the crowd (4/4 vs RT-DETR's 3/4 and YOLO's 1/4). Supporting evidence:

- **Output cleanliness, the likely mechanism.** Detections per frame at the conf-0.1 feed
  threshold: YOLOv8n 8.4, RF-DETR 17.9, **RT-DETR 37.4** (46/frame on sparse). RT-DETR's
  extra boxes are sub-0.25 partial-body/duplicate queries — the same clutter visible on the
  merge frame (5 boxes on the couple vs RF-DETR's 2). Spurious low-confidence boxes are what
  a tracker spawns junk tracks from, which is a concrete mechanism for the count_error
  regression Phase 5 saw with RT-DETR and could not explain.
- **Throughput.** Detector-only MPS: RF-DETR **14.0–14.9 FPS** vs RT-DETR **5.4–10.8**, i.e.
  ~2× faster before any optimization, and RF-DETR additionally offers an untested
  `optimize_for_inference(fp16)` path. This does not settle the T4 question, but RF-DETR is
  the cheaper of the two NMS-free options on every measurement taken here.
- Both NMS-free detectors fix the couple merge; YOLOv8n does not. Both reach 100% coverage
  on all three windows once the ROI is corrected.

Adopted: **RF-DETR-nano**. RT-DETR-R18's Phase 5 "identity recovery at the cost of
count_error" trade-off is not intrinsic to going NMS-free — RF-DETR gets the identity win
*and* a lower count_error, at half the boxes and twice the speed.

### Step 2 — tracker: BoT-SORT, custom stitch removed

BoT-SORT (boxmot) per the Phase 5 bake-off, with `reid.post_hoc_stitch: false`. The custom
gap/anchor/similarity stitch is **not** carried over: its values were fit to ByteTrack's
fragmentation pattern, and BoT-SORT already associates on OSNet appearance inside the
data-association step, so running both would double-apply appearance matching.

Confirmed it is not needed: on all three windows the purity floor is 0.998+ and the Phase 5
crowding-collapse failure (one canonical id drawn on several people) does not recur — that
bug was a property of the union-find post-hoc merge, which is now gone entirely rather than
guarded by an invariant.

### Step 3 — what the oracle says the tuning target is

Before sweeping, the Phase 5 oracle/coverage diagnostic on RF-DETR + default BoT-SORT:

| window  | real | count_err | matched | coverage | purity | fragments | **oracle any-overlap** |
| ------- | ---- | --------- | ------- | -------- | ------ | --------- | ---------------------- |
| crowded | 22 tracks | +18  | 4/4     | 100%     | 0.998  | 62 over 4 people | **+0**, 4 tracks |
| Slice B | 4 tracks  | +2   | 2/2     | 100%     | 1.000  | 8 over 2 people  | **+0**, 2 tracks |

Coverage is already 100% and purity ~1.0, and a perfect merge of the fragments that *already
exist* reaches count_error +0. So the entire residual error is **fragmentation — identities
dropped and re-spawned — not detection recall and not false tracks.** That makes BoT-SORT's
own association parameters (how long a lost track survives, and what can re-claim it) the
correct and sufficient lever, which is exactly what Step 3 tunes.

### Step 3 — tuning BoT-SORT for RF-DETR's output (5 staged sweeps)

Methodology as in Phase 5: detections replayed from cache so only association varies, every
point scored by the harness, and **every candidate checked on more than one window** — single-
window tuning already misled this project once. 47 sweep points via `sweep_botsort.py`.
Slice B is nearly insensitive (+1..+2 throughout); crowded is the discriminating window.

**Stage 1 (one axis at a time, from boxmot defaults; crowded count_err):**

| axis | result |
| ---- | ------ |
| default | +18, 4/4 |
| `track_buffer` 300 / 1500 / 3000 | +15 but **3/4** — and identical at all three, i.e. saturated |
| `proximity_thresh` 0.7 / 0.9 / 0.99 | +17 / +16 / **+15, still 4/4** |
| `appearance_thresh` 0.15 / 0.35 | **+18 — completely inert** |
| `cmc_method: sof` | +18 (no accuracy change; ECC is wasted work on a fixed camera) |
| `track_high_thresh` 0.4 | +15, 4/4 |
| `new_track_thresh` 0.5 / 0.8 | +19 / **+14** |

The inert `appearance_thresh` is the key mechanical finding, and it was predicted from the
source before running: in `BotSort._first_association`, `emb_dists[ious_dists_mask] = 1.0` —
**the appearance distance is masked wherever IoU distance exceeds `proximity_thresh`**. At the
0.5 default, a track that has drifted can never be re-claimed by appearance no matter what the
appearance threshold is, and lost tracks expire with `track_buffer` regardless. That is also
why `track_buffer` saturates. So the built-in ReID only becomes load-bearing once
`proximity_thresh` is opened, and only then does `appearance_thresh` do anything.

**Stages 2–3 (combinations).** `new_track_thresh` — the bar to *spawn* an identity — is the
dominant lever, since RF-DETR's low-confidence queries are what junk tracks are born from.
Best: `new 0.9 + prox 0.99 + buffer 300 + appearance 0.15` → crowded **+7, 4/4, purity 1.000**;
Slice B +1, 2/2. With proximity open, tightening appearance to 0.15 now *raises* purity to
1.000 while holding the count — exactly the predicted interaction.

**Stage 4 (robustness, all three windows).** `new_track_thresh` 0.95 produces **zero tracks**:
RF-DETR's confidence ceiling is **0.934–0.943**, so the threshold sits above anything the model
can emit. Measured share of in-zone detections above each value (crowded): 0.85 → 26%,
0.90 → 7.3%, 0.92 → 1.1%, 0.95 → 0%. The count curve is flat and the cliff is sharp:

| `new_track_thresh` | 0.70 | 0.75 | 0.80 | 0.85 | 0.88 | 0.90 | 0.95 |
| ------------------ | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| crowded count_err  | +11  | +10  | +10  | **+8** | +8 | +9 | **−4 (no tracks)** |
| sparse / Slice B   | +0/+2 | +0/+2 | +0/+2 | +0/+2 | +0/+2 | +0/+1 | none |

**Chose 0.85, not the marginally better 0.90.** 0.90 leaves ~0.04 of headroom below the model's
ceiling; 0.85 leaves ~0.09 and sits in a flat region. We are tested on a different video, so
paying ~1 count for that margin is the right trade — a threshold whose neighbour produces
*nothing* is not one to sit next to.

**Stage 5 — is the custom stitch still needed? Measured, not assumed.** Yes, and for a
structural reason rather than a threshold one: BoT-SORT can only re-claim a lost track through
its first association, which is proximity-masked and expires with `track_buffer`, so it handles
occlusion but cannot re-identify someone returning minutes later.

| config (crowded) | count_err | matched | purity |
| ---------------- | --------- | ------- | ------ |
| tuned BoT-SORT, no stitch | +9 | 4/4 | 0.993 |
| + stitch gap 900, sim 0.6 | +5 | 4/4 | 0.990 |
| + stitch gap 3000, sim 0.6 | **+3** | **4/4** | 0.975 |
| + stitch gap 9000, sim 0.6 | +3 | 4/4 | 0.978 |
| + stitch gap 3000, sim 0.5 | +3 | 3/4 (over-merges) | 0.975 |
| + stitch gap 3000, sim 0.7 | +4 | 3/4 | 0.984 |

Values were **re-derived for this combination**, not inherited. They land near the ByteTrack-era
gap 3000 / anchor 400 / sim 0.6, which is a real result — the stitch's good operating point is
similar across both trackers — but it is now supported by a sweep for *this* pipeline, and 9000
was rejected as buying nothing for a wider merge risk.

**Final v2:** RF-DETR-nano + BoT-SORT (`new_track_thresh` 0.85, `proximity_thresh` 0.99,
`track_buffer` 300, `appearance_thresh` 0.15) + post-hoc stitch (gap 3000 / anchor 400 / sim 0.6).

### Step 4 — staff filter: a real regression, and it is NOT v2's

The confirmed-staff-680 window (113000–119000) under v2: the staff member **is tracked**
(track 6, frames 115460–118999) but scores a staff-frame fraction of **0.581 < 0.7**, so it is
classified as a customer — **0 staff flagged**, a false negative costing +1 on the count there.

Attribution, isolated by re-running each change separately:

- **Not the detector's box geometry.** Matching the staff person's box per frame between YOLOv8n
  and RF-DETR: height ratio **1.012**, y1 delta −2.0 px, y2 delta +1.1 px, chest band within
  1.5 px. The two are effectively the same box. (An earlier averaged comparison suggested a 53 px
  offset; that was confounded by the detectors having different in-zone box counts.)
- **It is the ROI correction**, and it hits main's pipeline identically: YOLOv8n + ByteTrack +
  stitch flags staff-680 with the original ROI (raw track 904, fraction **0.715** — reproducing
  the 72.1% recorded in Phase 5) and flags **0 staff** with the corrected ROI. Better coverage
  means longer, more complete staff tracks that include frames where the chest stripe is not
  visible, diluting a fraction whose 0.7 threshold was calibrated on shorter, higher-quality tracks.
- **The heuristic's discrimination is weak independently of any of this.** On main's baseline with
  the original ROI, GT *customer* **P11 scores 0.924** on the staff test (100% GT overlap, crowded
  window) — above the threshold. Staff-680 scores 0.715. **The classes are inverted.** The ROI fix
  actually lowers P11 to 0.401.

So lowering `min_staff_frame_frac` to catch staff-680 at 0.581 would flag P11 and create a real
false positive on a customer. **Left at 0.7 deliberately**: v2 reports **0 staff false positives on
every window**, accepting the staff false negative. Recalibrating the staff filter for the
corrected ROI (and fixing the P11 confusion, which exists on main today) is separable work and is
logged as a known limitation rather than papered over with a threshold that trades a false
negative for a false positive.

### Step 5 — full comparison (three windows, local MPS, cached detections for accuracy)

| config | window | count_err | matched | coverage | purity | staffFP | dwell MAE |
| ------ | ------ | --------- | ------- | -------- | ------ | ------- | --------- |
| baseline (main, as-is) | sparse  | **+0** | 0/1 | 18.5% | 1.000 | 0 | 0.00s |
| baseline (main, as-is) | Slice B | **+0** | 1/2 | 67.6% | 1.000 | 0 | 35.67s |
| baseline (main, as-is) | crowded | **+2** | 2/4 | 95.1% | 0.979 | 0 | 60.39s |
| baseline + ROI fix | sparse  | +0 | 1/1 | 100% | 1.000 | 0 | 0.03s |
| baseline + ROI fix | Slice B | +0 | 1/2 | 67.9% | 1.000 | 0 | 34.54s |
| baseline + ROI fix | crowded | +3 | 3/4 | 98.4% | 0.991 | 0 | 12.01s |
| **v2 (RF-DETR + BoT-SORT)** | sparse  | +0 | **1/1** | **100%** | 1.000 | 0 | 0.03s |
| **v2 (RF-DETR + BoT-SORT)** | Slice B | +1 | **2/2** | **99.8%** | 1.000 | 0 | **0.28s** |
| **v2 (RF-DETR + BoT-SORT)** | crowded | +3 | **4/4** | **100%** | 0.975 | 0 | **10.01s** |

Throughput and VRAM, measured on **live-detector** runs (never a cache replay), sequentially so
they do not contend, on the crowded 3k sub-window (worst case, matching the Phase 5 method):

| config | FPS (MPS) | peak VRAM reserved |
| ------ | --------- | ------------------ |
| baseline (main) | **31.1** | not measurable on MPS |
| v2 | **12.1** | not measurable on MPS |

**Honest read — v2 does not win outright on count, and wins decisively on everything about identity:**

- **Count_error: v2 ties or slightly trails.** +0/+1/+3 against main's +0/+0/+2. On crowded v2
  matches baseline+ROI-fix (+3); the target of *beating* main's count was not met.
- **Identity recovery: v2 wins decisively** — matched 1/1, 2/2, 4/4 vs main's 0/1, 1/2, 2/4, and
  coverage 100 / 99.8 / 100% vs 18.5 / 67.6 / 95.1%.
- **Dwell, a primary deliverable, improves by orders of magnitude**: Slice B **35.67s → 0.28s**,
  crowded **60.39s → 10.01s**. Main's low count_error is partly bookkeeping — it reaches roughly
  the right *number* of tracks while covering far less of each person, so its per-person dwell is
  badly wrong. v2 counts about as well and actually measures the right people.
- **Cost: throughput.** 12.1 vs 31.1 FPS on MPS, i.e. below the ~30 FPS source rate locally.
- **VRAM is unresolved**: `torch.cuda` counters read 0.0 on MPS, so peak reserved VRAM **cannot be
  measured on this machine** and is not reported for either config. Phase 5's T4 figures
  (0.201 GB ByteTrack, 0.331 GB BoT-SORT+OSNet, both ~2% of the 16 GB budget) suggest headroom,
  but RF-DETR is a larger detector and needs its own T4 measurement. This is a genuine gap against
  the "report peak VRAM for every benchmarked configuration" constraint, not an oversight.

## Phase 6 (cont.) — final round before locking

### ROI zone-depth sweep — keep `box_depth_frac: 0.4`

Swept 0.4 / 0.5 / 0.6 / 0.7 on all three windows for both candidate pipelines, on cached
detections so only the gate varies. **Sparse and Slice B are completely insensitive** to depth
in both pipelines (identical count_err, matched, coverage, purity at all four values), so only
crowded discriminates:

| depth | v2 crowded | baseline+ROI-fix crowded |
| ----- | ---------- | ------------------------ |
| **0.4** | **+3, 4/4, cov 100%, pur 0.975** | **+3, 3/4, cov 98.4%, pur 0.991** |
| 0.5 | +3, **3/4**, cov 100%, pur 0.989 | +2, **2/4**, cov **93.0%**, pur 0.997 |
| 0.6 | +2, 3/4, cov 100%, pur 0.987 | +3, 2/4, cov 93.0%, pur 0.997 |
| 0.7 | +2, 3/4, cov 100%, pur 0.986 | +2, 2/4, cov 93.0%, pur 0.997 |

Going above 0.4 **costs a recovered identity in both pipelines** (v2 4/4 → 3/4, baseline
3/4 → 2/4) and costs the baseline 5.4 points of coverage, while count_error does not improve
monotonically (v2 +3/+3/+2/+2; baseline +3/+2/+3/+2 — inside the noise). That is the opposite
of the requested criterion, so **0.4 stays**; the strictest option is not chosen by default,
and here it is actively worse.

Why the gate has so little room: the in-zone depth distribution is **strongly bimodal**. Of
detections whose feet fall inside the ROI, the share still inside at increasing depth is
(RF-DETR) sparse 0.525 → 0.525, Slice B 0.726 → 0.722, crowded 0.971 → 0.883 for 0.4 → 0.7.
Boxes are almost always either well inside the zone or clipping it at the very edge, and the
**edge-clippers are already removed at 0.4** (they are the 47% of feet-in detections on sparse
that fail even the 0.4 test). Only ~9% of crowded detections live in the 0.4–0.7 band.

So the residual leg-clipping visible in the overlay is **not addressable by `box_depth_frac`** —
the remaining marginal boxes are people genuinely standing at the zone boundary, and excluding
them costs a real customer's identity. If it needs fixing, the lever is the polygon's shape at
the kiosk-side edge, not the depth fraction. Logged, not changed.

### Staff filter — separability measured; the earlier "customer P11 scores 0.924" was wrong

**Correction to the previous report.** The claim that GT customer P11 scored 0.924 on the staff
heuristic (and therefore that the classes were inverted) was an **attribution error, not a
classifier failure**. Tracks were attributed to GT people by *temporal* overlap, which cannot
distinguish "is P11" from "is standing next to P11". Every high-scoring track in the crowded
window coincides with a known staff sighting from `outputs/staff.yaml`:

| raw track | staff_frac | span | overlapping staff sighting |
| --------- | ---------- | ---- | -------------------------- |
| 25 | 0.793 | 108603–108999 | 510 (108490–108655), 512 (108548–108742) |
| 23 | 0.390 | 108503–108999 | 510, 512 |
| 4  | 0.339 | 103000–105507 | 471 (103808–103842) |
| 19 | 0.226 | 106335–108821 | 510, 512 |

Those were real staff, correctly scored.

**Separability, measured properly** (`diagnose_staff_separability.py`): the heuristic is scored
only on **solo frames** — frames where exactly one subject is present *and* the detector returns
exactly one in-zone box, so the crop is unambiguously that subject. Subjects without enough solo
frames are reported as such rather than guessed at.

| population | subjects with ≥10 scored frames | staff_frac |
| ---------- | ------------------------------- | ---------- |
| GT customers | P1, P6, P10, P12, P15 | **0.000, 0.000, 0.000, 0.000, 0.000** |
| staff sightings | staff-680, staff-37, staff-754, staff-706 | **0.750, 0.970, 0.980, 1.000** |

**Perfectly separable — any threshold in (0.000, 0.750] splits them cleanly.** The heuristic is
not the problem.

Scope caveat, stated exactly: the 10 confirmed staff ids from Phase 2 (680, 38, 37, 915, 719,
618, 665, 711, 828, 829) are *baseline track ids* whose frame ranges were never recorded, and the
Phase-2 run's `tracks.yaml` has since been overwritten — so only ids with recoverable spans could
be scored: the 7 sightings in the committed `staff.yaml` plus staff-680's window. Of those, 4 had
≥10 solo frames (the rest are short sightings that are never solo). Likewise 5 of 18 GT customers
have enough solo frames; the other 13 are never alone in the ROI, which is the same co-presence
limit already documented for dwell.

**The real failure was dilution, and one bounded fix resolves it.** The ROI correction makes staff
tracks longer and more complete, so they now include frames where the chest stripe is not visible;
staff-680 scores 0.750 on solo frames but **0.581 pooled over its full 3536-frame track**, under
the 0.7 threshold calibrated on shorter Phase-5 tracks. Sweeping the threshold:

| `min_staff_frame_frac` | staff window | crowded |
| ---------------------- | ------------ | ------- |
| 0.7 (previous) | count_err **+2**, staff missed, FP 0 | +3, 4/4, FP 0 |
| 0.6 | +2, staff missed, FP 0 | +3, 4/4, FP 0 |
| **0.5 (adopted)** | **+1, staff-680 correctly flagged**, FP 0 | +3, 4/4, FP 0 |
| 0.4 | +1, FP 0 | +3, 4/4, FP 0 |

**Adopted 0.5**: it sits far above every measured customer (0.000) and below every measured staff
sighting (0.750+), corrects the staff-window count, and changes nothing on sparse (+0, 1/1),
Slice B (+1, 2/2) or crowded (+3, 4/4) — staff false positives remain **0 on every window**.

Remaining quantified limitation, not claimed as solved: **staff false negatives = 1 of 1 testable
sighting before this change, 0 after**; staff false positives = 0 throughout. The separability
evidence rests on 4 staff sightings and 5 customers with clean solo frames — enough to show the
populations do not overlap, not enough to certify the threshold against all 18 customers, 13 of
which are never solo.

### CUDA benchmark procedure (not runnable here)

`docs/colab_perf.md` holds the copy-paste procedure for the real T4 numbers. Peak VRAM is
unmeasurable on this dev machine (`torch.cuda.max_memory_reserved()` reads 0.0 on MPS), so FPS
and VRAM must come from CUDA hardware. The doc pins the install order that keeps Colab's CUDA
torch intact — `boxmot` goes in with `--no-deps` because its `torchvision<0.18` / `numpy==1.26.4`
pins would otherwise downgrade the GPU stack out from under `rfdetr`.

## Phase 6 (cont. 2) — real T4 numbers, and the visual review

### T4 measurements (Colab, live detector, 3 windows, both pipelines)

The first real CUDA numbers in this project for v2. Throughput and VRAM measured on device;
`peak_vram_reserved_gb` is run.py's own counter, `device_peak` is sampled from `nvidia-smi`
throughout the run so it includes the CUDA context the allocator counter cannot see.

| config | window | FPS (T4) | torch reserved | device peak | context gap |
| ------ | ------ | -------- | -------------- | ----------- | ----------- |
| v2 | sparse  | **8.71** | 0.250 GB | 0.364 GB | 0.114 GB |
| v2 | Slice B | **8.04** | 0.250 GB | 0.364 GB | 0.114 GB |
| v2 | crowded | **8.29** | 0.250 GB | 0.364 GB | 0.114 GB |
| baseline+ROI-fix | sparse  | **31.79** | 0.099 GB | 0.245 GB | 0.146 GB |
| baseline+ROI-fix | Slice B | **31.57** | 0.099 GB | 0.245 GB | 0.146 GB |
| baseline+ROI-fix | crowded | **28.58** | 0.101 GB | 0.249 GB | 0.148 GB |

**Accuracy is identical on CUDA and MPS** — sparse +0 (1/1), Slice B +1 (2/2), crowded +3
(4/4), staff FP 0 — confirming the determinism claim the comparison table rests on. Only dwell
MAE moved marginally (crowded 10.01s MPS → 9.6s CUDA), the documented detector-float difference.

**Is the VRAM measurement sound? Yes — it understates by the CUDA context and nothing else.**
The gap between the allocator counter and true device usage is a consistent **0.114–0.148 GB**
across every run and both pipelines, exactly the signature of a fixed per-process CUDA context.
`reset_peak_memory_stats()` runs after model construction, which is correct because it resets
peak *to current*, so weights stay counted. So the historical "under 1 GB" figures were right,
just low by ~0.12 GB. Both pipelines are trivially inside the 16 GB budget: **v2 0.364 GB
(2.3%)**, **baseline 0.249 GB (1.6%)**. VRAM is a non-issue and was never the binding constraint.

**Two of my predictions were wrong, recorded so the reasoning isn't reused:**
- I extrapolated v2 at ~17 FPS on T4 from a 1.4x MPS→T4 ratio observed for the YOLOv8n pipeline.
  **Actual is 8.3 FPS — T4 is *slower* than local MPS (12.1) for RF-DETR, the opposite direction.**
  The ratio measured on a small CNN does not transfer to a DETR-family transformer; T4 has no
  fp16 tensor-core path engaged here and its fp32 throughput is modest.
- I estimated the CUDA context at 0.3–0.8 GB. **Actual is 0.114–0.148 GB**, so the understatement
  in the old numbers was much smaller than I implied.

### Throughput is the deciding constraint

| | v2 | baseline+ROI-fix |
| --- | --- | --- |
| worst-window FPS (T4) | 8.29 | 28.58 |
| vs ~30.08 fps source rate | **0.28x** | 0.95x |
| full video (216,306 frames) | **~7.25 h** | **~2.10 h** |

Neither clears real time on a T4, but v2 misses by **3.6x** while the baseline is essentially at
it. RF-DETR ships an untested `optimize_for_inference(dtype=torch.float16)` path (Roboflow claim:
~8x on T4 tensor cores) which would plausibly close this — **not enabled or tested here**, and
noted as the first lever if v2 is revisited, not as a claimed result.

### Visual review of the crowded overlay — two confirmed, quantified limitations

Reviewed the rendered crowded window. Both known issues are visible and neither is fixed; they
are recorded as measured limitations rather than described as solved.

**1. Staff still counted as customers (false negatives).** A staff member in full uniform —
green-over-red chest stripe clearly visible — renders as an orange customer box. Verified in the
data: in the crowded window the merged tracks overlapping known staff sightings score

| raw track | staff_frac | dwell | overlaps staff sighting |
| --------- | ---------- | ----- | ----------------------- |
| 25 | 0.793 | 13.2s | 510, 512 — **flagged correctly** |
| 23 | 0.390 | 16.5s | 510, 512 — missed |
| 4  | 0.339 | 83.2s | 471 — missed |
| 19 | 0.226 | 68.0s | 510, 512 — missed |

So at `min_staff_frame_frac: 0.5`, **1 of 4 staff-overlapping tracks is caught**. This is the same
dilution mechanism already documented (staff-680: 0.750 on solo frames → 0.581 pooled), and it
scales with track length: the longer and more mobile the staff track, the more frames it contains
where the stripe faces away or is occluded, and the lower the pooled fraction. Lowering the
threshold under ~0.34 to catch track 4 was **not** done — the separability evidence covers 4 staff
sightings and 5 customers, which is not enough to justify pushing the threshold that far, and the
downside (flagging a real customer) is worse than the current overcount. **Quantified limitation:
staff false positives 0; staff false negatives ~3 of 4 in the crowded window.**

**2. People at the ROI edge counted as in-zone.** A person standing visually outside the kiosk
zone is boxed and counted (3.4s dwell, just over the 3.0s `min_dwell_s` gate). The Phase 6 depth
sweep already established `box_depth_frac` **cannot** fix this: every value above 0.4 costs a
recovered identity in both pipelines (v2 4/4 → 3/4) and 5.4 points of baseline coverage, because
the in-zone depth distribution is bimodal and the remaining marginal boxes are people genuinely
standing at the boundary. The real lever is the **shape of `roi_polygon` at the kiosk-side edge**
(and secondarily `min_dwell_s`, since this track cleared it by 0.4s) — a re-authoring job against
the overlay, not a threshold sweep. Left unchanged and logged.

### ROI re-authoring attempted and rejected — the polygon is not the lever

The crowded overlay shows a track (ID 1) whose box is only half inside the zone on some
frames, which looked like a zone-geometry problem. Three candidate polygons were authored
and scored against the harness. All are worse than the production polygon:

| polygon (top edge vertices) | crowded count_err | matched | GT lost |
| --------------------------- | ----------------- | ------- | ------- |
| **production** (704,332) (1017,388) | +3 | **4/4** | — |
| (704,345) (1017,420) | +2 | 3/4 | **P8** |
| (704,340) (1017,410) | +3 | 3/4 | **P8** |

Sparse and Slice B are entirely insensitive to these edits (zero frames changed), so the
effect is confined to the crowded window.

The geometry explains why no edit along that edge can work:

| track | depth-margin median | frames below the 0.40 gate |
| ----- | ------------------- | -------------------------- |
| track 7 (GT **P8**'s tail) | 0.619 | **0%** |
| track 1 (the "half-in" ID 1) | **1.000** | **1%** |

**ID 1 is not an edge case.** It sits fully inside the zone for 99% of its 4240 frames; the
frame that looks wrong is a 1% transient where the person stepped to the back. P8's track
sits *higher* in the image (feet-y median 481 vs ID 1's 631), squarely in the band a lowered
top edge cuts through. So lowering that edge barely touches ID 1 and reliably destroys P8 —
whose 11s tail matches at IoU 0.565, just over the 0.5 floor, and collapses to 0.23–0.26.

ID 1 is a **fragmentation artifact, not a geometry one**: a spurious track thrown off by P11
and P12 being co-present for the whole window, matching no GT person under any polygon
tested. It is part of the residual +3 already attributed to the co-presence ceiling, and it
would take better identity association to remove, not a tighter zone.

**Decision: keep the production polygon.** Logged as a rejected avenue with its evidence so
it is not re-attempted.

Method note: `preview_roi.py`'s window-level table is an approximation and was wrong here —
it reported track 7 as shrinking when a real re-run lost it entirely, because the pipeline
re-tracks from scratch and fewer in-zone frames change association, stitching and the
stationarity outcome. Per-frame IN/OUT is exact; window-level claims need a re-score.

## Full-video run — main's pipeline on a T4 (the authoritative result)

YOLOv8n + ByteTrack + crowding-invariant stitch + OSNet long-gap merge, run end-to-end over
all 216,306 frames on a Colab T4 with `configs/cam1.yaml` exactly as committed — original ROI
polygon (bottom edge 716, **not** the Phase 6 fix) and `min_staff_frame_frac: 0.7`.

| metric | value |
| ------ | ----- |
| GT people | 18 |
| predicted | 41 |
| **count_error** | **+23** |
| **matched** | **9/18** |
| dwell MAE | 82.9s |
| dwell MAPE | 26.91% |
| staff flagged | 1, **0 false positives** |
| **throughput** | **34.79 FPS** (216,306 frames in 6218s = 1.73h) |
| peak VRAM allocated | 0.194 GB |
| peak VRAM reserved | 0.331 GB |
| **peak VRAM device (nvidia-smi)** | **0.493 GB** |

### This supersedes the Phase 3 "Final results" row

That row (157 tracks, count_error +139, 4/18 matched, dwell MAE 46.5s) predates the Phase 5
work now on main — the OSNet long-gap merge, the crowding invariant, the aspect gate and the
dominant-segment staff verdict. Measured end-to-end with all of it:

| | Phase 3 final | **this run** |
| --- | --- | --- |
| predicted | 157 | **41** |
| count_error | +139 | **+23** |
| matched | 4/18 | **9/18** |
| dwell MAE | 46.5s | 82.9s |

**The dwell MAE increase is largely an artifact and should not be reported as a regression.**
MAE is computed over matched pairs only; going from 4 to 9 matched people pulls in five harder
cases that previously contributed nothing to the average. The two figures are not computed over
comparable populations.

### Constraint compliance

- **Real-time: yes.** 34.79 FPS against the 30.077 fps source rate, i.e. the pipeline processes
  a 2h recording in 1.73h on the T4 edge target.
- **VRAM: 0.493 GB, 3.1% of the 16 GB budget.** This is the first *device-level* measurement in
  the project (sampled from nvidia-smi throughout the run) rather than PyTorch's allocator
  counters. The gap between the two is **0.162 GB**, consistent with the 0.114–0.148 GB seen on
  the window runs and matching the size of a CUDA context. Confirms the earlier finding: the
  allocator figures were sound but understated true footprint by roughly that fixed amount.

### Two caveats on this number

- **The ROI fix is not in it.** `cam1.yaml` still carries the 716 bottom edge. On the window
  tests that fix took the sparse window from 0/1 matched and 18.5% coverage to 1/1 and 100%, so
  a full-video run with `cam1_roifix.yaml` should be better again. Not yet measured end-to-end.
- **Staff recall is low: 1 flagged, against 7 in the Phase 3 run.** Zero false positives, so
  nothing is wrongly excluded, but this is the dilution mechanism already quantified in Phase 6 —
  the long-gap merge produces longer staff tracks that include frames where the chest stripe is
  not visible, pulling the pooled fraction under the 0.7 threshold. `cam1_v2.yaml` uses 0.5 on
  measured separability; `cam1.yaml` was deliberately left at 0.7 to keep it identical to main.

## Full-video run — baseline + ROI fix (isolating the ROI correction)

Same pipeline as the authoritative run above (YOLOv8n + ByteTrack + stitch + OSNet merge),
with `configs/cam1_roifix.yaml` — identical to `cam1.yaml` except the ROI bottom edge is
carried to 721 instead of 716. `min_staff_frame_frac` stays at 0.7, so the only change from
the baseline row is the ROI. Full 216,306 frames on the same Colab T4.

| metric | baseline (716) | **+ ROI fix (721)** |
| ------ | -------------- | ------------------- |
| predicted | 41 | 46 |
| count_error | +23 | **+28** |
| **matched** | 9/18 | **13/18** |
| dwell MAE | 82.9s | 97.5s |
| dwell MAPE | 26.9% | 29.1% |
| staff flagged / FP | 1 / 0 | 1 / **0** |
| FPS (T4) | 34.79 | **31.54** |
| peak VRAM device | 0.493 GB | 0.495 GB |
| boxes in render artifact | 119,388 | **203,573** |

**The ROI fix recovers four more people (9/18 -> 13/18 matched), the largest single jump in
match rate in the project.** It is what the near-edge coverage gap predicted: 71,961 -> 102,225
frames carry a box, and total boxes rise 70%, because people cut off at the near edge are no
longer gated out. This is the same fix that took the sparse window from 0/1 to 1/1 matched.

**Trade-offs, stated honestly:**
- **count_error rises +23 -> +28.** Recovering four occluded/edge people also spawns a few more
  fragments the stitch does not fully re-merge — the familiar coverage-vs-count tension. Match
  rate is the more meaningful metric here (it counts distinct real people found); the extra +5
  is residual fragmentation, not five phantom customers.
- **dwell MAE rises 82.9 -> 97.5s. Same averaging artifact as before, amplified:** MAE is over
  matched pairs only, and going 9 -> 13 matched pulls in four harder edge/occlusion cases that
  contributed nothing before. Not a real per-person degradation.
- **Throughput 34.79 -> 31.54 FPS.** Expected: 70% more in-zone boxes, each embedded by OSNet.
  Still above the 30.077 fps source rate, so still real-time on the edge target.
- **VRAM unchanged at 0.495 GB (3.1% of budget); staff still 1 flagged, 0 false positives.**

**Read: the ROI fix is a clear win on the primary goal (distinct people found) at a modest,
well-understood cost in count_error and ~9% throughput, both inside constraints.** It should be
in the production config. The count_error rise is a reason to revisit the stitch thresholds on
the full video, not a reason to keep the 716 edge that silently drops edge-standing customers.

### Correction to the entry above — that run was ROI fix + staff 0.5, not staff 0.7

The comparison table above was mislabelled. It was written after checking the *committed*
`cam1_roifix.yaml` (staff 0.7), but the Colab run edited its own copy to `min_staff_frame_frac:
0.5` before running (the "best final" config). Evidence the run used 0.5, not 0.7:

- The baseline run (staff 0.7) flagged one short staff track (461, 132 frames). This run
  flagged a **different, long** track (398, spanning 91712–96380 ≈ 4668 frames) — precisely the
  dilution case 0.5 recovers and 0.7 misses. Staff false positives remained 0 (track 398 matches
  no GT customer).

So the run is **YOLOv8n + ByteTrack + ROI fix + staff 0.5 — the best configuration of the
baseline detector/tracker family**, not a clean single-variable ROI isolation. Two knobs differ
from main's committed pipeline (ROI edge and staff threshold), so:

- The **+4 matched (9→13)** is still attributable to the ROI fix — the staff threshold only moves
  tracks between the customer and staff buckets and cannot *add* customer matches.
- The staff-threshold change is what flips the flagged staff track from the short 461 to the real,
  long 398, at no false-positive cost — the staff recall improvement, on the full video.

This is **not** the RF-DETR + BoT-SORT (v2) pipeline. "Best final" here means the best of the
*old* detector/tracker family. v2 is a separate pipeline, evaluated per-window via the cached
configs. The full comparison is: main (as committed) vs this best-old-family config vs v2.

## Phase A (in progress) — ReID backbone bake-off: does more compute fix ID switching?

Motivation: the full-video overlay shows ID switching and dwell carried over onto people as
they "enter" (id 50 reappears 84s after leaving already showing 317.8s dwell — the long-gap
OSNet stitch merging a re-appearance, sometimes onto the wrong person). VRAM headroom is huge
(~0.5 of 16 GB), so the hypothesis was: swap the smallest OSNet (x0.25) for a stronger ReID
backbone and open up the same-vs-different separation that bounds merge quality.

`diagnose_reid.py` (new, read-only) measures, per backbone, the figure of merit from Phase 5:
same-person 10th-pct cosine minus different-person 90th-pct cosine. **Positive margin = a
threshold cleanly separates identities = the ID problem is solvable.** Reproduced the known
negative baseline first (validates the method).

**Result — YOLO boxes, crowded window (103000-109000), 603 same-person / 667 different-person
pairs:**

| backbone | same median | diff median | **median gap** | margin (p10−p90) | size |
| -------- | ----------- | ----------- | -------------- | ---------------- | ---- |
| osnet_x0_25 (current) | 0.684 | 0.670 | 0.014 | −0.20 | 2.9 MB |
| osnet_x0_5     | 0.696 | 0.662 | 0.034 | −0.223 | ~5 MB |
| osnet_x1_0     | 0.722 | 0.692 | 0.030 | −0.20 | ~9 MB |
| osnet_ain_x1_0 (domain-general) | 0.682 | 0.637 | 0.045 | −0.231 | ~9 MB |
| osnet_ibn_x1_0 | 0.634 | 0.583 | 0.051 | −0.271 | ~9 MB |
| resnet50_fc512 | 0.912 | 0.906 | 0.006 | **−0.079** | ~90 MB |
| clip_market1501 | 0.478 | 0.416 | 0.062 | −0.37 | ~350 MB |

**Every backbone has a negative margin, and a tiny median gap (≤0.06).** On this crowded
top-down window, same-person crops are on average only ~0.01–0.06 cosine more similar than
different-person crops — barely any identity signal, *regardless of model size or type*. A 120×
bigger model (CLIP) does not help; resnet50 crushes everything to ~0.91 (indiscriminate); the
domain-generalizable OSNet variants give a marginally larger median gap but worse tails. This is
a property of the **data** (top-down viewpoint, similar clothing, mutual occlusion, co-presence),
not model capacity — exactly the risk flagged in the plan.

**Preliminary suggestion: stronger ReID does NOT give a genuine improvement on crowded scenes.**
The extra compute has no useful lever here through appearance. This aligns with the Phase 5
finding that count is a merge-algorithm bottleneck with a co-presence ceiling temporal GT can't
cross; the bake-off now adds that the appearance signal itself is too weak on this camera for any
backbone to break that ceiling.

### NOT YET DONE — the fair test, and tomorrow's resume point

The crowded window is arguably the *wrong* test for the stitch's actual job. The stitch targets
**long-gap re-entries** (a customer leaves, returns minutes later), where co-presence is low and a
returning person genuinely looks like themselves. That is where a better backbone could still
plausibly help, and it is untested. **Resume tomorrow with:**

1. Re-run `diagnose_reid.py` on a **re-entry window** (the dense slice 26700-41960 covers P4-P9's
   multiple visits) — same backbones, YOLO boxes. This is the decisive test.
2. Repeat on **RF-DETR boxes** (`configs/cam1_v2_cached_crowded.yaml` and a cached re-entry
   window) — cleaner boxes may give better crops; not yet measured.
3. **Only if a positive (or clearly larger) margin appears there** does Phase B/C proceed (swap
   backbone, retune gap/similarity, re-score windows + full video, then T4 FPS/VRAM). If the
   re-entry margins are also negative/tiny, the honest conclusion is that the ID limitation is a
   viewpoint/data problem, not a compute one — document it as a quantified ceiling and stop.

Commands to resume:
    .venv-boxmot/bin/python3 diagnose_reid.py --config configs/cam1.yaml --slice 26700 41960 \
      --backbones weights/osnet_x0_25_msmt17.pt weights/osnet_x1_0_msmt17.pt \
      weights/osnet_ain_x1_0_msmt17.pt weights/resnet50_fc512_msmt17.pt weights/clip_market1501.pt \
      --out outputs/reid_sep_yolo_reentry.json

### Correction — the crowded null is probably a confounded measurement, not proof ReID fails

Pushed on why every backbone (2.9 MB OSNet to 350 MB CLIP) gives ~0 margin. A flat null
across a 120x model-size range is a red flag for the *instrument*, not the data — ReID
demonstrably works (90%+ mAP on Market/MSMT). Three confounds make the crowded window the
wrong test:

1. **Label noise from co-presence (dominant).** With P7/P8/P11/P12 all at the kiosk at once,
   temporal-GT assignment cannot say which co-present person a track actually is, so a fraction
   of "same-person" and "different-person" pair labels are flipped. Randomly-flipped labels
   collapse the measured margin toward zero for ANY backbone — this alone can produce the null.
2. **Degraded crops.** Crowd boxes are occluded / clipped / straddling two people; ReID models
   are trained on clean full-body crops. Garbage in.
3. **Cross-domain penalty (real but partial).** Market/MSMT are street-level side-view; this is
   top-down CCTV. Out-of-distribution reduces — but does not zero — discriminability.

The re-entry test (tomorrow) removes confounds 1 and 2: one person present (reliable label),
clean full-body crop. To *prove* the confound rather than assume it, tomorrow's re-entry run
should additionally restrict same/different pairs to **solo-present tracks** (frames where GT
says exactly one person is in the window), which cleanly separates "measurement artifact" from
"real ReID failure". Only after that is a verdict on stronger-ReID defensible.

### User observation (for tomorrow) — two distinct ID-switch modes, only one is ReID-fixable

User reports ID switches are frequent **when one person passes in front of another**, plus a
separate case where **a new person enters carrying an old person's ID**. These are different
problems with different fixes; do not conflate them tomorrow.

**Mode 1 — crossing / pass-in-front swap (frame-level, in the TRACKER).**
Two people's boxes overlap during a crossing; the tracker's association swaps IDs. Critical:
in main's pipeline **the ReID backbone is not in the association path** — ByteTrack associates on
motion+IoU only, and OSNet runs only in the post-hoc stitch, which links an ended track to a
later-started one. During a crossing both tracks are simultaneously active, so the stitch's
no-simultaneous-merge invariant refuses to touch them. => **Swapping reid.model cannot fix
crossing swaps in main.** The only appearance lever for crossings is appearance-IN-association
(BoT-SORT / v2), and even that is degraded by occluded crops during the crossing. Tomorrow:
quantify whether v2 (BoT-SORT) has fewer crossing swaps than main on a crossing clip — that is a
tracker question, not a backbone question.

**Mode 2 — new person inherits an old ID (wrong re-link, in the STITCH / re-activation).**
This is the id-50 / dwell-carryover bug: the gap-stitch merges a new person onto an old ID when
appearance similarity crosses threshold. **This is the one a stronger backbone can genuinely
help** — if it separates the two on clean crops. This is exactly what the re-entry test targets.

**Calibrated expectation set with the user: "no ID switches" is NOT a realistic target on this
top-down camera.** A stronger backbone may reduce Mode 2 (wrong re-links); it will do little for
Mode 1 (crossings), which is a tracker/occlusion problem. Report this split honestly rather than
promising switch-free tracking.

Tomorrow's order: (1) re-entry ReID separation test with solo-present restriction (Mode 2, the
backbone question); (2) crossing-swap count main vs v2 (Mode 1, the tracker question); then a
combined, honest verdict on what the extra compute actually buys.

### Crossing-switch prevention + backbone leverage differs by pipeline (for tomorrow)

**Preventing crossing swaps — levers, best-first for this camera:**
1. Appearance IN association (BoT-SORT/v2 has it; ByteTrack/main does not — motion+IoU only).
2. NMS-free detection (RF-DETR/v2 keeps two boxes where YOLO+NMS merges the crossing pair into
   one, forcing an identity drop).
3. Tracker choice (Phase 5 bake-off already picked BoT-SORT; occlusion-specialised trackers help
   only marginally — appearance-under-occlusion is the real bottleneck).
4. Ceiling: crops are occluded *during* the cross, so appearance is degraded exactly when needed;
   crossings are reducible, not eliminable, on a top-down camera.
=> The crossing fix is largely "use v2" — it already carries levers 1 and 2. Tomorrow's Mode-1
test is therefore a tracker/architecture comparison (crossing-swap count main vs v2), not a
backbone sweep.

**Backbone leverage is asymmetric across pipelines — investigate BOTH but weight v2:**
| | main (YOLO+ByteTrack) | v2 (RF-DETR+BoT-SORT) |
| backbone in association? | NO (post-hoc stitch only) | YES (BoT-SORT) |
| can backbone fix crossings (Mode 1)? | no | possibly |
| can backbone fix wrong re-links (Mode 2)? | yes | yes |
In main a better backbone only touches re-links; in v2 it can touch both crossings and re-links,
and RF-DETR's cleaner boxes give better crops. So the ReID bake-off has more leverage on v2.
Tomorrow: run diagnose_reid.py separation on BOTH YOLO boxes and RF-DETR boxes (only YOLO-crowded
done tonight), on the re-entry window with the solo-present restriction.

## Phase A complete — the solo-restricted test (the confound control)

Ran the separation bake-off with the solo-present control: pooled 4 low-co-presence windows
(85000-91000 P10, 26700-41960 P1, 41960-56000 P4/P5, 135404-155000 P13/P15), restricting
same/different pairs to SOLO tracks (>=60% of frames with exactly one GT person present) so
labels are unambiguous AND crops unoccluded. 17 solo tracks, 46 same / 87 different pairs.

**SOLO-restricted separation (clean labels, clean crops) — the fair test:**

| backbone | same median | diff median | **median gap** | margin (p10−p90) |
| -------- | ----------- | ----------- | -------------- | ---------------- |
| osnet_x0_25 (current) | 0.705 | 0.580 | 0.125 | −0.128 |
| osnet_x1_0     | 0.722 | 0.552 | 0.170 | −0.142 |
| osnet_ain_x1_0 | 0.707 | 0.532 | 0.175 | −0.126 |
| resnet50_fc512 | 0.898 | 0.867 | 0.031 | −0.058 |
| **clip_market1501** | 0.694 | **0.281** | **0.413** | −0.159 |

**Two corrections to earlier conclusions, both from this table:**

1. **The crowded null WAS largely a confound.** Crowded median gaps were 0.006-0.062; on clean
   solo crops they rise to 0.125-0.413. Co-presence label noise + occlusion were suppressing the
   measured signal, exactly as predicted. My "no backbone separates" (crowded-based) was wrong.

2. **But "a bigger backbone fixes it" is also wrong — it's in between.** No backbone achieves
   *clean* separation (every p10−p90 margin is still negative: the tails always overlap, because
   even on this camera some same-person crops look different, pose to pose, and some different
   people dress alike). So switch-free tracking is not achievable by ReID here. HOWEVER, at the
   **median**, a stronger backbone genuinely discriminates far better: **CLIP-ReID's median gap
   0.413 is 3.3x the current osnet_x0_25's 0.125.** osnet_x1_0 / ain (~0.17) help modestly;
   resnet50 is useless (features crushed to ~0.9); CLIP is the clear standout.

**Operating-point reading (why the median gap matters for the stitch).** The stitch merges when
cosine > min_similarity. Current osnet_x0_25: same_median 0.705 vs diff_median 0.580 — so the
0.6 threshold sits between them and **over-merges different people** (the id-50 wrong-re-link /
dwell-carryover bug). CLIP: same_median 0.694 vs diff_median **0.281** — a threshold ~0.5
excludes most different-person pairs while keeping most same-person ones: a genuinely usable
operating point the current model does not offer.

**Asymmetry that decides WHERE to use it.** On the ALL (co-present / occluded) set, CLIP is the
WORST (margin −0.254) — it is the most occlusion-sensitive. So CLIP helps on CLEAN crops
(long-gap re-entries, Mode 2) and hurts on OCCLUDED crops (crossings, Mode 1). => Use CLIP in the
**post-hoc stitch** (main pipeline; end-of-track crops are cleaner), NOT in frame-level
association. For v2/BoT-SORT (appearance in association on possibly-occluded crops), CLIP is risky.

### Suggestion (honest)

There IS a genuine, evidence-backed improvement available, but bounded:
- **Swap the post-hoc stitch's ReID backbone from osnet_x0_25 to clip_market1501** (main pipeline).
  Expected: fewer wrong re-links / dwell-carryover (Mode 2), because CLIP's 3.3x-better median
  separation lets the merge threshold reject different people the current model waves through.
- **Realistic expectation, stated plainly:** this REDUCES wrong re-links; it does NOT eliminate ID
  switches. Crossings (Mode 1) are a tracker/occlusion problem CLIP can't fix (and would worsen if
  put in association). No backbone gives switch-free tracking on a top-down camera — the tails
  overlap irreducibly.
- **VRAM:** CLIP-ReID ~350 MB, still <3% of the 16 GB budget — the headroom is usable.
- **Cost to verify (needs Colab/T4, per the user's instruction):** (Phase B) run the pipeline on
  the 3 windows + full video with reid.model=clip_market1501, score wrong-re-link / dwell / purity
  vs the osnet baseline; (Phase C) measure the T4 FPS hit — CLIP is much slower per crop, but the
  stitch embeds only in-zone track crops, so it is bounded. Commands provided to the user.

## Phase B (local A/B) — a backbone swap in the stitch is NOT worth a full run

Ran the decisive local test the user asked for: detection+tracking ONCE on the dense re-entry
slice (26700-71000, 9 GT, 274 raw tracks, P4-P9 have genuine multi-visit re-entries — the exact
wrong-re-link case), embedding every crop with all candidate backbones in the same pass, then
replaying the exact post-stitch pipeline across a min_similarity sweep. Only the embedding
differs, so any change is the backbone's. gap 3000 / anchor 400 throughout.

**Current production point: osnet_x0_25 @ sim 0.6 -> count_err +4, 4/9 matched, mean_pur 0.937,
min_pur 0.518.**

| backbone | sim | count_err | matched | mean_pur | min_pur |
| -------- | --- | --------- | ------- | -------- | ------- |
| osnet_x0_25 (current) | 0.6 | **+4** | 4/9 | 0.937 | 0.518 |
| osnet_x1_0 | 0.6 | +3 | 5/9 | 0.920 | 0.550 |
| osnet_ain_x1_0 | 0.6 | +8 | 5/9 | 0.950 | 0.597 |
| osnet_ain_x1_0 | 0.7 | +15 | 7/9 | 0.991 | 0.901 |
| clip_market1501 | 0.5 | +13 | 7/9 | 0.969 | 0.655 |
| clip_market1501 | 0.6 | +18 | 6/9 | 0.989 | 0.780 |

**No backbone Pareto-beats the current one.** Every model sits on the SAME count-vs-purity
frontier: to merge more genuine re-entries (raise matched, cut count_err) you must lower the
threshold, which also merges more look-alike different people (drops purity), regardless of
backbone. The Phase A median-gap advantages do NOT convert:

- **CLIP under-merges.** Its 3.3x-better median separation is offset by a wide *same-person*
  tail (pose/viewpoint sensitivity, same_p10 0.28 from Phase A): genuine re-entries fall below
  threshold, so CLIP needs sim 0.5 to reach 7/9 matched and pays count_err +13 (vs current +4).
  It never reaches the current +4 count at any threshold.
- **Bigger/domain OSNet is within noise.** osnet_x1_0 @ 0.6 (+3, 5/9, min_pur 0.550) shifts by
  <=1 count and <=0.03 purity vs current — inside single-slice noise, not a real win.
- **The worst wrong-merge (min_pur ~0.5) is not fixable without ballooning count.** To get
  min_pur > 0.8 any backbone needs sim ~0.7, costing count_err +12 to +15.

**Cost of CLIP if run anyway:** 5.4x slower per crop (7.79 vs 1.44 ms/crop MPS). Embedding is a
small fraction of pipeline time (in-zone boxes only, ~2% of the T4 run), so the full-video T4
estimate is ~2.3-2.5 h (vs OSNet 1.9 h) at ~26 FPS — still real-time, VRAM ~0.6 GB. But it buys
no accuracy, so the estimate is moot.

**Verdict: do NOT run a full-video pass with a new ReID backbone.** The wrong-re-links the user
sees are a property of the precision/recall frontier on this top-down camera, not a
backbone-capacity limit — reconfirming, at the operating level, the Phase A finding. The local
A/B (a ~15-min diagnostic) saved a ~2.5 h T4 run that would not have improved on the current best.

## Mode-1 crossing test + the single root-cause factor (concrete evidence)

### Crossing swaps: v2 vs main — v2 is WORSE, and that is informative

`diagnose_crossing.py` counts crossing swaps from the per-frame artifacts (an id's centroid
jumping while another track is beside it — a proxy, but symmetric across pipelines). Crowded
window:

| pipeline | customer tracks | crossing swaps (jump>30px) | per track-min |
| -------- | --------------- | -------------------------- | ------------- |
| main (YOLO+ByteTrack) | 5 | 1 | 0.13 |
| v2 (RF-DETR+BoT-SORT) | 8 | 46 | 3.85 |

This overturns the earlier hypothesis that v2's appearance-in-association would REDUCE
crossings. It does the opposite, for two compounding reasons:
1. **main under-separates.** YOLO+NMS merges crossing people into one box, so main tracks only
   5 people (v2 tracks 8). main has fewer crossing *events* because it loses the second person
   entirely — its "0-1 swaps" is under-detection, not clean tracking.
2. **v2's appearance association can't hold IDs apart during a crossing**, because the crops are
   occluded exactly then, and appearance under occlusion is the weakest case (Phase A: CLIP/OSNet
   worst on co-present/occluded crops). So BoT-SORT falls back to IoU/motion and swaps.

Neither pipeline solves crossings; they fail differently (main merges, v2 swaps). The cause is
the same as everything else below.

### THE single factor, with hard evidence from the full run

Question: why do counting and dwell fail, why does "a new person get an old person's id", and
why wouldn't a better / larger-window ReID fix it? Measured on the full-video run's own
embeddings (`outputs/colab_full/stitch_state.pkl`, 297 raw tracks, OSNet), over every
disjoint-in-time re-link candidate pair, labelled same/different by GT:

**1. Appearance does not discriminate identity on this camera.**
- same-person re-link cosine: median **0.640**; different-person: median **0.615**. A 0.025 gap.
- At the production merge threshold 0.6: 65% of same-person pairs merge (correct), but **57% of
  different-person pairs also merge** — a new person taking an old id.

**2. The spatial gate is dead — everyone stands in the same spot.**
- entry-anchor distance between different-person candidates: median **156px** (same-person 150px —
  identical). **99%** of different-person pairs fall inside the 400px anchor gate. Because the
  kiosk queue is one small standing area, location cannot separate a returning customer from a
  new one.

**3. The base rate makes it unwinnable.** Different-person re-link candidates outnumber same-person
**5.9 : 1** (25,622 vs 4,316). To avoid wrong merges an appearance filter would have to reject
>83% of different-person pairs while keeping same-person ones — impossible when their cosine
medians are 0.640 vs 0.615. **Of all candidate pairs that clear the 0.6 gate, ~84% are different
people.**

**So the single factor is: on this top-down kiosk camera, two different customers are
indistinguishable by BOTH location (one shared standing spot) and appearance (the viewpoint
captures head/shoulders, not discriminative full-body clothing), so the re-link decision is
near-random.** This one factor produces every symptom:
- **Counting:** wrong merges (new person → old id, overcount masked) and missed merges (returning
  person → new id, overcount) — both from the same non-discrimination.
- **Dwell:** a wrong merge dumps a new person's frames onto an old id (inflated dwell); a missed
  merge splits one person's visits (fragmented dwell). The 82.9s dwell MAE is this, not a bug.
- **Crossings:** the same appearance failure, made worse by occlusion.

### Why a better / larger-window ReID backbone cannot fix it — directly refuted

- **Bigger window makes it WORSE, not better.** Widening `gap_frames` adds more candidate old-ids
  for a new person to be wrongly merged with, and different-person candidates already outnumber
  same-person 5.9:1 and grow faster with window size. The v2 stitch sweep already showed
  gap 3000 -> 9000 gave zero count improvement. The window is not the constraint.
- **Bigger backbone cannot recover information that is not in the crop.** Phase A: no backbone
  (up to CLIP, 120x larger) separates same from different on clean solo crops — the top-down
  viewpoint simply does not capture the discriminative signal ReID models rely on. Phase B: no
  backbone beats the current one at the stitch operating point. The 84% wrong-candidate rate is a
  property of the pixels, not the model.

**Conclusion for the write-up:** the ID / dwell limitation on this video is a *camera-geometry*
ceiling — a single shared standing area viewed top-down — not a model-capacity or window-size
problem. It is quantified (0.640 vs 0.615 cosine; 99% within the spatial gate; 5.9:1 base rate;
84% of gated merges wrong) and is not addressable by a stronger or larger-window ReID. The
current pipeline already operates near this ceiling; the honest deliverable is to document it.

## Conservative-stitch test — a real in-scope lever, and a correction

Motivation: the full run showed ~84% of long-gap merges are wrong, so the aggressive stitch
may be net-harmful to DWELL. Swept stitch aggressiveness (gap_frames x min_similarity; gap 0 =
stitch off), measuring the metric wrong merges corrupt most: dwell. Confirmed on the FULL video
offline by replaying the stitch on the real full-run tracks (`colab_full/stitch_state.pkl`, 297
raw tracks) — no re-run, no T4.

**Full video (216k frames, 18 GT), replay of the real tracks:**

| gap | sim | count_err | matched | dwell MAE | dwell MAPE | mean purity |
| --- | --- | --------- | ------- | --------- | ---------- | ----------- |
| 0 (stitch off) | 0.6 | +204 | 3/18 | 115.4s | 37.5% | 0.826 |
| 900 | 0.6 | +60 | 8/18 | **67.4s** | **18.4%** | 0.678 |
| 900 | 0.7 | +84 | 7/18 | 65.3s | 22.2% | 0.692 |
| **3000 | 0.6 (CURRENT)** | **+23** | 9/18 | 82.9s | 26.9% | 0.615 |
| 3000 | 0.7 | +51 | 9/18 | 78.4s | 25.0% | 0.634 |

**Correction to "the pipeline is near its ceiling."** That was true for COUNT. It is NOT true
for DWELL: the current config sits at a count-optimised point that is *bad* for dwell. A more
conservative stitch (gap 900) cuts **dwell MAPE 26.9% -> 18.4%** and **dwell MAE 82.9s -> 67.4s**
(a ~32% reduction in dwell error) and lifts purity, because it stops the wrong merges that dump a
new person's frames onto an old id — exactly the failure the user reported.

**But it is a TRADE on the same frontier, not a free win.** Conservative merging avoids wrong
re-links by re-linking *less*, so genuine re-entries fragment: count_error rises **+23 -> +60** and
matched dips 9 -> 8. This is the camera-geometry ceiling seen from the other side — appearance
cannot tell a returning customer from a new one, so you get accurate dwell (don't merge) OR
accurate count (merge, 84% wrong), not both.

**So there IS an in-scope architectural lever, and it is a config change (gap_frames), no new
backbone and no T4:**
- For **dwell-time analytics** (the project's named goal), **gap 900 is better**: dwell MAPE
  18.4% vs 26.9%, fewer wrong-id merges (the user's complaint), at a higher count_error.
- For **distinct-people count**, the current gap 3000 is better (+23 vs +60).
- The choice is a product-priority decision, not a technical unknown. Both points are now measured
  on the full video.

**This corrects the earlier "only out-of-scope solutions remain."** The ReID-backbone avenue is
exhausted, but stitch POLICY is an in-scope lever with a measured ~32% dwell-error improvement
available for free. Out-of-scope options (second camera / overhead-tuned model) remain the only
path to beating the *count-vs-dwell frontier itself*; within it, the operating point is tunable.

## ReID / ID-switching investigation — consolidated summary and conclusions

A single reference for the write-up, tying together Phase A/B, the root-cause evidence, the
crossing test and the conservative-stitch result. The question driving it: the pipeline gives
new people the IDs of people who walked past earlier, corrupting count and dwell — can a stronger
/ larger-window ReID backbone fix it?

**Answer: no on the backbone; yes on the stitch policy, as a trade.**

1. **The failure has one root cause: camera geometry.** On this top-down camera with a single
   shared kiosk standing area, two different customers are indistinguishable by BOTH location and
   appearance. Measured on the full run's own tracks: same-person vs different-person re-link
   cosine medians 0.640 vs 0.615 (indistinguishable); 99% of different-person pairs sit inside the
   400px spatial gate; different-person candidates outnumber same-person 5.9:1, so ~84% of pairs
   clearing the merge gate are different people. This one factor produces every symptom — wrong
   counts, wrong dwell, and crossing swaps.

2. **A stronger/larger ReID backbone cannot fix it (exhausted avenue).**
   - Phase A: across 7 backbones from OSNet-x0.25 (2.9 MB) to CLIP-ReID (350 MB), no backbone
     cleanly separates same from different even on clean solo crops — the top-down viewpoint does
     not capture the discriminative signal. (The crowded null was partly a co-presence measurement
     confound, but the clean solo test still showed no clean separation.)
   - Phase B: replaying the stitch with each backbone, none Pareto-beats the current OSNet — CLIP
     under-merges (wide same-person pose tail), OSNet-family upgrades shift within noise.
   - A *larger window* makes it worse, not better: it adds more wrong-merge candidates.

3. **The tracker/architecture does not rescue it either.** v2 (RF-DETR + BoT-SORT) is the better
   pipeline overall (coverage, matched, dwell) and is what a visual review prefers, but it does
   NOT reduce crossing swaps — appearance-in-association fails under occlusion (exactly when a
   crossing needs it). main "avoids" crossings only by under-separating (merging people).

4. **The in-scope lever that DOES help is the stitch policy (not the backbone).** Because ~84% of
   long-gap merges are wrong, the current aggressive stitch (gap 3000) is net-harmful to dwell. A
   conservative stitch (gap 900) cut full-video dwell MAPE 26.9% -> 18.4% and MAE 82.9s -> 67.4s
   with higher purity, at a count cost (+23 -> +60). This is a count-vs-dwell trade on the same
   frontier — accurate dwell (merge less) OR accurate count (merge more, 84% wrong), not both.
   It is a one-line config change (`configs/cam1_conservative.yaml`), no backbone, no T4.

5. **Only beating the frontier itself is out of scope:** a second camera angle or an
   overhead-tuned/re-trained model — i.e. capturing information the current top-down view does not
   physically contain.

**Status:** a full-video conservative run (gap 900, else identical to the aggressive ROI-fix run)
is being produced locally for a side-by-side overlay comparison, to distinct output paths so the
aggressive run is preserved. Numbers + video to be appended when it completes.

## Conservative-stitch — full-video LIVE run confirms the offline prediction

Ran the full pipeline live (216,306 frames, local MPS, 73 min) with `configs/cam1_conservative.yaml`
(identical to the aggressive ROI-fix run except reid.gap_frames 3000 -> 900). Clean A/B, both
roifix + staff 0.5, only the stitch gap differs:

| metric | aggressive (gap 3000) | conservative (gap 900) |
| ------ | --------------------- | ---------------------- |
| count_error | +28 | **+62** |
| matched | 13/18 | **14/18** |
| dwell MAE | 97.5s | **72.5s** |
| dwell MAPE | 29.1% | **24.2%** |
| staff flagged / FP | 1 / 0 | **5 / 0** |

The live run confirms the offline replay directionally: **dwell MAE -25.6% (97.5 -> 72.5s), MAPE
-16.8% relative (29.1 -> 24.2%)**, and it additionally *improved* matched (13 -> 14) and staff
recall (1 -> 5 flagged, still 0 false positives — the shorter gap dilutes staff tracks less). The
cost is count_error (+28 -> +62), from genuine re-entries fragmenting into separate ids. Exactly
the count-vs-dwell trade, now verified on the full video.

Outputs went to `outputs/full_conservative/` and `kiosk_overlay_conservative.mp4`; the aggressive
run's `outputs/full_roifix/` and `kiosk_overlay_roifix.mp4` were preserved untouched. The two
overlays share identical detections and differ only in ID grouping, for a direct visual A/B.

**Bottom line for the deliverable:** if the priority is dwell-time accuracy (the project's named
goal), the conservative stitch is the better operating point (dwell MAPE 24.2% vs 29.1%, matched
14 vs 13, staff recall 5 vs 1) and costs only the distinct-people count. Both configs are locked
and reproducible; the choice is a product-priority decision, now backed by full-video numbers on
both sides.

## Correction — the conservative video's "missing people" is an MPS-vs-CUDA artifact, not the stitch

Visual review of the conservative overlay showed it detecting far fewer people than the
aggressive one. Investigated rather than accepted at face value, and it is a **device confound I
introduced**, not a property of the conservative stitch:

- **The stitch does not drop detections.** Replaying gap 3000 vs gap 900 on IDENTICAL raw tracks
  (the conservative run's own stitch_state): gap 3000 keeps 141,653 boxes, gap 900 keeps 141,051
  (97% vs 96% of raw in-zone boxes). The stitch only regroups ids; it does not gate people out.
- **The real cause: MPS detects ~28% fewer people than CUDA.** The aggressive video was rendered
  from the Colab **CUDA** run; the conservative one from a local **MPS** run. Measured: MPS has
  85% as many frames-with-a-person and **72% as many in-zone boxes** (145,784 vs 203,573), running
  ~1 person short on most crowded frames (3 vs 4, 3 vs 5). YOLOv8n on MPS materially under-detects
  versus CUDA — the Phase 5 "detection floats differ" note, far larger than a float difference at
  the box level.

**Consequences, stated honestly:**
1. **The conservative overlay I produced is not a fair visual A/B** — it is under-detected by the
   device it ran on, not by the pipeline. A fair conservative-vs-aggressive video needs BOTH on
   CUDA (Colab).
2. **My earlier live A/B (conservative dwell 72.5 vs aggressive 97.5) was device-confounded**
   (MPS conservative vs CUDA aggressive). The device-CLEAN A/B is the offline replay on identical
   CUDA detections — gap 900 dwell MAE 67.4s vs gap 3000 82.9s — which still shows the dwell
   improvement, so the count-vs-dwell conclusion holds directionally; the specific live numbers do
   not isolate the stitch.
3. **Lesson recorded:** local MPS renders are not representative of the CUDA pipeline for anything
   detection-sensitive (coverage, visual review, count). Authoritative videos and coverage numbers
   must come from CUDA. The conservative stitch remains a real dwell-vs-count lever (offline-proven),
   but its full-video artifacts should be regenerated on CUDA before being used for comparison.

## Phase 7 (separate experiment) — mask-memory tracking with SAM 2.1 / EdgeTAM

Recorded here for completeness: a parallel investigation (its own `sam-pipeline` branch /
`Tenyks-SAM-Experiment` project) tested whether a **video segmentation model's learned memory**
maintains identity through occlusion better than the appearance-embedding stitching that both
main (post-hoc OSNet) and v2 (BoT-SORT ReID) rely on — the exact identity limitation this project
kept hitting. Distinct from the FastSAM experiment (Phase 5 cont. 3, correctly ruled out): that
tested *unprompted* whole-frame segmentation (162 junk segments/frame); this tests *box-prompted
video propagation with memory*, where YOLOv8n still detects and prompts, SAM only maintains the
identity, and zone/stationarity/staff/dwell are reused unchanged.

- **Models:** SAM 2.1 Hiera-tiny (39M) primary, EdgeTAM (13.9M) edge variant, via transformers'
  `Sam2VideoProcessor` (swappable on a config string). Larger SAM 2.1 / SAM 3 ruled out (throughput;
  SAM 3 is a detector+tracker in one, changing two variables).
- **The load-bearing engineering finding — memory leak:** naive streaming grew live tensors
  0.42 -> 3.62 GB over 40 frames (~80 MB/frame), fatal for a 6000-frame window and the 16 GB budget.
  Fixed by pruning unreachable per-object outputs (older than ~16-24 frames are never read) and
  streamed frames, plus explicit object retirement + contiguous re-indexing (cost then scales with
  *concurrent* not *cumulative* people). After the fix: **~2 GB live tensors, bounded and flat over
  6000-frame runs.**
- **Results (EdgeTAM + YOLOv8n, MPS, 3 windows):**

| window | count_err | matched | coverage | purity | dwell MAE | FPS (MPS) |
| ------ | --------- | ------- | -------- | ------ | --------- | --------- |
| sparse | +0 | 1/1 | 100% | 1.000 | 0.1s | 10.9 |
| Slice B | +1 | 2/2 | 99.7% | 1.000 | 0.3s | 5.4 |
| crowded | **+9** | **4/4** | 99.9% | 0.999 | **3.2s** | 2.4 |

- **What it proved:** mask memory essentially SOLVES fragmentation — crowded dwell MAE **3.2s, the
  best any pipeline produced** (v2 10.0s, main 60.4s), with only 2 duplicate identities across
  6000 frames of a 4-person crowd (vs main's 38/32/16 fragments per person). Identity through
  occlusion is genuinely better than appearance stitching.
- **What it cost:** crowded count_err **+9 — the WORST any pipeline produced** (v2 +3, main +2), and
  by the opposite mechanism: better persistence tracks *non-customers* (companions, lingerers) too
  well, so they clear the 3s stationarity gate instead of shattering into sub-gate fragments as they
  do on main/v2. Against a *behavioural* GT (only kiosk users counted), better tracking hurts count.
- **Throughput:** slow — SAM 2.1-tiny ~1.9-2.4 FPS, EdgeTAM ~5-11 FPS (MPS). The brief has no
  real-time requirement (offline batch analytics), so this is a cost, not a disqualifier. VRAM
  against the 16 GB budget needs a CUDA run (MPS reads 0.0; ~2 GB live-tensor leak-check passes).

**Read:** mask-memory tracking is the one approach that broke the fragmentation/identity ceiling
appearance ReID could not (Phase A/B) — a genuinely different architecture, not a bigger backbone.
Its cost is throughput and a count regression that is really a *stationarity/behavioural-filter*
problem (too-good tracking of non-customers), not an identity failure. It is the strongest lead for
future work on the identity problem, and unlike a stronger ReID backbone it is evidence-backed to help.
