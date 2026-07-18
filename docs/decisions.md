# Decision log

One entry per module. Per CLAUDE.md, no module lands without a line here.

## 2026-07-16 — Skeleton + eval harness (no detector/tracker yet)

- `src/detect/base.py`, `src/track/base.py` — swappable interfaces as `Protocol`s
  (`Detection`/`Detector`, `Track`/`Tracker`), signatures only. No implementation
  yet: the baseline detector/tracker choice is a later, measured decision.
- `src/zones|dwell|overlay` — empty packages, placeholders for the pipeline stages.
- `eval/label/schema.py` — GT/prediction interval formats + YAML loaders.
  `PersonInterval` and `TrackInterval` kept as distinct types so a person/track
  mix-up can't slip through matching.
- `eval/metrics.py` — distinct-count error (signed pred - gt) and per-person dwell
  MAE / MAPE. Predicted tracks matched to GT persons by temporal IoU under a
  Hungarian assignment (`scipy.linear_sum_assignment`), with a `min_iou` floor so
  spurious tracks stay unmatched and show up as count error rather than corrupting
  dwell error.
- `eval/report.py` — prints and saves (CSV) a comparison table keyed by config name,
  so future configurations diff against the baseline.
- `eval/fixtures/{gt,pred}.yaml` + `eval/tests/test_metrics.py` — synthetic fixture
  with hand-computed IoUs and dwell errors that pins the metrics.
- `eval/run_eval.py` — standalone entrypoint to run the harness on a GT/pred pair
  before any pipeline exists.
- `configs/cam1.yaml` — example per-camera config (ROI polygon, detector conf,
  tracker, dwell + eval thresholds). Values are placeholders, not tuned to a clip.


## GT Labelling Protocol — Kiosk Enter/Exit Events

**Population labelled:** only people who queue for and/or use the kiosk. Anyone
who does not independently engage (walk-throughs, staff, companions/children who
never interact themselves) is excluded from GT entirely — not logged, not
timestamped.

**ROI:** the kiosk queue lane overlaps the shop entrance walking path in this
camera's layout (kiosks sit right at the entrance; see reference frame, CAM 1,
09/07/2026 17:27:18). Because geometry alone cannot separate customers from
pass-through shoppers here, ROI membership was NOT used as the inclusion rule.
Inclusion was instead decided by behavioural judgment: did the person stop,
queue behind another kiosk user, or face/interact with a kiosk screen? If yes,
included. If they maintained walking pace through the zone without stopping,
excluded, even if their path clipped the ROI polygon.

**Enter/exit timestamps:** for included people only, enter_frame = ROI entry
(joining the back of the queue), exit_frame = ROI exit after finishing at the
kiosk. This directly follows the task's own definition of dwell time
(queue wait + active use, combined).

**Accompanying persons:** a companion (e.g. a child standing with a parent who
is using the kiosk) is not logged separately unless they independently interact
with the kiosk screen themselves. Otherwise they're treated as part of the
primary user's visit.

**Returning people (leave and come back):** if the same person leaves the ROI
and later queues again, they keep the same person_id rather than a new one —
a new ID would double-count someone already counted, and person count is
defined as distinct people, not distinct visits. This produces multiple
(enter_frame, exit_frame) segments under one person_id. Brief step-outs
(<~2s) are not split into a new segment; longer gaps are. Segment durations
are summed per person_id at aggregation time, not reported per-segment, to
give the single combined dwell duration the task asks for.

**Tooling:** custom OpenCV labelling script (label_gt.py) supporting frame-seek
navigation, multiple simultaneously-active people, undo, and atomic
autosave/resume so a 2-hour video could be labelled across multiple sittings
without losing progress or duplicating IDs. A separate read-only viewer
(review_frames.py) was used to re-check specific enter/exit frames without
risk of mutating the saved GT file. A validator (validate_gt.py) automatically
flags unclosed people, exit-before-enter errors, overlapping segments under
one ID, and outlier-length segments for manual review before finalizing.

**Result:** 18 distinct people across a labelled span of ~216,300 frames
(~2hr @ ~30.08 fps). 8 of the 18 returned for a second (or third/fourth)
segment. All flagged long-duration segments (up to ~500s) were manually
re-verified against the footage and are genuine — this kiosk's dwell time
combines queue wait with a comparatively slow self-service ordering flow,
not a labelling error.

## Coordinate-space verification (baseline pipeline)

Confirmed ROI gating operates in native 1280x720 space, not the 640x384
letterboxed detector input:
- anchor() uses bottom-center of box: ((x1+x2)/2, y2)
- observed anchor/box values exceed letterbox bounds (e.g. y2=647.6,
  anchors at y=648), only possible post scale_boxes mapping to native frame
- indirect proof: placeholder ROI spans y~420-720; if anchors were still in
  letterbox space (y capped at 384) the gate could never return a non-empty
  in-zone set, yet a test slice returned 13 in-zone tracks
Conclusion: cam1.yaml's roi_polygon must stay authored in native 1280x720
pixels (matches what define_roi.py outputs) for the gate to remain correct.

## Baseline pipeline results (Phase 2)

Ran the naive baseline (YOLOv8n + ByteTrack, no ReID, no stationarity gate)
over the full 216,306-frame video and scored it against the 18-person hand-
labelled GT.

Numbers:
- 354 predicted tracks vs 18 GT people. count_error +336.
- Only 2/18 matched at IoU >= 0.5.
- dwell MAE 121.8s, dwell MAPE 46.8% — but this is only over the 2 matched
  people, so it's a thin, not-yet-trustworthy signal. Count and match rate
  are the real headline numbers right now, not dwell.
- Throughput 105.5 FPS on MPS, full video in ~34 min. Peak VRAM read 0.0GB,
  which is expected — torch.cuda.max_memory_allocated only reports on CUDA,
  so this needs re-measuring on the actual T4 target, and FPS will differ
  there too.

Both predictions from the pre-run decision log entry held up:

**No stationarity/min-dwell gate → the ROI counts walkers as customers.**
The ROI reaches toward the entrance walkway (unavoidable given the kiosk's
position at the door — see the earlier note on this). The gate is just
"feet inside the polygon," so anyone walking to a table, the counter, or
the door gets counted the same as someone queueing. Median predicted dwell
is 3.5s. 62% of the 354 tracks are under 5s — that's the walk-through
population, not kiosk users. This is the single biggest driver of the
overcount, bigger than fragmentation on its own.

**No ReID + a 1-second track buffer → every real person shatters into
multiple IDs.** ByteTrack is motion-only — a Kalman filter predicting
where a box should be next frame, matched to new detections by overlap.
It has no idea who anyone is, only where their box was. With
track_buffer=30 (1s at ~30fps), anyone occluded for longer than a second
comes back as a brand new ID rather than the same one. The kiosk area has
people constantly crossing in front of each other, so this fires
constantly. No predicted track covers more than IoU 0.06-0.55 of any real
person's actual presence — best case is barely over half.

The two effects compound rather than just add. A real visit gets fragmented
into several short tracks by occlusion, and separately, walk-throughs are
adding hundreds of tracks that were never a real person to begin with —
so the 354 number is "18 real people, badly fragmented" plus "a large pile
of things that were never customers," not one clean multiple of the other.

Worth noting on the matching side specifically: it's not just that
fragments are short. When multiple GT people are at the kiosk at the same
time (P7/P8/P9 all present 61k-67k), one predicted track spanning that
whole window is the closest match to all three of them, but one-to-one
Hungarian assignment can only give it to one. The other two drop out
entirely, even though a track existed nearby. So part of the 2/18 match
rate is a crowding problem specifically, separate from fragmentation
in general.

Confirmed this isn't an eval bug before trusting any of it: coordinate
spaces were already verified correct (see earlier entry), and the
low IoU isn't an artifact of how collapse_segments handles multi-visit
people — that logic only changes anything for people with a real re-entry
gap, and single-visit people keep their true span untouched. The
fragmentation is real, not a measurement artifact.

Ranked, what this means for Phase 3:
1. Add a min-dwell / stationarity gate (or tighten the ROI further away
   from the walkway) — biggest lever, attacks the majority failure mode.
2. Add appearance ReID and/or a longer track buffer with gap-stitching —
   fixes fragmentation of real users, and is also directly the skill the
   actual Tenyks project needs.
3. Crowding will need attention specifically for the matching metric, since
   even fixing 1 and 2 individually won't necessarily resolve one track
   incorrectly representing multiple simultaneous people.
4. Keep in mind the eval is intentionally strict (min_iou 0.5, one-to-one)
   — appropriate for establishing a true floor, but it's part of why 4
   good-ish matches becomes 2 counted ones.

## Baseline diagnostic deep-dive (diagnose_baseline.py, post Phase 2)

Cross-referenced predicted tracks.yaml against GT at the per-person level,
not just aggregate metrics. This corrects part of the earlier root-cause
ranking:

**Fragmentation is the dominant driver of the +336 overcount, not
pass-throughs.** 285 of 354 predicted tracks overlap a real GT person —
they are fragments of the 18 real customers, not false tracks. Only 69/354
have zero overlap with any GT person at all (candidate walk-throughs or
missed GT). The earlier "62% of tracks under 5s -> walk-throughs" reading
was misleading: it didn't separate short-because-fragment from
short-because-walker. The 69 true no-overlap tracks skew short as expected
(71% under 5s), consistent with genuine walk-throughs, but they are a much
smaller share of the problem than initially estimated.

Fragmentation severity correlates with crowd density. P1/P2/P3, all
present in the same ~15-minute dense window (frames ~26700-41960), shatter
into 41/35/21 tracks respectively. P16/17/18, in a lower-traffic stretch
later in the video, fragment far less (6-15 tracks) with much healthier
IoU (0.3-0.7). Confirms the crowding hypothesis directly: track 305
(62588-65827) is independently the best-overlapping track for three
different simultaneously-present GT people (P7, P8, P9) -- exactly the
one-track-many-people failure predicted before running the eval.

Re-entry / no-ReID hypothesis confirmed for 7 of 8 multi-segment people
(P4, P5, P7, P8, P9, P16, P18): each return after leaving gets a distinct
track_id, never the same one. The eighth, P6's third segment
(frames 75471-75650), has NO predicted track overlapping it at all -- a
genuine detection/tracking miss, not a fragmentation or ReID issue. This
needs separate investigation (likely a detector confidence gap or
occlusion by a kiosk stand) rather than being lumped in with the ReID fix.

Open item before finalizing Phase 3 scope: spot-checked several long
no-overlap tracks against the video directly (review_frames.py). [fill in
findings once checked, e.g. "track 680 (84.7s, frame 115464) is/is not an
unlabelled child" / "confirmed genuine walk-through" / "GT miss, added a
19th person".]

Revised priority for Phase 3, given this evidence:
1. ReID / appearance-based re-association — biggest lever by a wide
   margin now that fragmentation is confirmed as ~85% of the overcount,
   not ~55%.
2. Crowding-aware matching / track splitting during dense windows —
   specifically targets the P7/P8/P9-one-track case.
3. Min-dwell or stationarity gate for the remaining true pass-throughs —
   still worth doing, but a smaller lever than originally ranked.
4. Investigate the P6 segment #3 total miss separately — may point to a
   detector recall gap under specific occlusion, not a tracking problem.

   ## Baseline diagnostic — corrections after spot-checking no-overlap tracks

Spot-checked flagged long no-overlap tracks against the raw footage:

- P6 segment #3 (frames 75471-75650): person is clearly visible in the ROI
  in the source video. This is a genuine detector/tracker recall miss, not
  a labelling gap or fragmentation artifact. Needs its own root-cause check
  (occlusion, confidence threshold, lighting) separate from the ReID work.

- Track 680 (frames 115464-118011, 84.7s): confirmed staff, not an
  unlabelled customer or a labelling error. GT correctly excludes staff.
  However, the pipeline itself has no staff/customer distinction — it
  tracks anyone whose feet enter the ROI. Staff (identifiable by a
  consistent black uniform + kitchen hygiene headcover, including staff
  who actively assist customers at the kiosk) will keep appearing as
  false long-dwell tracks until filtered explicitly. Planned fix: a
  lightweight appearance-based heuristic (color/region classifier on the
  uniform, not a trained detector) applied per track, with explicit
  false-positive checking against the 18 known GT customers before
  trusting it.

Revised Phase 3 priority: ReID re-association (biggest lever, ~85% of
overcount) > staff-exclusion filter (new, cheap, explains part of the
no-overlap bucket) > min-dwell/stationarity gate for remaining true
walk-throughs > crowding-specific handling (re-assess after ReID) >
P6-style detector-recall investigation (separate, smaller).

## No-overlap track classification (69 tracks, full manual review)

Classified every predicted track with zero temporal overlap with any GT
person, using classify_tracks.py:
- staff: 11 (16%)
- walkthrough (genuine pass-by, not staff): 42 (61%)
- other/ambiguous: 11 (16%) — see below, this is not noise
- missed_customer (candidate GT gaps): 5 (7%)

**Staff confirmed as a real, distinct failure mode** — present both
in isolation and interleaved with active customer interactions, so
timing/context cannot be used as a proxy for staff detection. Uniform
(black outfit + kitchen hygiene headcover) is a strong, consistent visual
signal; planned fix is a lightweight appearance heuristic per track, not
a trained detector.

**"Other" bucket is mostly a third, previously-unidentified failure
mode: phantom tracks with no real person present.** 7-8 of the 11 were
noted as "no one in ROI" during manual review. Suspected causes: false
detections on kiosk-screen digital signage/imagery, or window
reflections, occasionally exceeding the 0.6 new-track confidence
threshold. Needs a quick direct check (what pixels is the box actually
drawn around) before deciding the fix — likely candidates are masking
out known screen/signage regions from the detector, or a stricter
zone-membership check (see below) that would incidentally kill these too
since a phantom box's edge can still clip the ROI boundary.

**ROI gating is confirmed too permissive — single feet-anchor point is
not sufficient.** Two independent lines of evidence: (1) all 42
walkthrough tracks are real people whose feet legitimately clip the
polygon while passing through, not queueing; (2) several phantom/ghost
tracks were only caught by the gate because a box edge grazed the
polygon boundary. Planned fix: require a meaningful portion of the box
(e.g. ~40% of box height from the bottom) to fall inside the polygon,
not a single boundary point. This is separate from, and additional to,
the previously-planned min-dwell/stationarity gate — the box-depth fix
targets edge-clipping specifically, the dwell/stationarity gate targets
people who walk fully through the ROI's interior (unavoidable given this
layout, per the earlier entrance-overlap analysis).

**5 candidate missed-GT-customers found, pending policy check.** Most
appear to be children. Before adding any to GT, each needs re-checking
against the existing accompanying-person policy: only counts as a
separate GT person if they independently interacted with the kiosk, not
merely present alongside an already-counted parent. GT person count
(currently 18) may increase, pending this check — re-run eval only
after this is resolved, since it changes the denominator for every
metric.

Revised Phase 3 scope, in priority order:
1. ReID / appearance-based re-association — dominant lever, ~285/354
   fragmented tracks.
2. Zone-membership hardening — box-depth check (not just feet point) to
   kill edge-clipping walkthroughs and phantom-track leakage.
3. Min-dwell / stationarity gate — remaining true walkthroughs whose
   path runs through the ROI interior.
4. Staff-exclusion filter — appearance heuristic on uniform/headcover.
5. Ghost-detection investigation — likely signage/reflection false
   positives; may resolve partly as a side effect of #2, needs a direct
   pixel check to confirm root cause.
6. GT reconciliation — resolve the 5 candidate missed customers against
   the accompanying-person policy before the next full eval run.
7. P6-segment-3-style detector recall miss — separate, smaller,
   unrelated to the above.
## Phase 3 Step 1 — Appearance-based ReID re-association

Fragmentation is ~85% of the baseline overcount (285/354 predicted tracks are
fragments of the 18 real people). This step re-associates those fragments by
appearance, as a post-process on top of the existing track/dwell output —
ByteTrack itself is untouched, per the eval-first / one-lever-at-a-time rule.

**Backbone choice: torchvision `mobilenet_v3_small`, classifier head removed,
ImageNet weights.** The obvious pick was OSNet (the standard lightweight ReID
net), but its only turnkey source, torchreid's `FeatureExtractor`, downloads
pretrained weights from Google Drive: the download hung for >2 min locally and
never completed, which both blocks the required local sanity check and is a live
risk on the Colab run the whole handoff workflow exists to protect. torchvision
hosts weights on download.pytorch.org (~10 MB, ~2 s, reliable). The backbone
with its head removed is 0.93M params and emits a 576-d global-pooled vector;
peak VRAM is a few MB on top of YOLOv8n, comfortably inside the 16 GB edge
budget. Cost per frame is one batched forward over the in-zone boxes only.

Trade-off, logged honestly: ImageNet features are less person-discriminative
than ReID-trained OSNet, and because they are post-ReLU/global-pooled they are
non-negative, so cosine similarity between unrelated crops sits high rather than
near zero. Mitigation is to never merge on appearance alone — see gates below.
The backbone is config-driven (`reid.model`), so OSNet can be swapped in on
Colab later if its weights are sourced reliably; that is a measured decision for
a future run, not this one.

**Stitching (src/reid/stitch.py): union-find over tracks, merge only when all
three gates hold.** A track that ends is merged with a later one that (a) starts
within `gap_frames` (90, ~3 s) of it ending, (b) re-appears within
`max_anchor_dist` (250 px) of where it left off, and (c) has mean-embedding
cosine >= `min_similarity` (0.8). Requiring temporal + spatial + appearance
agreement together is what keeps the weaker appearance signal from over-merging
two different people who happen to be at the kiosk at the same time. Canonical id
is the earliest track_id in each merged group, so merged ids stay interpretable.

**Scope is within-visit occlusion breaks, not multi-visit returns.** A person who
leaves and comes back minutes later exceeds `gap_frames` and correctly stays a
separate identity; the existing `segment_gap_frames` logic in aggregate still
re-segments a merged id if it contains a genuine long internal gap, so dwell
summing continues to mirror the GT protocol.

Thresholds (gap_frames 90, max_anchor_dist 250, min_similarity 0.8) are
principled starting values in cam1.yaml, not tuned to the dev clip — they are to
be validated on the Colab slice-eval and revised there if needed. `min_similarity`
in particular is expected to want tuning given the non-negative-feature note above.

**Slicing (run.py --slice).** Added a frame-range option so a step can be scored
on the dense/crowded diagnostic window (`eval_slice: [26700, 71000]`, covers
P1-P9) for fast iteration; the full video stays the default. perf.yaml now
reports processed-frame count and throughput net of the seek offset.

Local sanity check only (400-frame slice, MPS): pipeline runs end to end, emits
correctly-shaped tracks.yaml/perf.yaml, throughput ~48 FPS with embedding on top
of detect+track. Not a measurement — real numbers come from the Colab slice-eval.

## Tooling — reproducible baseline eval

`evaluate_baseline.py` scores `outputs/tracks.yaml` against
`eval/label/kiosk_gt.yaml`, collapsing each person_id's repeat visits with the
pipeline's own `collapse_segments` and calling the (untouched) eval harness. It
regenerates `outputs/eval_report.csv` exactly, so the baseline number is
reproducible from the repo rather than a scratch script.
