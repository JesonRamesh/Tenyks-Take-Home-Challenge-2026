# CUDA throughput + VRAM benchmark (run on Colab / any real CUDA GPU)

Peak VRAM cannot be measured on this project's dev machine: `torch.cuda.max_memory_reserved()`
reads 0.0 on MPS, so every FPS/VRAM number produced locally is an MPS proxy only. This is the
procedure for getting the real numbers on the T4-class target named in the constraints.

Runs the **live** detector (never the detection cache — a cached run's FPS measures tracking
only) across all three validation windows, for both candidate pipelines. `run.py` calls
`reset_peak_memory_stats()` before the loop and `synchronize()` before reading the peaks, and
times the loop only, so artifact writing does not inflate FPS.

## Runtime

Colab → Runtime → Change runtime type → **T4 GPU**.

## 1. Clone and install

The two detector stacks have genuinely conflicting pins: `rfdetr` needs `transformers>=5.1`,
while `boxmot==12.0.2` pins `torchvision<0.18` / `numpy==1.26.4`. Installing boxmot normally
would downgrade Colab's CUDA torch and break the GPU. So boxmot goes in with `--no-deps` and
its real runtime imports are installed separately — this is the same resolution used locally
(see `requirements-rfdetr.txt`).

```bash
git clone -b detector-tracker-v2 https://github.com/JesonRamesh/Tenyks-Take-Home-Challenge-2026.git
cd Tenyks-Take-Home-Challenge-2026

# RF-DETR + its stack (leaves Colab's CUDA torch in place: rfdetr only needs torch>=2.2)
pip install -q rfdetr ultralytics pyyaml opencv-python lap

# boxmot without its pins, then the packages it actually imports at runtime
pip install -q --no-deps boxmot==12.0.2 filterpy gdown loguru ftfy
pip install -q scikit-learn pandas beautifulsoup4 wcwidth "setuptools<81"

# hard gate: if this does not print cuda=True, stop — the install broke the GPU torch
python -c "import torch, boxmot, rfdetr; print('torch', torch.__version__, 'cuda=', torch.cuda.is_available())"
```

Then put `digital_kiosk.mp4` in the repo root (upload, or mount Drive and copy/symlink it).

## 2. Benchmark

`emit_render_frames` is forced off for both configs so the two are measured identically and no
large per-frame artifact is written.

```bash
python - <<'PY'
import yaml, pathlib
for name in ("configs/cam1_v2.yaml", "configs/cam1_roifix.yaml"):
    cfg = yaml.safe_load(pathlib.Path(name).read_text())
    cfg.setdefault("overlay", {})["emit_render_frames"] = False
    out = name.replace(".yaml", "_perf.yaml")
    pathlib.Path(out).write_text(yaml.safe_dump(cfg, sort_keys=False))
    print("wrote", out)
PY

for spec in "sparse 85000 91000" "sliceb 26700 30000" "crowded 103000 109000"; do
  set -- $spec
  for cfg in cam1_v2 cam1_roifix; do
    echo "=== $cfg / $1 ($2-$3) ==="
    python -m src.run --video digital_kiosk.mp4 --config configs/${cfg}_perf.yaml \
      --out-dir outputs/perf_cuda/${cfg}_$1 --slice $2 $3
    cat outputs/perf_cuda/${cfg}_$1/perf.yaml
  done
done
```

## 3. Summary table to paste back

```bash
python - <<'PY'
import yaml, pathlib
print(f"{'config':14} {'window':9} {'frames':>7} {'FPS':>8} {'VRAM_resv_GB':>13} {'VRAM_alloc_GB':>14}")
for cfg in ("cam1_v2", "cam1_roifix"):
    for w in ("sparse", "sliceb", "crowded"):
        p = pathlib.Path(f"outputs/perf_cuda/{cfg}_{w}/perf.yaml")
        if not p.exists():
            print(f"{cfg:14} {w:9} {'MISSING':>7}"); continue
        d = yaml.safe_load(p.read_text())
        print(f"{cfg:14} {w:9} {d['frames']:>7} {d['fps']:>8.2f} "
              f"{d['peak_vram_reserved_gb']:>13.3f} {d['peak_vram_allocated_gb']:>14.3f}")
PY
```

## Notes on reading the result

- **`peak_vram_reserved_gb` is the number to compare against the 16 GB budget** (Phase 5
  methodology): it includes the caching allocator's freed-but-retained blocks and is the closer
  proxy to what `nvidia-smi` shows. `peak_vram_allocated_gb` is live tensor bytes only.
- The ~30 fps source rate is the real-time bar. Locally on MPS v2 measured 12.1 FPS against the
  baseline's 31.1, but MPS is a poor proxy for a DETR-family model — RF-DETR additionally ships
  an untested `optimize_for_inference(fp16)` path claiming ~8x on T4 tensor cores, which is not
  enabled here and would be the first lever if v2 lands under 30 FPS.
- Accuracy is not re-measured here: the pipeline is deterministic given the same detections, so
  the count/coverage numbers in `decisions.md` stand. Only throughput and VRAM are hardware-dependent.
