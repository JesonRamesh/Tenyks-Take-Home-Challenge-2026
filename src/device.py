"""Compute-device resolution, shared by the entrypoint and the diagnostic scripts.

Kept out of run.py so a script that only needs a detector (dump_detections,
compare_detectors) doesn't have to import the tracker stack with it — the detector
backends deliberately live in environments where boxmot isn't installed.
"""

from __future__ import annotations

import torch


def resolve_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
