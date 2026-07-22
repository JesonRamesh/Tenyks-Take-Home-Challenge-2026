"""Builds the configured detector. Shared by run.py and dump_detections.py.

Imports are deferred to the selected branch on purpose: the detector backends have
mutually incompatible dependency pins (rfdetr needs transformers>=5.1, the RT-DETR
wrapper is pinned to 4.45.2), so importing one must not require the others to be
installed in the same environment.
"""

from __future__ import annotations

from src.detect.base import Detector


def build_detector(det_cfg: dict, device: str) -> Detector:
    detector_type = det_cfg.get("type", "yolo")
    if detector_type == "cached":
        from src.detect.cached import CachedDetector

        return CachedDetector(det_cfg["cache"])

    args = (det_cfg["model"], det_cfg["confidence"], det_cfg["classes"], det_cfg["imgsz"], device)
    if detector_type == "rtdetr":
        from src.detect.rtdetr import RTDetrDetector

        return RTDetrDetector(*args)
    if detector_type == "rfdetr":
        from src.detect.rfdetr import RFDetrDetector

        return RFDetrDetector(*args)
    if detector_type == "yolo":
        from src.detect.yolo import YoloDetector

        return YoloDetector(*args)
    raise ValueError(f"unknown detector.type: {detector_type}")
