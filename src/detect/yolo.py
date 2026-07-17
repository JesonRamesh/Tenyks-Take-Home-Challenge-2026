"""YOLOv8 person detector implementing the Detector Protocol (ultralytics)."""

from __future__ import annotations

import numpy as np
from ultralytics import YOLO

from src.detect.base import Detection


class YoloDetector:
    def __init__(
        self,
        model: str,
        confidence: float,
        classes: list[int],
        imgsz: int,
        device: str,
    ):
        self.model = YOLO(f"{model}.pt")
        self.confidence = confidence
        self.classes = classes
        self.imgsz = imgsz
        self.device = device

    def detect(self, frame: np.ndarray) -> list[Detection]:
        result = self.model.predict(
            frame,
            conf=self.confidence,
            classes=self.classes,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )[0]
        xyxy = result.boxes.xyxy.cpu().numpy()
        conf = result.boxes.conf.cpu().numpy()
        return [Detection(x1, y1, x2, y2, c) for (x1, y1, x2, y2), c in zip(xyxy, conf)]
