"""RT-DETR person detector implementing the Detector Protocol (HuggingFace transformers).

RT-DETR is set-based and NMS-free, so it separates adjacent people that YOLOv8n + NMS
merge into one box (the density-driven merge measured in decisions.md). The small R18
backbone (`PekingU/rtdetr_r18vd`) is the production choice. It emits xyxy pixel boxes,
the same format YoloDetector produces, so every downstream stage (zone gate, staff-crop
classification, ReID embedding) consumes it unchanged.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch

from src.detect.base import Detection

# transformers' RT-DETR forward calls torch.compiler.is_compiling(), which does not exist
# before torch 2.3; our stack is pinned to torch 2.2.x (boxmot's torchvision), so add the
# (always-False, i.e. "not compiling") shim once at import.
if not hasattr(torch.compiler, "is_compiling"):
    torch.compiler.is_compiling = lambda *args, **kwargs: False

from transformers import RTDetrForObjectDetection, RTDetrImageProcessor  # noqa: E402


class RTDetrDetector:
    def __init__(self, model: str, confidence: float, classes: list[int], imgsz: int, device: str):
        self.processor = RTDetrImageProcessor.from_pretrained(model)
        self.model = RTDetrForObjectDetection.from_pretrained(model).to(device).eval()
        self.confidence = confidence
        self.classes = set(classes)
        self.size = {"height": imgsz, "width": imgsz}
        self.device = device

    @torch.no_grad()
    def detect(self, frame: np.ndarray) -> list[Detection]:
        inputs = self.processor(
            images=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), size=self.size, return_tensors="pt"
        ).to(self.device)
        outputs = self.model(**inputs)
        height, width = frame.shape[:2]
        # post_process rescales boxes to (height, width) xyxy pixels and drops sub-threshold
        # detections, matching what conf does for YoloDetector.
        result = self.processor.post_process_object_detection(
            outputs, target_sizes=[(height, width)], threshold=self.confidence
        )[0]
        return [
            Detection(*box.tolist(), float(score))
            for box, score, label in zip(result["boxes"], result["scores"], result["labels"])
            if int(label) in self.classes
        ]
