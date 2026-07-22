"""RF-DETR person detector implementing the Detector Protocol (Roboflow rfdetr).

RF-DETR is a DETR-family, NMS-free detector Roboflow reports as strong on occlusion and
domain shift — the two failure modes that dominate this camera — so it is the alternative
tested against RT-DETR-R18. Like RT-DETR it emits xyxy pixel boxes, so every downstream
stage consumes it unchanged.

rfdetr pulls transformers>=5.1, which conflicts with the transformers==4.45.2 the RT-DETR
wrapper is pinned to, so this runs in its own venv and reaches the pipeline through a
detection cache rather than sharing a process with the RT-DETR path (see decisions.md).
"""

from __future__ import annotations

import cv2
import numpy as np

from src.detect.base import Detection

# COCO "person" in rfdetr's label space is 1, not 0: its pretrained checkpoints keep the
# original 91-class COCO ids (1-indexed, background at 0) rather than the 80-class
# contiguous remap ultralytics uses. The config's class ids are YOLO-space, so translate.
_YOLO_TO_RFDETR_CLASS = {0: 1}


class RFDetrDetector:
    def __init__(self, model: str, confidence: float, classes: list[int], imgsz: int, device: str):
        from rfdetr import RFDETRBase, RFDETRLarge, RFDETRNano, RFDETRSmall

        variants = {
            "rfdetr-nano": RFDETRNano,
            "rfdetr-small": RFDETRSmall,
            "rfdetr-base": RFDETRBase,
            "rfdetr-large": RFDETRLarge,
        }
        self.model = variants[model](resolution=imgsz, device=device)
        self.confidence = confidence
        self.classes = {_YOLO_TO_RFDETR_CLASS.get(c, c) for c in classes}

    def detect(self, frame: np.ndarray) -> list[Detection]:
        # rfdetr's predict takes RGB and already applies the threshold, returning a
        # supervision Detections with xyxy in source-image pixels.
        result = self.model.predict(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), threshold=self.confidence
        )
        return [
            Detection(float(box[0]), float(box[1]), float(box[2]), float(box[3]), float(score))
            for box, score, class_id in zip(result.xyxy, result.confidence, result.class_id)
            if int(class_id) in self.classes
        ]
