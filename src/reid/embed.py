"""Appearance embedding for track re-association.

A track's bounding-box crop is mapped to an L2-normalized feature vector so two
fragments of the same person can be recognised by appearance. Two backbones are
supported, chosen by reid.model:

- a boxmot ReID backbone named by its weight file (e.g. weights/osnet_x0_25_msmt17.pt):
  a real person-ReID model. Preferred -- the diagnostic showed ImageNet features barely
  separate identities on this camera (same- and different-person cosine distributions
  overlap), which is the dominant merge blocker.
- a torchvision classifier (e.g. mobilenet_v3_small) with its head removed, used as a
  weaker feature extractor. Kept for comparison against the original pipeline.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import models

# Person crops are resized to a portrait box, not a square, so the standing-body
# aspect ratio is preserved before the ImageNet-normalized forward pass.
_INPUT_H, _INPUT_W = 256, 128
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _boxmot_device(device: str) -> str:
    # boxmot's select_device wants a device index ("0"), not torch's "cuda" string.
    if device == "cuda":
        return "0"
    if device.startswith("cuda:"):
        return device.split(":", 1)[1]
    return device


class Embedder:
    def __init__(self, model: str, device: str):
        self.device = device
        if model.endswith(".pt"):
            # boxmot auto-downloads the ReID weights but does not create the directory.
            from boxmot.appearance.reid_auto_backend import ReidAutoBackend

            weights = Path(model)
            weights.parent.mkdir(parents=True, exist_ok=True)
            self._backend = ReidAutoBackend(
                weights=weights, device=_boxmot_device(device), half=False
            ).get_backend()
            self.net = None
        else:
            net = models.get_model(model, weights=models.get_model_weights(model).DEFAULT)
            # Drop the classification head; what remains outputs the global-pooled
            # feature vector we use as the appearance embedding.
            net.classifier = torch.nn.Identity()
            self.net = net.eval().to(device)
            self._backend = None
            self.mean = torch.tensor(_IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
            self.std = torch.tensor(_IMAGENET_STD, device=device).view(1, 3, 1, 1)

    def embed(self, frame: np.ndarray, boxes: list[tuple[float, float, float, float]]) -> np.ndarray:
        """Embed every box in one frame as a batch. Returns [N, D], L2-normalized."""
        if self._backend is not None:
            # boxmot crops internally and its resize raises on a zero-area crop, so clamp
            # each box to a >=1px extent within the frame first (clamp, not drop, to keep
            # one embedding per input box). Same guard as the torchvision path below.
            height, width = frame.shape[:2]
            clamped = np.empty((len(boxes), 4), dtype=np.float32)
            for i, (x1, y1, x2, y2) in enumerate(boxes):
                cx1 = min(max(0.0, x1), width - 1.0)
                cy1 = min(max(0.0, y1), height - 1.0)
                clamped[i] = (cx1, cy1, max(min(float(width), x2), cx1 + 1.0), max(min(float(height), y2), cy1 + 1.0))
            features = np.asarray(self._backend.get_features(clamped, frame))
            norms = np.linalg.norm(features, axis=1, keepdims=True)
            return features / np.where(norms == 0, 1.0, norms)
        return self._embed_torchvision(frame, boxes)

    @torch.no_grad()
    def _embed_torchvision(
        self, frame: np.ndarray, boxes: list[tuple[float, float, float, float]]
    ) -> np.ndarray:
        height, width = frame.shape[:2]
        crops = np.empty((len(boxes), _INPUT_H, _INPUT_W, 3), dtype=np.float32)
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            # Clamp to frame bounds (tracker boxes can extend a pixel past the edge) and
            # to a >=1px extent: a sub-pixel-wide box would otherwise clamp to a zero-area
            # crop that cv2.resize rejects, aborting the run on one spurious thin box.
            xi1 = min(max(0, int(x1)), width - 1)
            yi1 = min(max(0, int(y1)), height - 1)
            xi2 = max(min(width, int(x2)), xi1 + 1)
            yi2 = max(min(height, int(y2)), yi1 + 1)
            crops[i] = cv2.resize(frame[yi1:yi2, xi1:xi2], (_INPUT_W, _INPUT_H))

        batch = torch.from_numpy(crops[..., ::-1].copy())  # BGR -> RGB
        batch = batch.permute(0, 3, 1, 2).to(self.device) / 255.0
        batch = (batch - self.mean) / self.std
        features = self.net(batch)
        return torch.nn.functional.normalize(features, dim=1).cpu().numpy()
