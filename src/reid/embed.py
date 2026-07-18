"""Appearance embedding for track re-association.

A track's bounding-box crop is mapped to an L2-normalized feature vector so two
fragments of the same person can be recognised by appearance. Backbone is a
torchvision classifier with its head removed (global-pooled features); the model
name is config-driven. torchvision hosts pretrained weights on
download.pytorch.org, which is why this is preferred over an OSNet/torchreid
build whose Google-Drive weight download is unreliable (see decisions.md).
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from torchvision import models

# Person crops are resized to a portrait box, not a square, so the standing-body
# aspect ratio is preserved before the ImageNet-normalized forward pass.
_INPUT_H, _INPUT_W = 256, 128
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class Embedder:
    def __init__(self, model: str, device: str):
        net = models.get_model(model, weights=models.get_model_weights(model).DEFAULT)
        # Drop the classification head; what remains outputs the global-pooled
        # feature vector we use as the appearance embedding.
        net.classifier = torch.nn.Identity()
        self.net = net.eval().to(device)
        self.device = device
        self.mean = torch.tensor(_IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(_IMAGENET_STD, device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def embed(self, frame: np.ndarray, boxes: list[tuple[float, float, float, float]]) -> np.ndarray:
        """Embed every box in one frame as a batch. Returns [N, D], L2-normalized."""
        height, width = frame.shape[:2]
        crops = np.empty((len(boxes), _INPUT_H, _INPUT_W, 3), dtype=np.float32)
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            # Clamp to frame bounds: tracker boxes can extend a pixel past the edge.
            xi1, yi1 = max(0, int(x1)), max(0, int(y1))
            xi2, yi2 = min(width, int(x2)), min(height, int(y2))
            crops[i] = cv2.resize(frame[yi1:yi2, xi1:xi2], (_INPUT_W, _INPUT_H))

        batch = torch.from_numpy(crops[..., ::-1].copy())  # BGR -> RGB
        batch = batch.permute(0, 3, 1, 2).to(self.device) / 255.0
        batch = (batch - self.mean) / self.std
        features = self.net(batch)
        return torch.nn.functional.normalize(features, dim=1).cpu().numpy()
