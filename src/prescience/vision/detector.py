"""Ultralytics YOLO detection wrapper."""

from __future__ import annotations

import numpy as np
from ultralytics import YOLO

from prescience.types import Detection


class YoloDetector:
    """Run YOLO object detection on a frame."""

    def __init__(self, model_path: str, conf: float = 0.35, classes: list[int] | None = None):
        self.conf = conf
        self.classes = classes
        self.model = YOLO(model_path)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Return typed detections for one frame."""
        results = self.model(frame, conf=self.conf, classes=self.classes, verbose=False)
        result = results[0]

        detections: list[Detection] = []
        if result.boxes is None or len(result.boxes) == 0:
            return detections

        xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()

        for (x1, y1, x2, y2), conf, class_id in zip(xyxy, confs, class_ids):
            class_idx = int(class_id)
            class_name = self.model.names.get(class_idx, f"class_{class_idx}")
            detections.append(
                Detection(
                    box=(int(x1), int(y1), int(x2), int(y2)),
                    confidence=float(conf),
                    class_id=class_idx,
                    class_name=str(class_name),
                )
            )
        return detections
