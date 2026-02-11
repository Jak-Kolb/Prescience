"""Ultralytics + ByteTrack wrapper producing typed tracked detections."""

from __future__ import annotations

import numpy as np
from ultralytics import YOLO

from prescience.types import TrackedDetection


class YoloTracker:
    """Run object tracking and return detections with track ids."""

    def __init__(
        self,
        model_path: str,
        conf: float = 0.35,
        classes: list[int] | None = None,
        tracker_cfg: str = "bytetrack.yaml",
    ):
        self.conf = conf
        self.classes = classes
        self.tracker_cfg = tracker_cfg
        self.model = YOLO(model_path)

    def track(self, frame: np.ndarray) -> list[TrackedDetection]:
        """Track objects for one frame."""
        results = self.model.track(
            frame,
            persist=True,
            conf=self.conf,
            classes=self.classes,
            verbose=False,
            tracker=self.tracker_cfg,
        )
        result = results[0]

        out: list[TrackedDetection] = []
        if result.boxes is None or len(result.boxes) == 0:
            return out

        xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()

        ids = None
        if result.boxes.id is not None:
            ids = result.boxes.id.cpu().numpy().astype(int)

        for i, ((x1, y1, x2, y2), conf, class_id) in enumerate(zip(xyxy, confs, class_ids)):
            class_idx = int(class_id)
            class_name = self.model.names.get(class_idx, f"class_{class_idx}")
            out.append(
                TrackedDetection(
                    box=(int(x1), int(y1), int(x2), int(y2)),
                    confidence=float(conf),
                    class_id=class_idx,
                    class_name=str(class_name),
                    track_id=int(ids[i]) if ids is not None else None,
                )
            )
        return out
