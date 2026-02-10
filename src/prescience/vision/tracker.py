
import numpy as np
from ultralytics import YOLO
from prescience.types import TrackedDetection

class YoloTracker:
    """
    accepts list of Detection objects and returns list of TrackedDetection objects with track ids
    """
    
    def __init__(self, model_path: str, conf: float = 0.35, classes: list[int] | None = None, tracker_cfg: str = "bytetrack.yaml"):
        self.conf = conf
        self.classes = classes
        self.model = YOLO(model_path)
        self.tracker_cfg = tracker_cfg # initialize the tracker with the given config file

    def track(self, frame: np.ndarray) -> list[TrackedDetection]:
        """run YOLO on a single frame and return tracked detections as objects""" 
        results = self.model.track(
            frame, 
            persist=True, # keep the tracker state across frames
            conf=self.conf,
            classes=self.classes,
            verbose=False,
            tracker=self.tracker_cfg)
        
        r = results[0]
        
        out: list[TrackedDetection] = []
        
        if r.boxes is None or len(r.boxes) == 0:
            return out
        
        xyxy = r.boxes.xyxy.cpu().numpy()  # (N, 4) array of (x1, y1, x2, y2)
        confs = r.boxes.conf.cpu().numpy()   # (N,) array of confidence scores
        class_ids = r.boxes.cls.cpu().numpy() # (N,) array
        
        # extract track ids if available, otherwise set to None
        ids = None
        if r.boxes.id is not None:
            ids = r.boxes.id.cpu().numpy().astype(int)     # (N,) array of track ids
            
        for i, ((x1, y1, x2, y2), conf, class_id) in enumerate(zip(xyxy, confs, class_ids)):
            x1_i = int(x1)
            y1_i = int(y1)
            x2_i = int(x2)
            y2_i = int(y2)
            
            class_name = self.model.names.get(class_id, f"class_{class_id}") # get class name from model, or default to "class_{id}"
            
            track_id = ids[i] if ids is not None else None
            
            out.append(TrackedDetection(
                box=(x1_i, y1_i, x2_i, y2_i),
                confidence=float(conf),
                class_id=int(class_id),
                class_name=class_name,
                track_id=track_id
            ))
            
        return out