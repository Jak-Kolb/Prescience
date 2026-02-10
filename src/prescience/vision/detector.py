
import numpy as np
from ultralytics import YOLO
from prescience.types import Detection

class YoloDetector:
    """
    accepts openCV frame and returns list of Detection objects
    """
    
    def __init__(self, model_path: str, conf: float = 0.35, classes: list[int] | None = None):
        self.conf = conf
        self.classes = classes
        self.model = YOLO(model_path)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """run YOLO on a single frame and return detections as objects""" 
        results = self.model(frame, conf=self.conf, classes=self.classes, verbose=False)
        r = results[0]
        
        detections: list[Detection] = []
        
        if r.boxes is None or len(r.boxes) == 0:
            return detections
        
        xyxy = r.boxes.xyxy.cpu().numpy()  # (N, 4) array of (x1, y1, x2, y2)
        confs = r.boxes.conf.cpu().numpy()   # (N,) array of
        class_ids = r.boxes.cls.cpu().numpy() # (N,) array of class ids
        
        for (x1, y1, x2, y2), conf, class_id in zip(xyxy, confs, class_ids):
            x1_i = int(x1)
            y1_i = int(y1)
            x2_i = int(x2)
            y2_i = int(y2)
            
            class_name = self.model.names.get(class_id, f"class_{class_id}")
            
            detections.append(Detection(
                box=(x1_i, y1_i, x2_i, y2_i),
                confidence=float(conf),
                class_id=int(class_id),
                class_name=class_name
            ))
        
        return detections