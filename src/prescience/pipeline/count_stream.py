import cv2

# from prescience.vision.detector import YoloDetector # runs YOLO and returns list of Detection objects
from prescience.vision.tracker import YoloTracker # runs YOLO and returns list of TrackedDetection objects with track ids
from prescience.pipeline.zone_count import UniqueTrackCounter

BOTTLE_CLASS_ID = 39 # COCO class ID for "bottle"


def run(source:int | str = 0, model_path: str = "yolov8n.pt", conf: float = 0.35, min_hits: int = 5) -> None:
    """
    - Stream frames from frome source
    - gives stable IDs per bottle while on screen
    - counter increments when a new bottle is detected and tracked for at least `min_hits` frames
    """
    
    cap = cv2.VideoCapture(source) # open video capture from source (0 for webcam, or path to video file)
    
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source {source}")
    
    tracker = YoloTracker(
        model_path=model_path, 
        conf=conf, 
        classes=[BOTTLE_CLASS_ID],
        tracker_cfg="bytetrack.yaml"
    )
    
    counter = UniqueTrackCounter(min_hits=min_hits)
    
    
    while True:
        ok, frame = cap.read() # read a frame from the video source
        if not ok:
            break
        # detections = detector.detect(frame) # run YOLO on the frame to get detections
        tracked = tracker.track(frame)
        
        active_ids: list[int] = []
        for t in tracked:
            if t.track_id is not None:
                active_ids.append(t.track_id)
                
        new_counts = counter.update(active_ids) # update the counter with the active track IDs for this frame
        
        if new_counts > 0:
            print(f"New counts added in this frame: {new_counts}. Total count: {counter.total}")
        
        for t in tracked:
            x1, y1, x2, y2 = t.box # unpack bounding box
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2) # draw bounding box
            
            tid_txt = f"ID:{t.track_id}" if t.track_id is not None else "ID ?"
            label = f"{t.class_name} {t.confidence:.2f} {tid_txt}" # create label with class name, confidence and track ID
            
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2) # draw label above box
        
        
        # for det in detections:
        #     x1, y1, x2, y2 = det.box # unpack bounding box
            
        #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # draw bounding box
        #     label = f"{det.class_name} {det.confidence:.2f}" # create label with class name and confidence
        #     cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2) # draw label above box
            
        cv2.imshow("Tracking with ID", frame) # show the frame with detections
        
        key = cv2.waitKey(1)
        if (key & 0xFF) == ord('q'):   
            break
        
    cap.release()
    cv2.destroyAllWindows() # close the openCV windows
            
            
    
    
    