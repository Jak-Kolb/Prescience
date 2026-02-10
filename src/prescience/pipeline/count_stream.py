import cv2

from prescience.vission.detector import YoloDetector # runs YOLO and returns list of Detection objects

def run(source:int | str = 0, model_path: str = "yolov8n.pt", conf: float = 0.35) -> None:
    """Stream frames from a source and draw detections"""
    
    cap = cv2.VideoCapture(source) # open video capture from source (0 for webcam, or path to video file)
    
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source {source}")
    
    detector = YoloDetector(model_path=model_path, conf=conf, classes=None)
    
    while True:
        ok, frame = cap.read() # read a frame from the video source
        if not ok:
            break
        detections = detector.detect(frame) # run YOLO on the frame to get detections
        
        for det in detections:
            x1, y1, x2, y2 = det.box # unpack bounding box
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # draw bounding box
            label = f"{det.class_name} {det.confidence:.2f}" # create label with class name and confidence
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2) # draw label above box
            
        cv2.imshow("Detections", frame) # show the frame with detections
        
        key = cv2.waitKey(1)
        if (key & 0xFF) == ord('q'):   
            break
        
    cap.release()
    cv2.destroyAllWindows() # close the openCV windows
            
            
    
    
    