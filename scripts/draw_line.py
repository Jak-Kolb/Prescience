import cv2

def main() -> None:
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")
    
    line_y = 300  # y-coordinate of the line to be drawn
    
    while True:
        ok, frame = cap.read()
        
        if not ok:
            break
        
        h, w = frame.shape[:2]
        
        # lind( image, start_point(x, y), end_point(x, y), color(B, G, R), thickness)
        cv2.line(frame, (0, line_y), (w, line_y), (0, 255, 255), 2)
        cv2.imshow("Webcam + Line", frame)
        
        
        key = cv2.waitKey(1)
        
        if (key & 0xFF) == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()
        
if __name__ == "__main__":
    main()