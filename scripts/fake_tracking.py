import cv2

from collections import deque # list like structure that can keep only the last N points (for trail)

def main() -> None:
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")
    
    line_y = 300  # y-coordinate of the line to be drawn
    
    track_id = 1
    fake_y = 0
    fake_x = None
    
    trail = deque(maxlen=30)  # store the last 20 positions for the trail
        


    while True:
        ok, frame = cap.read()
        
        if not ok:
            break
        
        h, w = frame.shape[:2]
        
        
        if fake_x is None:
            fake_x = w // 2
            
        fake_y = (fake_y + 5) % h  # move down and wrap around
        
        trail.append((fake_x, fake_y))  # add current position to the trail
        
        
        # lind( image, start_point(x, y), end_point(x, y), color(B, G, R), thickness)
        cv2.line(frame, (0, line_y), (w, line_y), (0, 255, 255), 2)
        
        for i in range(1, len(trail)):
            cv2.line(frame, trail[i-1], trail[i], (0, 255, 0), 2)  # draw the trail
            
        cv2.circle(frame, (fake_x, fake_y), 10, (0, 0, 255), -1)  # draw the fake object
        
        cv2.putText(frame, f"ID: {track_id}", (fake_x - 20, fake_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow("Fake Tracking", frame)
        
        
        key = cv2.waitKey(1)
        
        if (key & 0xFF) == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()
        
if __name__ == "__main__":
    main()