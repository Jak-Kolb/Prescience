import cv2

def main() -> None:
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    while True:
        ok, frame = cap.read()
        
        if not ok:
            break
        
        cv2.imshow("Webcam", frame)
        
        key = cv2.waitKey(1)
        
        if (key & 0xFF) == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()
        
if __name__ == "__main__":
    main()