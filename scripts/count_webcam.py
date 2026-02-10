
import argparse # 1. Import argparse for command-line argument parsing

from prescience.pipeline.count_stream import run

def main() -> None:
    
    p = argparse.ArgumentParser()
    p.add_argument("--source", default=0, help="Video source (0 for webcam, or path to video file)")
    p.add_argument("--model", default="yolov8n.pt", help="Path to YOLO model")
    p.add_argument("--conf", type=float, default=0.35, help="Confidence threshold for detections")
    
    args = p.parse_args()
    
    # Convert source to int if it's a digit, otherwise keep as string (for file paths)
    source = int(args.source) if args.source.isdigit() else args.source 
    
    run(source=source, model_path=args.model, conf=args.conf)
    
    
if __name__ == "__main__":
    main()