# Prescience

Object detection, tracking, and counting system using YOLO and computer vision.

## Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Scripts

### Webcam Utilities

- **show_webcam.py** - Simple webcam viewer for testing camera functionality
- **draw_line.py** - Interactive tool to draw counting lines on webcam feed

### Tracking & Counting

- **count_webcam.py** - Real-time object detection and counting from webcam
- **count_video.py** - Process video files for object detection and counting
- **fake_tracking.py** - Simulated object tracking for testing
- **fake_tracking_with_count.py** - Simulated tracking with line-crossing counter demo

### Enrollment

- **enroll.py** - Capture and save profiles for object recognition

## Core Modules

### `prescience/ingest/`

- **video_to_frames.py** - Video frame extraction utilities
- **frame_filter.py** - Frame preprocessing and filtering

### `prescience/pipeline/`

- **count_stream.py** - Real-time counting pipeline
- **count_video.py** - Video file processing pipeline
- **zone_count.py** - Zone-based counting logic
- **enroll.py** - Profile enrollment pipeline

### `prescience/vision/`

- **detector.py** - YOLO object detection wrapper
- **embeddings.py** - Feature extraction for object recognition
- **matcher.py** - Object matching and identification
- **tracker.py** - Multi-object tracking

### `prescience/profiles/`

- **io.py** - Profile storage and retrieval
- **schema.py** - Profile data structures

## Configuration

Config files in `configs/` directory control detection parameters, model paths, and counting zones.

## Usage

```bash
# View webcam
python scripts/show_webcam.py

# Count objects from webcam
python scripts/count_webcam.py --source 0 --model yolov8n.pt --conf 0.35

# Count objects in video
python scripts/count_video.py --source path/to/video.mp4

# Test tracking simulation
python scripts/fake_tracking_with_count.py
```
