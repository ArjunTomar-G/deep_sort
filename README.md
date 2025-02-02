
# Person ID Tracking System

## Overview
This system implements a robust person tracking and identification solution using YOLO object detection, deep feature embeddings, and similarity matching. The system is designed to track and maintain consistent IDs for people across video frames, even when they leave and re-enter the frame.

## Features
- Real-time person detection using YOLO
- Persistent ID assignment using deep feature embeddings
- Automatic frame capture of first appearances
- Cosine similarity-based re-identification
- Support for high-resolution video processing
- Frame skipping capability for performance optimization

## Requirements
- Python 3.7+
- PyTorch
- OpenCV (cv2)
- Ultralytics YOLO
- ONNX Runtime
- scikit-learn
- PIL (Python Imaging Library)
- NumPy
- DeepSORT tracking framework

## Model Requirements
- YOLO model trained for person detection (`best_body.pt`)
- ONNX Re-ID model for feature extraction (`nvidia_reid_model_modified.onnx`)

## Installation
1. Install the required Python packages:
```bash
pip install torch opencv-python ultralytics onnxruntime scikit-learn pillow numpy
```

2. Place the required model files in your project directory:
   - YOLO model (`best_body.pt`)
   - Re-ID ONNX model (`nvidia_reid_model_modified.onnx`)

## Configuration
Key parameters that can be adjusted:
- `yolo_conf_threshold`: Confidence threshold for YOLO detections (default: 0.7)
- `matching_threshold`: Threshold for the cosine similarity matching (default: 0.25)
- `frame_skip`: Number of frames to skip during processing (default: 1)
- `max_age`: Maximum number of frames to keep track of disappeared objects (default: 60)

## Usage
1. Update the `video_path` variable with your input video file path
2. Run the script:
```bash
python pipeline.py
```

## How It Works
1. **Person Detection**: 
   - YOLO model detects people in each frame
   - Only the largest detected person is tracked to avoid confusion

2. **Feature Extraction**:
   - Crops detected person from frame
   - Processes through Re-ID model to extract feature embeddings
   - Normalizes embeddings for consistent comparison

3. **ID Assignment**:
   - New persons get new sequential IDs
   - Returning persons are matched using cosine similarity
   - High similarity (>0.8) indicates same person

4. **Output**:
   - Displays real-time tracking with bounding boxes and IDs
   - Saves first appearance of each unique ID to `tracked_frames` directory
   - Shows tracking visualization in real-time

## Output Directory Structure
```
tracked_frames/
├── ID_1.jpg
├── ID_2.jpg
├── ID_3.jpg
...
```

## Limitations
- Currently tracks only the largest detected person in frame
- May struggle with very crowded scenes
- Requires good lighting conditions for optimal performance
- Processing speed depends on hardware capabilities

## Performance Optimization
- Adjust `frame_skip` for faster processing
- Modify `yolo_conf_threshold` for detection accuracy
- Tune `matching_threshold` for ID consistency

## Troubleshooting
1. If the video doesn't open:
   - Check if the video path is correct
   - Ensure video file format is supported by OpenCV

2. If tracking is inconsistent:
   - Adjust the matching threshold
   - Increase YOLO confidence threshold
   - Check lighting conditions

## Future Improvements
- Multi-person tracking support
- GPU acceleration
- Real-time performance optimization
- Support for multiple camera views
- Integration with database for long-term storage
