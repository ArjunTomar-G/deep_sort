import cv2
import torch
import numpy as np
from ultralytics import YOLO
from deep_sort.tracker import Tracker
from deep_sort.nn_matching import NearestNeighborDistanceMetric
from deep_sort.detection import Detection
from collections import defaultdict
import os

# Initialize YOLO and DeepSORT
video_path = "/Users/arjuntomar/Desktop/GPT/GH010023_1.mp4"  # Update with your video file path
yolo_model = YOLO("best_body.pt")
metric = NearestNeighborDistanceMetric("cosine", matching_threshold=0.2)
tracker = Tracker(metric=metric, max_age=30)

global_id_counter = 1  # Start tracking from ID 1
track_id_map = {}  # Mapping from DeepSORT IDs to sequential IDs
active_ids = set()  # Track currently active IDs

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    exit(1)

frame_count = 0
frame_skip = 1  # Process every 3rd frame

def assign_sequential_id(track_id):
    global global_id_counter
    if track_id not in track_id_map:
        track_id_map[track_id] = global_id_counter
        global_id_counter += 1
    return track_id_map[track_id]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    if frame_count % frame_skip != 0:
        continue  # Skip some frames
    
    detections = yolo_model(frame)[0]
    persons = []
    for det in detections.boxes.data:
        x1, y1, x2, y2, conf, cls = det.cpu().numpy()
        if cls == 0 and conf > 0.4:
            persons.append([x1, y1, x2, y2])
    
    if not persons:
        continue
    
    detection_list = []
    for person in persons:
        bbox = person[:4]  # x1, y1, x2, y2
        conf = 1.0  # Dummy confidence
        feature = np.zeros((1, 128))  # Dummy feature vector
        detection_list.append(Detection(bbox, conf, feature))
    
    tracker.predict()
    tracker.update(detection_list)
    
    active_ids.clear()
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        
        track_id = track.track_id
        seq_id = assign_sequential_id(track_id)
        active_ids.add(seq_id)
        x1, y1, x2, y2 = map(int, track.to_tlbr())
        
        # Draw bounding box and ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {seq_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
