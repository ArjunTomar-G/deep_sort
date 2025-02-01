import cv2
import torch
import numpy as np
from ultralytics import YOLO
from deep_sort.tracker import Tracker
from deep_sort.nn_matching import NearestNeighborDistanceMetric
from deep_sort.detection import Detection
from collections import defaultdict, deque
import os
from onnxruntime import InferenceSession
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import torchvision.transforms as transforms

# Initialize YOLO and DeepSORT
video_path = "/Users/arjuntomar/Desktop/GPT/Challenge1_course3_persons.mp4q"  # Update with your video file path
yolo_model = YOLO("/Users/arjuntomar/Desktop/GPT/best_body.pt")
yolo_conf_threshold = 0.7  # Adjustable confidence threshold

metric = NearestNeighborDistanceMetric("cosine", matching_threshold=0.25)  # Increased matching threshold
tracker = Tracker(metric=metric, max_age=60)  # Extended max age for better tracking

# Load ONNX Re-ID model
onnx_model_path = "/Users/arjuntomar/Desktop/GPT/nvidia_reid_model_modified.onnx"
session = InferenceSession(onnx_model_path)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Define preprocessing for Re-ID model
preprocess = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create directory for storing first frames of new IDs
output_dir = "tracked_frames"
os.makedirs(output_dir, exist_ok=True)

# ID tracking and embeddings storage
global_id_counter = 1  # Start tracking from ID 1
track_id_map = {}  # Mapping from DeepSORT IDs to sequential IDs
saved_ids = set()  # Track IDs for which the first frame has been saved
embeddings_store = {}  # Store only one embedding per ID

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    exit(1)

frame_count = 0
frame_skip = 1  # Process every 3rd frame

def extract_embedding(cropped_frame):
    """Extracts feature embedding from the cropped person image."""
    image = Image.fromarray(cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB))
    input_tensor = preprocess(image).unsqueeze(0).numpy()
    output = session.run([output_name], {input_name: input_tensor})
    embedding = output[0][0]
    return embedding / np.linalg.norm(embedding)

def match_embedding(new_embedding):
    """Matches the new embedding with stored embeddings and returns best ID."""
    global global_id_counter
    if not embeddings_store:
        return None  # No previous embeddings to compare
    
    best_match_id = None
    best_similarity = 0
    for track_id, stored_embedding in embeddings_store.items():
        similarity = cosine_similarity([new_embedding], [stored_embedding])[0, 0]
        if similarity > 0.8 and similarity > best_similarity:
            best_similarity = similarity
            best_match_id = track_id
    return best_match_id

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    if frame_count % frame_skip != 0:
        continue  # Skip some frames
    
    detections = yolo_model(frame)[0]
    largest_person = None
    max_area = 0
    
    for det in detections.boxes.data:
        x1, y1, x2, y2, conf, cls = det.cpu().numpy()
        if cls == 0 and conf > yolo_conf_threshold:
            area = (x2 - x1) * (y2 - y1)
            if area > max_area:
                max_area = area
                largest_person = [x1, y1, x2, y2]
    
    if not largest_person:
        continue
    
    x1, y1, x2, y2 = map(int, largest_person)
    cropped_person = frame[y1:y2, x1:x2]
    person_embedding = extract_embedding(cropped_person)
    
    if len(embeddings_store) == 0 or (match_id := match_embedding(person_embedding)) is None:
        seq_id = global_id_counter
        global_id_counter += 1
        embeddings_store[seq_id] = person_embedding  # Store new embedding only when assigning a new ID
    else:
        seq_id = match_id
    
    # Save first frame when a new ID is assigned
    if seq_id not in saved_ids:
        frame_path = os.path.join(output_dir, f"ID_{seq_id}.jpg")
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {seq_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imwrite(frame_path, frame)
        saved_ids.add(seq_id)
    
    # Draw bounding box and ID
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, f"ID: {seq_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
