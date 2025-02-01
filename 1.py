import cv2
import torch
import numpy as np
from ultralytics import YOLO
import sys
import os
from onnxruntime import InferenceSession



# from openpose import pyopenpose as op  # If using OpenPose
from torchvision import models, transforms
from sklearn.cluster import DBSCAN


sys.path.append(os.path.abspath("/Users/arjuntomar/Desktop/GPT/deep_sort"))
from deep_sort.tracker import Tracker





# Load YOLOv8 for person detection
yolo_model = YOLO("/Users/arjuntomar/Desktop/GPT/best_body.pt")

# Initialize DeepSORT for tracking
tracker = Tracker(max_age=30)

# Load Re-ID Model (OSNet)
session = InferenceSession('/Users/arjuntomar/Desktop/GPT/nvidia_reid_model_modified.onnx')

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name


# Load DeepLabV3 for clothing segmentation
segmentation_model = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()

# Load video
video_path = "input_video.mp4"
cap = cv2.VideoCapture(video_path)

# Feature storage
feature_store = {}
frame_count = 0

preprocess = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    detections = yolo_model(frame)[0]  # Detect people

    persons = []
    for det in detections.boxes.data:
        x1, y1, x2, y2, conf, cls = det.cpu().numpy()
        if cls == 0 and conf > 0.4:  # Class 0 = person
            persons.append([x1, y1, x2, y2])

    # Tracking people across frames
    tracks = tracker.update_tracks(persons, frame=frame)

    for track in tracks:
        track_id = track.track_id
        x1, y1, x2, y2 = track.to_tlbr()

        person_crop = frame[int(y1):int(y2), int(x1):int(x2)]
        
        # Extract Pose (if using OpenPose)
        # pose_features = extract_pose(person_crop)

        # Extract Clothing Features
        clothing_features = extract_clothing(segmentation_model, person_crop)

        
            
        input_tensor = preprocess(person_crop).unsqueeze(0).numpy()
                
                
        output = session.run([output_name], {input_name: input_tensor})
        embedding = output[0][0]
        norm = np.linalg.norm(embedding)
        embedding = embedding / norm
                
        print(f"Extracted embeddings for {len(embedding)} images.")
        reid_features = embedding

        # Combine Features (Pose + Clothing + Re-ID)
        final_features = np.concatenate((clothing_features, reid_features), axis=0)

        # Store for Clustering
        feature_store[track_id] = final_features

        # Display ID on frame
        cv2.putText(frame, f'ID: {track_id}', (int(x1), int(y1)-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show frame
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# **Step 5: Cluster embeddings using DBSCAN**
features_array = np.array(list(feature_store.values()))
clustering = DBSCAN(eps=0.5, min_samples=5).fit(features_array)

# Assign final unique IDs
for i, track_id in enumerate(feature_store.keys()):
    final_id = clustering.labels_[i]
    print(f"Person {track_id} → Assigned ID: {final_id}")
