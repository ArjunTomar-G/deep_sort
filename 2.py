import cv2
import torch
import numpy as np
from ultralytics import YOLO
from torchvision import models, transforms
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from onnxruntime import InferenceSession
from skimage.feature import local_binary_pattern
from deep_sort.detection import Detection
import sys
import os
from PIL import Image

# Add DeepSORT to path
sys.path.append(os.path.abspath("/Users/arjuntomar/Desktop/GPT/deep_sort"))
from deep_sort.tracker import Tracker
from deep_sort.nn_matching import NearestNeighborDistanceMetric

def extract_clothing_batch(segmentation_model, person_crops):
    """Batch process clothing segmentation with error handling."""
    batch_tensors = []
    original_sizes = []

    for crop in person_crops:
        try:
            # Convert tensor to numpy with correct dimensions
            if torch.is_tensor(crop):
                crop = crop.permute(1, 2, 0).cpu().numpy()
            
            # Ensure uint8 dtype
            crop = np.ascontiguousarray(crop, dtype=np.uint8)
            
            # Debug info
            print(f"After conversion - shape: {crop.shape}, dtype: {crop.dtype}")
            
            # Convert BGR to RGB
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            
            # Convert to tensor for model
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            tensor = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )(tensor)
            
            batch_tensors.append(tensor)
            original_sizes.append(crop.shape[:2])
            
        except Exception as e:
            print(f"Error processing crop: {e}")
            continue

    if not batch_tensors:
        return []
    
    # Process batch
    with torch.no_grad():
        output = segmentation_model(torch.stack(batch_tensors))['out']
    
    masks = []
    for idx, out in enumerate(output):
        mask = out.argmax(0).byte().cpu().numpy()
        mask = (mask == 15).astype(np.uint8)
        h, w = original_sizes[idx]
        masks.append(cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST))
    
    return masks
def extract_reid_features(segmented_clothing, reid_preprocess, reid_session):
    try:
        # Debug print
        print(f"Input shape: {segmented_clothing.shape}, dtype: {segmented_clothing.dtype}")
        
        # Ensure correct shape and type
        if len(segmented_clothing.shape) != 3 or segmented_clothing.shape[2] != 3:
            raise ValueError(f"Invalid image shape: {segmented_clothing.shape}")
            
        # Convert to uint8 if needed
        if segmented_clothing.dtype != np.uint8:
            segmented_clothing = (segmented_clothing * 255).astype(np.uint8)
            
        # Convert BGR to RGB
        rgb_img = cv2.cvtColor(segmented_clothing, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        
        # Process image
        tensor = reid_preprocess(pil_img)
        tensor = tensor.unsqueeze(0)
        
        # Run inference
        reid_output = reid_session.run(None, {reid_session.get_inputs()[0].name: tensor.numpy()})[0][0]
        reid_feat = reid_output / (np.linalg.norm(reid_output) + 1e-6)
        return reid_feat
        
    except Exception as e:
        print(f"Error extracting Re-ID features: {e}")
        return np.zeros(512)

def extract_color_features(segmented_clothing):
    """HSV color histogram features with error handling."""
    try:
        # Convert tensor to numpy if needed
        if torch.is_tensor(segmented_clothing):
            segmented_clothing = segmented_clothing.cpu().numpy()
        
        # Ensure correct data type
        segmented_clothing = np.asarray(segmented_clothing, dtype=np.uint8)
        
        hsv = cv2.cvtColor(segmented_clothing, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist
    except Exception as e:
        print(f"Error in extract_color_features: {e}")
        return np.zeros(512)  # Return zero features on error

def extract_texture_features(segmented_clothing):
    """LBP texture features with error handling."""
    try:
        # Convert tensor to numpy if needed 
        if torch.is_tensor(segmented_clothing):
            segmented_clothing = segmented_clothing.cpu().numpy()
            
        # Ensure correct data type
        segmented_clothing = np.asarray(segmented_clothing, dtype=np.uint8)
        
        gray = cv2.cvtColor(segmented_clothing, cv2.COLOR_BGR2GRAY)
        lbp = local_binary_pattern(gray, 24, 3, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 27), range=(0, 26))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-6)
        return hist
    except Exception as e:
        print(f"Error in extract_texture_features: {e}")
        return np.zeros(26)  # Return zero features on error
# Initialize models with error handling
try:
    yolo_model = YOLO("/Users/arjuntomar/Desktop/GPT/best_body.pt")
    metric = NearestNeighborDistanceMetric("cosine", matching_threshold=0.2)
    tracker = Tracker(metric=metric, max_age=30)

    # Fix 'pretrained' warning: Use weights='DEFAULT' instead
    segmentation_model = models.segmentation.deeplabv3_resnet101(weights="DEFAULT").eval()
    reid_session = InferenceSession('/Users/arjuntomar/Desktop/GPT/nvidia_reid_model_modified.onnx')
except Exception as e:
    print(f"Error loading models: {e}")
    sys.exit(1)

# Video processing setup
video_path = "/Users/arjuntomar/Desktop/GPT/GH010023_1.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    sys.exit(1)

feature_store = {}
frame_skip = 2  # Process every 3rd frame
frame_count = 0

# Preprocessing for Re-ID model
reid_preprocess = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue  # Skip processing for some frames

    # Person detection
    detections = yolo_model(frame)[0]

    persons = []
    for det in detections.boxes.data:
        x1, y1, x2, y2, conf, cls = det.cpu().numpy()
        if cls == 0 and conf > 0.4:
            persons.append([x1, y1, x2, y2])

    if not persons:
        continue  # Skip frame if no persons detected

    # Batch clothing segmentation
    fixed_size = (256, 256)
    person_crops = [cv2.resize(frame[int(y1):int(y2), int(x1):int(x2)], fixed_size) for x1, y1, x2, y2 in persons]

    # Convert person_crops to torch.Tensor
    person_crops = [torch.from_numpy(crop).permute(2, 0, 1).float() for crop in person_crops]

    clothing_masks = extract_clothing_batch(segmentation_model, person_crops)

    # Convert person_crops back to numpy arrays for OpenCV
    person_crops = [crop.permute(1, 2, 0).numpy() for crop in person_crops]

    # Create Detection objects
    detection_list = []
    for person in persons:
        bbox = person[:4]  # x1, y1, x2, y2
        conf = 1.0  # Dummy confidence score
        feature = np.zeros((1, 128))  # Dummy feature vector
        detection_list.append(Detection(bbox, conf, feature))

    # Update tracks
    tracker.predict()  # Update track states
    tracker.update(detection_list)  # Update tracker with new detections
    tracks = tracker.tracks  # Get the list of active tracks

    # Ensure the length of clothing_masks matches the number of tracks
    if len(clothing_masks) < len(tracks):
        clothing_masks.extend([None] * (len(tracks) - len(clothing_masks)))

    for track_idx, track in enumerate(tracks):
        track_id = track.track_id
        x1, y1, x2, y2 = track.to_tlbr()

        if track_idx >= len(clothing_masks):
            continue  # Handle possible mismatches

        clothing_mask = clothing_masks[track_idx]
        if clothing_mask is not None:
            segmented_clothing = cv2.bitwise_and(person_crops[track_idx], person_crops[track_idx], mask=clothing_mask)
            
            color_feat = extract_color_features(segmented_clothing)
            texture_feat = extract_texture_features(segmented_clothing)
            reid_feat = extract_reid_features(segmented_clothing, reid_preprocess, reid_session)
            
            # Combine features with proper shapes
            combined_features = np.concatenate([
                color_feat.flatten(),  # Shape: (512,)
                texture_feat.flatten(),  # Shape: (26,)
                reid_feat.flatten()  # Shape: (512,)
            ])
            
            feature_store[track_id] = combined_features

    # Visualization
    cv2.putText(frame, f'ID: {track_id}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


    # ...existing code for frame processing...

    # PCA and clustering at end of frame processing
    if feature_store:
        features = np.array(list(feature_store.values()))
        if len(features) >= 2:  # Need at least 2 samples
            try:
                # Print feature shape for debugging
                print(f"\nFeature matrix shape: {features.shape}")
                
                n_components = min(128, features.shape[0] - 1, features.shape[1])
                pca = PCA(n_components=n_components)
                reduced_features = pca.fit_transform(features)
                
                # Print explained variance
                explained_var = np.sum(pca.explained_variance_ratio_)
                print(f"Explained variance with {n_components} components: {explained_var:.2f}")
                
                # Perform clustering
                clustering = DBSCAN(eps=0.3, min_samples=2).fit(reduced_features)
                
                # Print clustering results
                n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
                print(f"Number of clusters found: {n_clusters}")
                print(f"Cluster labels: {clustering.labels_}")
                
            except Exception as e:
                print(f"Error during PCA/Clustering: {e}")
        else:
            print("Not enough samples for clustering")
    else:
        print("No features extracted, skipping clustering.")

    # Show frame and check for quit
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()