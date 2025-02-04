import os
import cv2
import torch
import heapq
import numpy as np
from collections import deque
from ultralytics import YOLO
from onnxruntime import InferenceSession
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import torchvision.transforms as transforms

# ------------------------- PersonReIDAgent Class -------------------------
class PersonReIDAgent:
    def __init__(self, sim_threshold=0.6, embed_weight=0.7, hist_weight=0.3, max_entries=1000):
        """
        sim_threshold: similarity threshold to assign an id
        embed_weight: weight for image embeddings (70%)
        hist_weight: weight for color histogram (30%)
        max_entries: Maximum number of entries in the database
        """
        self.database = {} 
        # Structure: { id: {'embed': embedding_vector, 'hist': histogram_vector, 'count': update_count} }
        self.current_id = 0
        self.sim_threshold = sim_threshold
        self.embed_weight = embed_weight
        self.hist_weight = hist_weight
        self.max_entries = max_entries
        self.queue = deque()  # For recency-based cache management

    @staticmethod
    def cosine_similarity(vec_1, vec_2):
        dot = np.dot(vec_1, vec_2)
        mag_1 = np.linalg.norm(vec_1)
        mag_2 = np.linalg.norm(vec_2)
        return dot / (mag_1 * mag_2 + 1e-10)

    @staticmethod
    def histogram_similarity(hist_1, hist_2):
        # Compute histogram intersection similarity
        return np.minimum(hist_1, hist_2).sum() / (hist_2.sum() + 1e-10)

    def _update_access(self, person_id):
        if person_id in self.queue:
            self.queue.remove(person_id)
        self.queue.appendleft(person_id)
        if len(self.database) > self.max_entries:
            oldest = self.queue.pop()
            del self.database[oldest]

    def _get_new_id(self):
        self.current_id += 1
        return f"ID_{self.current_id:04d}"

    def add_to_database(self, embed, hist, person_id):
        count = self.database.get(person_id, {}).get('count', 0) + 1
        self.database[person_id] = {'embed': embed, 'hist': hist, 'count': count}
        self._update_access(person_id)

    def find_top_matches(self, query_embed, query_hist, top_k=5):
        scores = []
        for pid, data in self.database.items():
            embed_sim = self.cosine_similarity(query_embed, data['embed'])
            hist_sim = self.histogram_similarity(query_hist, data['hist'])
            combined = self.embed_weight * embed_sim + self.hist_weight * hist_sim
            heapq.heappush(scores, (combined, pid))
            if len(scores) > top_k:
                heapq.heappop(scores)
        return sorted(scores, reverse=True)

    def decide_id_assignment(self, query_embed, query_hist):
        """Decide on an ID based on the weighted similarity of embedding and histogram."""
        matches = self.find_top_matches(query_embed, query_hist)
        if not matches:
            new_id = self._get_new_id()
            self.add_to_database(query_embed, query_hist, new_id)
            return new_id, 1.0  # Maximum confidence for the first entry
        best_score, best_id = matches[0]
        if best_score >= self.sim_threshold:
            # Update existing entry
            self.add_to_database(query_embed, query_hist, best_id)
            return best_id, best_score
        else:
            new_id = self._get_new_id()
            self.add_to_database(query_embed, query_hist, new_id)
            return new_id, best_score

# ------------------------- Helper Functions -------------------------
def calculate_color_histogram(image, bins=8):
    """Calculate a normalized color histogram in HSV color space."""
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    channels = [0, 1, 2]  
    hist_size = [bins, bins, bins]
    ranges = [0, 180, 0, 256, 0, 256]
    histogram = cv2.calcHist([hsv_image], channels, None, hist_size, ranges)
    cv2.normalize(histogram, histogram, alpha=0, beta=1, norm_type=cv2.NORM_L1)
    return histogram.flatten()

def extract_embedding(cropped_frame, session, input_name, output_name, preprocess):
    """Extract feature embedding from a cropped person image using the ONNX model."""
    image = Image.fromarray(cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB))
    input_tensor = preprocess(image).unsqueeze(0).numpy()
    output = session.run([output_name], {input_name: input_tensor})
    embedding = output[0][0]
    return embedding / np.linalg.norm(embedding)

# ------------------------- Model Initialization -------------------------
# Paths and parameters
video_path = "/Users/arjuntomar/Desktop/Person-ReID/Challenge1_course3_persons.mp4"  # Update with your video file path
yolo_weights = "/Users/arjuntomar/Desktop/Person-ReID/best_body.pt"  # YOLO model weights path
onnx_model_path = "/Users/arjuntomar/Desktop/Person-ReID/nvidia_reid_model_modified.onnx"  # ONNX Re-ID model path

# Initialize YOLO (ultralytics)
yolo_model = YOLO(yolo_weights)
yolo_conf_threshold = 0.85  # Adjustable confidence threshold

# Initialize ONNX session for Re-ID
session = InferenceSession(onnx_model_path)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Preprocessing for Re-ID model (resize, normalize, etc.)
preprocess = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Create output directory for first frames of new IDs
output_dir = "tracked_frames"
os.makedirs(output_dir, exist_ok=True)

# ------------------------- Initialize the Agent -------------------------
# This agent will use 70% of the similarity from the embedding and 30% from the color histogram.
person_agent = PersonReIDAgent(sim_threshold=0.65, embed_weight=0.7, hist_weight=0.3)

# ------------------------- Video Processing -------------------------
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    exit(1)

frame_count = 0
frame_skip = 1  # Process every frame (adjust if needed)
saved_ids = set()  # To store IDs for which the first frame has been saved

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    # Use YOLO to detect objects in the frame.
    # (The YOLO model returns a list; here we use the first prediction set.)
    detections = yolo_model(frame)[0]

    # For simplicity, process all person detections.
    # (You might want to loop over all detections; here we mimic your example by choosing the largest person.)
    largest_person = None
    max_area = 0

    for det in detections.boxes.data:
        # Unpack detection information
        x1, y1, x2, y2, conf, cls = det.cpu().numpy()
        # Assuming class 0 corresponds to 'person'
        if int(cls) == 0 and conf > yolo_conf_threshold:
            area = (x2 - x1) * (y2 - y1)
            if area > max_area:
                max_area = area
                largest_person = [int(x1), int(y1), int(x2), int(y2)]

    if largest_person is None:
        continue

    x1, y1, x2, y2 = largest_person
    cropped_person = frame[y1:y2, x1:x2]

    # Extract the embedding and color histogram features
    person_embedding = extract_embedding(cropped_person, session, input_name, output_name, preprocess)
    person_hist = calculate_color_histogram(cropped_person)

    # Decide the person ID using a combination of the embedding and histogram features.
    assigned_id, similarity = person_agent.decide_id_assignment(person_embedding, person_hist)

    # Save the first frame for this new ID if not already saved.
    if assigned_id not in saved_ids:
        frame_path = os.path.join(output_dir, f"{assigned_id}.jpg")
        cv2.imwrite(frame_path, frame)
        saved_ids.add(assigned_id)

    # Draw bounding box and ID on the frame
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, f"{assigned_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
