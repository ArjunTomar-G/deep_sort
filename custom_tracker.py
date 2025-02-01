import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort.preprocessing import non_max_suppression
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import nn_matching

class CustomDeepSORT:
    def __init__(self, yolo_weights_path):
        """
        Initialize tracker with custom YOLOv8 weights
        Args:
            yolo_weights_path: Path to your YOLOv8 weights file
        """
        max_cosine_distance = 0.3
        nn_budget = None
        
        # Initialize tracker components
        metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric)
        
        # Load YOLOv8 model with custom weights
        self.detector = YOLO(yolo_weights_path)
    
    def detect(self, frame):
        """Run YOLOv8 detection"""
        results = self.detector(frame, verbose=False)[0]  # Get first frame result
        detections = []
        
        for r in results.boxes.data.cpu().numpy():
            x1, y1, x2, y2, conf, cls = r
            detection = np.array([x1, y1, x2, y2, conf])
            detections.append(detection)
            
        return np.array(detections) if detections else np.empty((0, 5))
    
    def update(self, frame):
        """Update tracker with new frame"""
        # Get detections
        detections = self.detect(frame)
        
        # Create Detection objects
        detection_list = []
        if len(detections) > 0:
            for det in detections:
                bbox = det[:4]  # x1, y1, x2, y2
                conf = det[4]   # confidence score
                feature = np.zeros((1, 128))  # Dummy feature vector
                
                detection_list.append(Detection(bbox, conf, feature))
        
        # Update tracker
        self.tracker.predict()
        self.tracker.update(detection_list)
        
        return self.tracker.tracks
    
    def draw_results(self, frame, tracks):
        """Draw tracking results"""
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            bbox = track.to_tlbr()
            cv2.rectangle(frame, 
                        (int(bbox[0]), int(bbox[1])), 
                        (int(bbox[2]), int(bbox[3])),
                        (0, 255, 0), 2)
            
            # Draw ID
            text = f"ID: {track.track_id}"
            cv2.putText(frame, text,
                        (int(bbox[0]), int(bbox[1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)
        
        return frame

def main():
    # Initialize tracker with your YOLOv8 weights
    yolo_weights_path = "/Users/arjuntomar/Desktop/GPT/best_body.pt"  # Replace with your weights path
    tracker = CustomDeepSORT(yolo_weights_path)
    
    # Open webcam
    cap = cv2.VideoCapture(0)  # Use 0 for webcam or provide video path
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Update tracker
        tracks = tracker.update(frame)
        
        # Draw results
        frame = tracker.draw_results(frame, tracks)
        
        # Show frame
        cv2.imshow('Tracking', frame)
        
        # Break if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()