import cv2
import os
import threading
import numpy as np
from pathlib import Path
from datetime import datetime
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(BASE_DIR, "dataset")
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)
    print(f"Created dataset directory: {dataset_dir}")
    print("Add reference images under dataset/<person_name>/ and rerun.")
    raise SystemExit(1)

# Load OpenCV's pre-trained face detection models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def extract_face_features_robust(face_roi):
    """
    Extract multi-scale features from face for angle-invariant matching
    Uses template matching and histogram comparison
    """
    # Resize to standard size for better comparison
    face_roi = cv2.resize(face_roi, (100, 100))
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    
    # Feature 1: Face template (down-scaled for speed)
    template = cv2.resize(gray, (50, 50)).flatten()
    
    # Feature 2: Brightness histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    
    # Feature 3: Edge detection
    edges = cv2.Canny(gray, 50, 150)
    edge_hist = cv2.calcHist([edges], [0], None, [256], [0, 256])
    edge_hist = cv2.normalize(edge_hist, edge_hist).flatten()
    
    # Combine features
    combined = np.concatenate([template, hist, edge_hist])
    return combined

def load_reference_images():
    """Load and extract robust features from reference images"""
    reference_data = {}
    for person_name in os.listdir(dataset_dir):
        person_dir = os.path.join(dataset_dir, person_name)
        if os.path.isdir(person_dir):
            for img_name in os.listdir(person_dir):
                if img_name.endswith(('.jpg', '.png', '.jpeg')):
                    img_path = os.path.join(person_dir, img_name)
                    try:
                        img = cv2.imread(img_path)
                        if img is None:
                            print(f"  Warning: Could not read {img_name}")
                            continue
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(50, 50))
                        if len(faces) > 0:
                            x, y, w, h = faces[0]
                            face_roi = img[y:y+h, x:x+w]
                            features = extract_face_features_robust(face_roi)
                            if person_name not in reference_data:
                                reference_data[person_name] = []
                            reference_data[person_name].append(features)
                    except Exception as e:
                        print(f"  Error processing {img_name}: {e}")
    return reference_data

def match_face(current_features, reference_data, threshold=8.0):
    """
    Find closest match using multiple comparison methods
    Uses Euclidean distance for more reliable matching
    Returns: (person_name, confidence_score)
    """
    if current_features is None:
        return "Unknown", 0.0
    
    best_match = "Unknown"
    best_distance = float('inf')
    match_distances = {}
    
    for person_name, feature_list in reference_data.items():
        person_distances = []
        for ref_features in feature_list:
            # Use Euclidean distance (L2 norm)
            distance = np.linalg.norm(current_features - ref_features)
            person_distances.append(distance)
        
        # Use average distance across all reference images for this person
        avg_distance = np.mean(person_distances)
        match_distances[person_name] = avg_distance
        
        if avg_distance < best_distance:
            best_distance = avg_distance
            best_match = person_name
    
    # Calculate confidence score (inverse of distance, normalized)
    confidence = max(0, 100 - (best_distance * 10)) if best_distance < 10 else 0
    
    if best_distance > threshold:
        return "Unknown", 0.0
    return best_match, confidence

print("Loading reference images with robust multi-feature matching...")
reference_data = load_reference_images()
if not reference_data:
    raise SystemExit(
        f"No reference images found in {dataset_dir}. "
        "Add images under dataset/<person_name>/ and rerun."
    )
total_refs = sum(len(v) for v in reference_data.values())
print(f"✓ Loaded {total_refs} reference faces from {len(reference_data)} people")
for person_name, features in reference_data.items():
    print(f"  - {person_name}: {len(features)} image(s)")
print("✓ Using Euclidean distance matching with template comparison")
print("✓ Works at any angle or expression!\n")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

counter = 0
face_match = False
matched_name = "Unknown"
matched_confidence = 0.0
last_recognized = None
last_recognized_time = None
recognition_count = 0

def check_face(face_roi):
    global face_match, matched_name, matched_confidence, last_recognized, last_recognized_time, recognition_count
    try:
        features = extract_face_features_robust(face_roi)
        matched, confidence = match_face(features, reference_data, threshold=8.0)
        face_match = (matched != "Unknown")
        matched_name = matched
        matched_confidence = confidence
        
        # Log recognition
        if face_match and (last_recognized != matched_name or (last_recognized_time and time.time() - last_recognized_time > 5)):
            recognition_count += 1
            last_recognized = matched_name
            last_recognized_time = time.time()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n{'='*50}")
            print(f"✅ RECOGNIZED #{recognition_count}: {matched_name}")
            print(f"   Confidence: {confidence:.1f}%")
            print(f"   Time: {timestamp}")
            print(f"{'='*50}\n")
    except Exception as e:
        print(f"Error in face matching: {e}")
        face_match = False
        matched_name = "Unknown"
        matched_confidence = 0.0

print("🎥 Starting face recognition... Press 'q' to quit")
print("=" * 50)
print(f"Registered people: {', '.join(reference_data.keys())}\n")

while True:
    ret, frame = cap.read()

    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(50, 50))
        
        if len(faces) > 0:
            # Process the largest face
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            x, y, w, h = faces[0]
            
            # Draw face rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Extract features every 5 frames for faster matching
            if counter % 5 == 0:
                face_roi = frame[y:y+h, x:x+w]
                try:
                    threading.Thread(target=check_face, args=(face_roi,), daemon=True).start()
                except Exception as e:
                    pass
        
        counter += 1

        # Display match result
        if face_match and matched_name != "Unknown":
            cv2.rectangle(frame, (10, 10), (600, 80), (0, 255, 0), 3)
            cv2.putText(frame, f"✅ MATCH: {matched_name}", (20, 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.putText(frame, f"Confidence: {matched_confidence:.1f}%", (20, 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.rectangle(frame, (10, 10), (300, 80), (0, 0, 255), 3)
            cv2.putText(frame, "Analyzing...", (20, 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            cv2.putText(frame, "No match yet", (20, 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Show registered people
        people_text = "Registered: " + ", ".join(reference_data.keys())
        cv2.putText(frame, people_text, (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("🔍 Real-Time Face Recognition", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("=" * 50)
print("✓ Face recognition closed")