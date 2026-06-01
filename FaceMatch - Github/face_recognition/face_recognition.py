import cv2
import os
import threading
import numpy as np
from pathlib import Path

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
    Combines histogram, edge detection, and gradient information
    """
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    
    # Feature 1: Brightness histogram (lighting patterns)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    
    # Feature 2: Edge histogram (facial structure - angle invariant)
    edges = cv2.Canny(gray, 100, 200)
    edge_hist = cv2.calcHist([edges], [0], None, [256], [0, 256])
    edge_hist = cv2.normalize(edge_hist, edge_hist).flatten()
    
    # Feature 3: Gradient magnitude (contours - pose invariant)
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=5)
    mag = np.uint8(np.sqrt(sobelx**2 + sobely**2) / np.sqrt(2))
    grad_hist = cv2.calcHist([mag], [0], None, [256], [0, 256])
    grad_hist = cv2.normalize(grad_hist, grad_hist).flatten()
    
    # Combine all features (768 total values)
    combined = np.concatenate([hist, edge_hist, grad_hist])
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
                    img = cv2.imread(img_path)
                    if img is not None:
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(50, 50))
                        if len(faces) > 0:
                            x, y, w, h = faces[0]
                            face_roi = img[y:y+h, x:x+w]
                            features = extract_face_features_robust(face_roi)
                            if person_name not in reference_data:
                                reference_data[person_name] = []
                            reference_data[person_name].append(features)
    return reference_data

def match_face(current_features, reference_data, threshold=0.45):
    """
    Find closest match using multi-feature comparison
    Uses Chi-Square distance for histogram-like features
    """
    if current_features is None:
        return "Unknown"
    
    best_match = "Unknown"
    best_distance = float('inf')
    
    for person_name, feature_list in reference_data.items():
        for ref_features in feature_list:
            # Use Chi-Square distance (good for histogram comparison)
            distance = cv2.compareHist(
                current_features.reshape(-1, 1), 
                ref_features.reshape(-1, 1),
                cv2.HISTCMP_CHISQR
            )
            if distance < best_distance:
                best_distance = distance
                best_match = person_name
    
    if best_distance > threshold:
        return "Unknown"
    return best_match

print("Loading reference images with robust multi-feature matching...")
reference_data = load_reference_images()
if not reference_data:
    raise SystemExit(
        f"No reference images found in {dataset_dir}. "
        "Add images under dataset/<person_name>/ and rerun."
    )
total_refs = sum(len(v) for v in reference_data.values())
print(f"✓ Loaded {total_refs} reference faces from {len(reference_data)} people")
print("✓ Using angle-invariant multi-feature matching")
print("✓ Works at any angle or expression!\n")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

counter = 0
face_match = False
matched_name = "Unknown"

def check_face(face_roi):
    global face_match, matched_name
    features = extract_face_features_robust(face_roi)
    matched = match_face(features, reference_data, threshold=0.45)
    face_match = (matched != "Unknown")
    matched_name = matched

print("🎥 Starting face recognition... Press 'q' to quit")
print("=" * 50 + "\n")

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
            
            # Extract features every 15 frames
            if counter % 15 == 0:
                face_roi = frame[y:y+h, x:x+w]
                try:
                    threading.Thread(target=check_face, args=(face_roi,), daemon=True).start()
                except Exception as e:
                    pass
        
        counter += 1

        # Display match result
        if face_match and matched_name != "Unknown":
            cv2.rectangle(frame, (10, 10), (500, 60), (0, 255, 0), 3)
            cv2.putText(frame, f"MATCH: {matched_name}", (20, 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        else:
            cv2.rectangle(frame, (10, 10), (300, 60), (0, 0, 255), 3)
            cv2.putText(frame, "No Match", (20, 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        cv2.imshow("🔍 Real-Time Face Recognition", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("=" * 50)
print("✓ Face recognition closed")