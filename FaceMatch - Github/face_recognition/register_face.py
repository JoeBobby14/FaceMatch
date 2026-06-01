import cv2
import os
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(BASE_DIR, "dataset")

# Load OpenCV's pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def register_face():
    """Register a new user's face by capturing multiple images"""
    
    print("=" * 60)
    print("🔒 FACE REGISTRATION SYSTEM")
    print("=" * 60)
    
    # Get user name
    person_name = input("\n📝 Enter your name: ").strip()
    
    if not person_name:
        print("❌ Name cannot be empty!")
        return False
    
    # Create directory for this person
    person_dir = os.path.join(dataset_dir, person_name)
    os.makedirs(person_dir, exist_ok=True)
    
    # Check if person already exists
    existing_images = len([f for f in os.listdir(person_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
    if existing_images > 0:
        response = input(f"\n⚠️  {person_name} already has {existing_images} images. Add more? (y/n): ").strip().lower()
        if response != 'y':
            print("❌ Registration cancelled.")
            return False
    
    print(f"\n✓ Registering: {person_name}")
    print("=" * 60)
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    image_count = existing_images
    captured_count = 0
    frames_since_last_capture = 0
    target_images = 10  # Capture 10 images for better recognition
    
    print(f"\n📸 Capturing {target_images} images from different angles...")
    print("Instructions:")
    print("  • Press SPACE to capture image")
    print("  • Try different angles: left, right, up, down")
    print("  • Try different expressions: smile, normal, serious")
    print("  • Press 'q' to finish\n")
    
    while captured_count < target_images:
        ret, frame = cap.read()
        
        if not ret:
            print("❌ Camera error!")
            break
        
        # Flip for mirror view
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        
        # Detect face with Haar Cascade
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(100, 100))
        
        # Draw status bar
        cv2.rectangle(frame, (0, 0), (640, 60), (20, 20, 20), -1)
        cv2.putText(frame, f"Captured: {captured_count}/{target_images}", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        face_detected = False
        if len(faces) > 0:
            face_detected = True
            # Get largest face
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            x, y, w_face, h_face = faces[0]
            
            # Draw face box
            cv2.rectangle(frame, (x, y), (x + w_face, y + h_face), (0, 255, 0), 3)
            cv2.putText(frame, "Face Detected - Press SPACE to capture", (20, 500),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            # Red border when no face detected
            cv2.rectangle(frame, (10, 70), (630, 470), (0, 0, 255), 3)
            cv2.putText(frame, "No face detected - Position your face in the frame", (20, 500),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow("🔒 Face Registration", frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' ') and face_detected:  # SPACE to capture
            image_count += 1
            captured_count += 1
            filename = f"{person_name}_{image_count:03d}.jpg"
            filepath = os.path.join(person_dir, filename)
            cv2.imwrite(filepath, frame)
            print(f"  ✓ Image {captured_count}/{target_images} captured: {filename}")
            frames_since_last_capture = 0
        elif key == ord('q'):  # Q to quit
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 60)
    if captured_count > 0:
        print(f"✅ Registration complete!")
        print(f"   Name: {person_name}")
        print(f"   Images captured: {captured_count}")
        print(f"   Location: {person_dir}")
        print("=" * 60)
        return True
    else:
        print("❌ No images captured. Registration cancelled.")
        print("=" * 60)
        return False


if __name__ == "__main__":
    try:
        success = register_face()
        if success:
            print("\n🎉 You're all set! Run 'python face_recognition.py' to start recognition.")
        else:
            print("\n⚠️  Registration incomplete. Try again.")
    except KeyboardInterrupt:
        print("\n⚠️  Registration cancelled by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
