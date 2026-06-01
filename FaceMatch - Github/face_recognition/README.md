# FaceMatch - Real-Time Face Recognition System

FaceMatch is a real-time face recognition system that captures images of individuals, saves them into a dataset, and recognizes faces using DeepFace in real-time. The project is divided into two main scripts: one for capturing and saving images, and another for performing real-time face recognition.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Capture and Save Images](#capture-and-save-images)
  - [Real-Time Face Recognition](#real-time-face-recognition)
- [How It Works](#how-it-works)
- [Requirements](#requirements)
- [Contributing](#contributing)
- [License](#license)

## Features
- Capture images of individuals and save them with their names.
- Recognize faces in real-time using DeepFace.
- Display the name of the recognized individual or "No Match" if the face is not recognized.

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/FaceMatch.git
   cd FaceMatch
   ```

2. Install the required dependencies:
   ```sh
   pip install opencv-python-headless deepface
   ```

## Usage

### Capture and Save Images
This script (`facedata.py`) captures images from the webcam, asks for the person's name, and saves the images in a directory structure organized by the person's name.

1. Run the script:
   ```sh
   python facedata.py
   ```
2. Follow the on-screen instructions:
   - Press 's' to save an image.
   - Enter the name of the person.
   - Press 'q' to quit the script.

### Real-Time Face Recognition
This script (`face_recognition.py`) loads the saved images, uses them as reference images for face recognition, and displays the person's name if a match is found. If no match is found, it displays "No Match".

1. Run the script:
   ```sh
   python face_recognition.py
   ```

## How It Works
1. **Capture and Save Images** (`facedata.py`):
   - The script initializes the webcam and waits for the user to press 's' to save an image or 'q' to quit.
   - When 's' is pressed, the user is prompted to enter the name of the person.
   - The image is saved in a directory named after the person in the `dataset` directory.

2. **Real-Time Face Recognition** (`face_recognition.py`):
   - The script loads all images from the `dataset` directory as reference images.
   - It captures frames from the webcam and checks for a match every 30 frames using DeepFace.
   - If a match is found, it displays the name of the person. If no match is found, it displays "No Match".

## Requirements
- Python 3.6+
- OpenCV
- DeepFace

You can install the required packages using:
```sh
pip install opencv-python-headless deepface
```

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
