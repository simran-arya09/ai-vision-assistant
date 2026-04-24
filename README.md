Repository Name:
ai-vision-assistant


Repository Description:
AI-powered real-time vision assistance system using YOLOv8, OpenCV, and voice alerts for object awareness and navigation support.


README.md Content:

# AI Vision Assistant

AI-powered real-time computer vision system that detects surrounding objects and provides voice-based alerts with direction and approximate distance estimation.

## Features

- Real-time object detection using YOLOv8
- Laptop webcam + Phone IP camera support
- Voice alerts using pyttsx3
- Direction guidance (Left / Ahead / Right)
- Approx distance estimation
- Human/object color-coded bounding boxes
- Camera source selection menu

## Tech Stack

- Python
- OpenCV
- Ultralytics YOLOv8
- pyttsx3

## How to Run

python app.py

Choose camera source:

1 = Laptop Webcam
2 = Phone IP Camera

## Project Structure

app.py
detector.py
distance.py
voice.py
README.md

## Screenshots

### Detection Window

### Camera Selection Menu

### Phone Camera Integration

## Future Improvements

- Face recognition for known contacts
- Danger warning mode
- Hindi / multilingual voice alerts
- GUI dashboard
- Custom dataset training for better accuracy
- Emergency SOS feature

## Author

Simran Arya


.gitignore Content:

venv/
__pycache__/
*.pyc
*.pt


First Commit Message:

Initial AI vision assistant prototype with YOLO detection and voice alerts
