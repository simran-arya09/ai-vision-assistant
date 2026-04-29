from ultralytics import YOLO
import os
import cv2
import time
import csv
from datetime import datetime
from distance import get_direction, estimate_distance
from voice import speak
import mediapipe as mp  

# Faster YOLO model
model = YOLO("yolov8n.pt")

# Face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

labels = {}
with open("labels.txt", "r") as f:
    for line in f:
        idx, name = line.strip().split(",")
        labels[int(idx)] = name

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
# Hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils


def draw_panel(frame):
    overlay = frame.copy()

    # Main side panel
    cv2.rectangle(overlay, (10, 50), (340, 360), (20, 20, 20), -1)

    # Footer controls bar
    cv2.rectangle(overlay, (0, 680), (1280, 720), (20, 20, 20), -1)

    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)


def mark_attendance(name):
    if name in ["Unknown", "None"]:
        return

    base_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(base_dir, "attendance.csv")

    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    file_exists = os.path.exists(filename)
    already_logged = False

    if file_exists:
        with open(filename, "r", newline="") as f:
            reader = csv.reader(f)

            for row in reader:
                if len(row) >= 2 and row[0] == name and row[1] == date_str:
                    already_logged = True
                    break

    if not already_logged:
        with open(filename, "a", newline="") as f:
            writer = csv.writer(f)

            if not file_exists:
                writer.writerow(["Name", "Date", "Time"])

            writer.writerow([name, date_str, time_str])

        print("Attendance saved:", filename)  

def run_detection(source=0, mode="1"):
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("Camera not found")
        return

    cv2.namedWindow("AI Vision Assistant V5", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("AI Vision Assistant V5", 1200, 700)

    last_spoken = 0
    cooldown = 4
    greeted = False

    voice_enabled = True
    gesture_time = 0

    prev_time = time.time()

    person_count = 0
    vehicle_count = 0
    total_seen = set()

    recording = False
    writer = None

    danger_objects = ["person", "car", "bus", "truck", "motorcycle", "bicycle", "dog"]
    nearby_objects = ["chair", "bench", "bag", "bottle", "cup", "laptop"]

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, (1280, 720))
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result_hands = hands.process(rgb)

        if result_hands.multi_hand_landmarks:

            for hand_landmarks in result_hands.multi_hand_landmarks:

                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                lm = hand_landmarks.landmark

                fingers = []
                tips = [8, 12, 16, 20]

                for tip in tips:
                    if lm[tip].y < lm[tip - 2].y:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                total = sum(fingers)

                now_gesture = time.time()

                if now_gesture - gesture_time > 2:

                    # Open Palm = mute/unmute voice
                    if total == 4:
                        voice_enabled = not voice_enabled

                    # Two fingers = switch mode
                    elif total == 2:
                        mode = "2" if mode == "1" else "1"

                    # Fist = start/stop recording
                    elif total == 0:

                        if not recording:
                            filename = f"recording_{int(time.time())}.avi"
                            fourcc = cv2.VideoWriter_fourcc(*"XVID")
                            writer = cv2.VideoWriter(
                                filename,
                                fourcc,
                                20.0,
                                (1280, 720)
                            )
                            recording = True

                        else:
                            recording = False
                            writer.release()
                            writer = None

                    gesture_time = now_gesture

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ---------------- FACE RECOGNITION ----------------
        detected_name = "None"

        faces = face_cascade.detectMultiScale(gray, 1.2, 5)

        for (fx, fy, fw, fh) in faces:

            face_img = gray[fy:fy+fh, fx:fx+fw]

            id_, conf = recognizer.predict(face_img)

            if conf < 80:
                detected_name = labels.get(id_, "Unknown")
            else:
                detected_name = "Unknown"

            cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (255, 0, 255), 2)

            cv2.putText(
                frame,
                detected_name,
                (fx, fy - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 255),
                2
            )

            if detected_name != "Unknown" and not greeted:
             if voice_enabled:   
                speak(f"Welcome back {detected_name}")
                mark_attendance(detected_name)
                greeted = True

        if len(faces) == 0:
            greeted = False

        # ---------------- YOLO OBJECT DETECTION ----------------
        results = model(frame, verbose=False)

        best_box = None
        best_width = 0

        for result in results:
            for box in result.boxes:

                conf = float(box.conf[0])

                if conf < 0.25:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if (x2 - x1) < 45 or (y2 - y1) < 45:
                    continue

                width = x2 - x1

                if width > best_width:
                    best_width = width
                    best_box = box

        status = "CLEAR"
        color = (0, 255, 0)
        focus_label = "None"

        if best_box is not None:

            x1, y1, x2, y2 = map(int, best_box.xyxy[0])

            cls = int(best_box.cls[0])
            label = model.names[cls]
            focus_label = label

            if label not in total_seen:
                total_seen.add(label)

                if label == "person":
                    person_count += 1

                elif label in ["car", "bus", "truck", "motorcycle", "bicycle"]:
                    vehicle_count += 1

            box_width = x2 - x1
            center_x = (x1 + x2) // 2

            direction = get_direction(center_x, 1280)
            distance = estimate_distance(box_width)

            if label in danger_objects and distance <= 1.5:
                status = "DANGER"
                color = (0, 0, 255)

            elif label in nearby_objects and distance <= 2.5:
                status = "NEARBY"
                color = (0, 255, 255)

            else:
                status = "CLEAR"
                color = (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

            cv2.rectangle(frame, (x1, y1 - 35), (x1 + 340, y1), color, -1)

            cv2.putText(
                frame,
                f"{label.upper()} | {direction} | {distance}m",
                (x1 + 10, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 0),
                2
            )

            now = time.time()

            if now - last_spoken > cooldown:

                if mode == "1":
                 if voice_enabled:
                    speak(f"{label} {direction}")

                elif mode == "2":

                    if distance <= 1:

                        if direction == "Left":
                           if voice_enabled:
                               speak("Move right")

                        elif direction == "Right":
                            if voice_enabled:speak("Move left")

                        else:
                           if voice_enabled:
                               speak("Stop obstacle ahead")

                    else:
                        if voice_enabled:
                            speak("Path clear")

                last_spoken = now

        # ---------------- HEADER ----------------
        cv2.rectangle(frame, (0, 0), (1280, 45), (15, 15, 15), -1)

        cv2.putText(
           frame,
           "AI VISION ASSISTANT V6",
            (20, 30),
             cv2.FONT_HERSHEY_SIMPLEX,
             0.95,
             (0, 255, 255),
              2
        )

        cv2.putText(
             frame,
             time.strftime("%H:%M:%S"),
             (1120, 30),
             cv2.FONT_HERSHEY_SIMPLEX,
             0.75,
             (255, 255, 255),
             2
        )

        # ---------------- UI PANEL ----------------
        draw_panel(frame)

        current = time.time()
        fps = int(1 / (current - prev_time)) if current != prev_time else 0
        prev_time = current

        mode_name = "Detection" if mode == "1" else "Navigation"
        
        badge_color = (0,255,0)

        if status == "DANGER":
           badge_color = (0,0,255)
        elif status == "NEARBY":
             badge_color = (0,255,255)

        cv2.rectangle(frame, (1030, 55), (1245, 95), badge_color, -1)

        cv2.putText(
             frame,
             status,
             (1085, 82),
             cv2.FONT_HERSHEY_SIMPLEX,
             0.8,
             (0,0,0),
              2
        )   
        info = [
    f"Mode: {mode_name}",
    f"Status: {status}",
    f"Focus: {focus_label}",
    f"User: {detected_name}",
    f"Voice: {'ON' if voice_enabled else 'OFF'}",
    "Gestures Active",
    f"FPS: {fps}",
    f"Persons: {person_count}",
    f"Vehicles: {vehicle_count}",
    f"Objects: {len(total_seen)}",
    "Q = Quit",
    "S = Screenshot",
    "R = Record"
]

        y = 90
        for text in info:
            cv2.putText(
                frame,
                text,
                (25, y),
                 cv2.FONT_HERSHEY_SIMPLEX,
                 0.72,
                 (230, 230, 230),
                 2
            )
            y += 28
            
            cv2.putText(
                frame,
               "[Q] Quit   [S] Screenshot   [R] Record",
              (360, 707),
              cv2.FONT_HERSHEY_SIMPLEX,
              0.65,
              (255,255,255),
              2
            )      
            
        if recording and writer is not None:
            writer.write(frame)

        cv2.imshow("AI Vision Assistant V5", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("s"):
            filename = f"capture_{int(time.time())}.png"
            cv2.imwrite(filename, frame)
            print("Saved", filename)

        elif key == ord("r"):

            if not recording:
                filename = f"recording_{int(time.time())}.avi"
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                writer = cv2.VideoWriter(filename, fourcc, 20.0, (1280, 720))
                recording = True
                print("Recording Started")

            else:
                recording = False
                writer.release()
                writer = None
                print("Recording Stopped")

    cap.release()

    if writer is not None:
        writer.release()

    cv2.destroyAllWindows()