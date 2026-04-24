from ultralytics import YOLO
import cv2
import time
from distance import get_direction, estimate_distance
from voice import speak

model = YOLO("yolov8m.pt")


def run_detection(source=0):
    cap = cv2.VideoCapture(source)

    last_spoken_time = 0
    cooldown = 3

    if not cap.isOpened():
        print("Camera not found")
        return

    cv2.namedWindow("Vision Assist v2", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Vision Assist v2", 1000, 600)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, (960, 540))

        results = model(frame, verbose=False)

        for result in results:
            for box in result.boxes:

                confidence = float(box.conf[0])

                if confidence < 0.25:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cls = int(box.cls[0])
                label = model.names[cls]

                box_width = x2 - x1
                center_x = (x1 + x2) // 2

                direction = get_direction(center_x, 960)
                distance = estimate_distance(box_width)

                if label == "person":
                    color = (0, 255, 0)
                else:
                    color = (255, 0, 0)

                text = f"{label} | {direction} | {distance}m"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                cv2.putText(
                    frame,
                    text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

                current_time = time.time()

                if current_time - last_spoken_time > cooldown:
                    speak(f"{label} {direction} at {distance} meters")
                    last_spoken_time = current_time

        cv2.imshow("Vision Assist v2", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()