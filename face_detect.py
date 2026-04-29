import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mp_face = mp.solutions.face_detection
detector = mp_face.FaceDetection(min_detection_confidence=0.6)

prev_time = 0

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = detector.process(rgb)

    face_count = 0

    if results.detections:
        for detection in results.detections:
            face_count += 1

            bbox = detection.location_data.relative_bounding_box
            h, w, c = frame.shape

            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)

            conf = int(detection.score[0] * 100)

            cv2.rectangle(frame, (x, y), (x+bw, y+bh), (0,255,0), 3)
            cv2.rectangle(frame, (x, y-35), (x+180, y), (0,255,0), -1)
            cv2.putText(frame, f"Face {conf}%", (x+10, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)

    # FPS
    current_time = time.time()
    fps = int(1 / (current_time - prev_time)) if prev_time else 0
    prev_time = current_time

    cv2.putText(frame, f"FPS: {fps}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

    cv2.putText(frame, f"Faces: {face_count}", (20,80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

    cv2.imshow("AI Vision Pro", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()