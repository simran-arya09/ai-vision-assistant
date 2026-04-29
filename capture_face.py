import cv2
import os

name = input("Enter your name: ").strip()

folder = f"known_faces/{name}"
os.makedirs(folder, exist_ok=True)

cap = cv2.VideoCapture(0)

detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

count = 0
max_images = 20

print("Face capture started...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

        if count < max_images:
            face = gray[y:y+h, x:x+w]
            file = f"{folder}/{count}.jpg"
            cv2.imwrite(file, face)
            count += 1
            print("Saved", file)

    cv2.putText(frame, f"Captured: {count}/{max_images}",
                (20,40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255,255,255), 2)

    cv2.imshow("Capture Faces", frame)

    if count >= max_images:
        break

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("Capture complete.")