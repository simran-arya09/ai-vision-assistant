import cv2
import os
import numpy as np

data_path = "known_faces"

recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
labels = []
names = {}
current_id = 0

for person_name in os.listdir(data_path):

    person_folder = os.path.join(data_path, person_name)

    if not os.path.isdir(person_folder):
        continue

    names[current_id] = person_name

    for image_name in os.listdir(person_folder):

        image_path = os.path.join(person_folder, image_name)

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        faces.append(img)
        labels.append(current_id)

    current_id += 1

recognizer.train(faces, np.array(labels))
recognizer.save("trainer.yml")

# Save names
with open("labels.txt", "w") as f:
    for idx, name in names.items():
        f.write(f"{idx},{name}\n")

print("Training complete.")