import face_recognition
import os
import pickle
import cv2
import numpy as np

KNOWN_FACES_DIR = "known_faces"
ENCODINGS_FILE = "encodings.pickle"

known_face_encodings = []
known_face_names = []

# Loop through each person's images in `known_faces`
for filename in os.listdir(KNOWN_FACES_DIR):
    image_path = os.path.join(KNOWN_FACES_DIR, filename)
    image = face_recognition.load_image_file(image_path)
    encoding = face_recognition.face_encodings(image)

    if len(encoding) > 0:
        known_face_encodings.append(encoding[0])
        known_face_names.append(os.path.splitext(filename)[0])
        print(f"✅ Encoded {filename}")
    else:
        print(f"⚠️ No face found in {filename}")

# Save encodings to file
with open(ENCODINGS_FILE, "wb") as f:
    pickle.dump({"encodings": known_face_encodings, "names": known_face_names}, f)

print(f"✅ Encoding complete. Saved to {ENCODINGS_FILE}")
