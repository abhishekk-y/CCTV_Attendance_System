import cv2
import face_recognition
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Error: Could not read frame.")
        continue

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame = np.array(rgb_frame, dtype=np.uint8)  # Ensure 8-bit format
    rgb_frame = np.ascontiguousarray(rgb_frame)  # Fix memory layout issues

    # Debugging: Print image details
    print(f"✅ Image Shape: {rgb_frame.shape}, Data Type: {rgb_frame.dtype}")

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_frame)

    if face_locations:
        print("✅ Face detected!")
        break

    cv2.imshow("Face Capture", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
