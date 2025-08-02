import cv2
import mediapipe as mp
import math

# Setup Mediapipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Eye landmarks (for one eye)
LEFT_EYE = [33, 159, 158, 133, 153, 144]  # Approximate
RIGHT_EYE = [362, 386, 385, 263, 380, 373]

# Calculate Euclidean distance
def euclidean(p1, p2):
    return math.dist(p1, p2)

# Calculate Eye Aspect Ratio (EAR)
def calculate_ear(landmarks, eye_indices, image_width, image_height):
    coords = [(int(landmarks[i].x * image_width), int(landmarks[i].y * image_height)) for i in eye_indices]
    vertical1 = euclidean(coords[1], coords[5])
    vertical2 = euclidean(coords[2], coords[4])
    horizontal = euclidean(coords[0], coords[3])
    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear

# Webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_ear = calculate_ear(face_landmarks.landmark, LEFT_EYE, w, h)
            right_ear = calculate_ear(face_landmarks.landmark, RIGHT_EYE, w, h)
            avg_ear = (left_ear + right_ear) / 2.0

            # EAR threshold for open/closed detection
            if avg_ear < 0.23:
                status = "Eyes Closed"
                color = (0, 0, 255)
            else:
                status = "Eyes Open"
                color = (0, 255, 0)

            # Draw text
            cv2.putText(frame, status, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    cv2.imshow("Eye Status Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
