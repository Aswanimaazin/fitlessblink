import cv2
import mediapipe as mp
import math

# Initialize Mediapipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)

# Eye landmark indices
LEFT_EYE = [33, 159, 158, 133, 153, 144]
RIGHT_EYE = [362, 386, 385, 263, 380, 373]

def euclidean(p1, p2):
    return math.dist(p1, p2)

def calculate_ear(landmarks, eye_points, w, h):
    coords = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_points]
    # EAR formula
    vertical1 = euclidean(coords[1], coords[5])
    vertical2 = euclidean(coords[2], coords[4])
    horizontal = euclidean(coords[0], coords[3])
    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear

blink_count = 0
eye_closed = False

# Start webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        face = results.multi_face_landmarks[0]
        left_ear = calculate_ear(face.landmark, LEFT_EYE, w, h)
        right_ear = calculate_ear(face.landmark, RIGHT_EYE, w, h)
        avg_ear = (left_ear + right_ear) / 2.0

        # Eye is closed if EAR is less than threshold
        if avg_ear < 0.23:
            if not eye_closed:
                blink_count += 1
                eye_closed = True
        else:
            eye_closed = False

        # Display blink count
        cv2.putText(frame, f"Blinks: {blink_count}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    cv2.imshow("Eye Blink Counter", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
