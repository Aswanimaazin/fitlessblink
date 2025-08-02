# blink_detector.py
import cv2

blink_count = 0
eye_closed = False
running = True  # NEW FLAG

def generate_frames():
    global blink_count, eye_closed, running
    blink_count = 0
    eye_closed = False
    running = True

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera not accessible")

    while running:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        eyes = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

        face_rects = faces.detectMultiScale(gray, 1.3, 5)
        eye_detected = False

        for (x, y, w, h) in face_rects:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            eyes_rects = eyes.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes_rects:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0,255,0), 2)
                eye_detected = True

        if not eye_detected and not eye_closed:
            eye_closed = True
        elif eye_detected and eye_closed:
            blink_count += 1
            eye_closed = False

        cv2.putText(frame, f"Blinks: {blink_count}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


def stop():
    global running
    running = False
