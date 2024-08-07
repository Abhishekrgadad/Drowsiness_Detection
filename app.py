import cv2
import numpy as np
from flask import Flask, render_template, Response

app = Flask(__name__)

# Load pre-trained face and eye detection models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def detect_drowsiness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    status = "Awake"
    color = (0, 255, 0)  # Green
    
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        if len(eyes) == 0:
            status = "Drowsy"
            color = (0, 0, 255)  # Red
        else:
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                
                # Check if eyes are open (very simple check)
                eye_roi = roi_gray[ey:ey+eh, ex:ex+ew]
                if np.average(eye_roi) < 50:  # If average pixel value is low, consider eyes closed
                    status = "Drowsy"
                    color = (0, 0, 255)  # Red
                    break
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    cv2.putText(frame, f"Status: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return frame, status

def gen_frames():
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: Camera not found or cannot be opened.")
        return
    while True:
        success, frame = camera.read()
        if not success:
            print("Error: Failed to capture image.")
            break
        else:
            frame, _ = detect_drowsiness(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
