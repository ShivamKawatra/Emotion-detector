import warnings
warnings.filterwarnings('ignore', category=UserWarning)
from flask import Flask, render_template, Response
import cv2
from emotion_detector import EmotionDetector
import time

app = Flask(__name__)
emotion_detector = EmotionDetector()

def generate_frames():
    # Try different camera indices
    cap = None
    for i in range(3):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, test_frame = cap.read()
            if ret:
                print(f"Camera {i} working for dashboard")
                break
        cap.release()
        cap = None
    
    if not cap or not cap.isOpened():
        print("No camera available")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get emotion
        emotion, confidence = emotion_detector.predict_emotion(frame)
        
        # Add text overlay
        cv2.rectangle(frame, (5, 5), (400, 80), (0, 0, 0), -1)
        cv2.putText(frame, f'Emotion: {emotion}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f'Confidence: {confidence:.2f}', (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('simple_dashboard.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, port=5003)