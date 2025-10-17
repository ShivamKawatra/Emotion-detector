import warnings
warnings.filterwarnings('ignore', category=UserWarning)
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2
import base64
import json
from emotion_detector import EmotionDetector
from voice_emotion import VoiceEmotionDetector
import threading
import time

app = Flask(__name__)
app.config['SECRET_KEY'] = 'emotion_detector'
socketio = SocketIO(app, cors_allowed_origins="*")

emotion_detector = EmotionDetector()
voice_detector = VoiceEmotionDetector()

@app.route('/')
def index():
    return render_template('dashboard.html')

detection_active = False

@socketio.on('start_detection')
def handle_detection():
    global detection_active
    if detection_active:
        return
    
    detection_active = True
    
    def detection_loop():
        global detection_active
        # Try different camera indices
        cap = None
        for i in range(3):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, test_frame = cap.read()
                if ret:
                    print(f"Camera {i} opened for web dashboard")
                    break
            cap.release()
            cap = None
        
        if not cap or not cap.isOpened():
            socketio.emit('error', {'message': 'Camera not accessible. Close other camera apps first.'})
            detection_active = False
            return
        
        try:
            while detection_active:
                ret, frame = cap.read()
                if not ret:
                    socketio.emit('error', {'message': 'Failed to grab frame'})
                    break
                    
                # Face emotion detection
                face_emotion, face_confidence = emotion_detector.predict_emotion(frame)
                
                # Encode frame for web display
                _, buffer = cv2.imencode('.jpg', frame)
                frame_data = base64.b64encode(buffer).decode('utf-8')
                
                # Send data to frontend
                socketio.emit('emotion_data', {
                    'frame': frame_data,
                    'face_emotion': face_emotion,
                    'face_confidence': face_confidence,
                    'timestamp': time.time()
                })
                
                time.sleep(0.1)  # 10 FPS
        finally:
            cap.release()
            detection_active = False
    
    # Run detection in separate thread
    thread = threading.Thread(target=detection_loop)
    thread.daemon = True
    thread.start()

@socketio.on('stop_detection')
def handle_stop():
    global detection_active
    detection_active = False

if __name__ == '__main__':
    socketio.run(app, debug=True, port=5002)