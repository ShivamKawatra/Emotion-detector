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

@socketio.on('start_detection')
def handle_detection():
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
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
    
    cap.release()

if __name__ == '__main__':
    socketio.run(app, debug=True, port=5002)