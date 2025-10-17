import warnings
warnings.filterwarnings('ignore', category=UserWarning)
import cv2
import mediapipe as mp
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

class EmotionDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.model = self._load_or_create_model()
        
    def _load_or_create_model(self):
        if os.path.exists('emotion_model.pkl'):
            with open('emotion_model.pkl', 'rb') as f:
                return pickle.load(f)
        else:
            # Simple model with dummy training data
            model = RandomForestClassifier(n_estimators=10)
            X = np.random.rand(100, 20)  # 20 facial features
            y = np.random.randint(0, 4, 100)  # 4 emotions: neutral, happy, sad, stressed
            model.fit(X, y)
            with open('emotion_model.pkl', 'wb') as f:
                pickle.dump(model, f)
            return model
    
    def extract_features(self, landmarks):
        if not landmarks:
            return np.zeros(20)
        
        # Extract key facial points
        points = []
        for lm in landmarks.landmark[:20]:  # First 20 landmarks
            points.extend([lm.x, lm.y])
        
        return np.array(points[:20])  # Ensure exactly 20 features
    
    def predict_emotion(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            features = self.extract_features(landmarks)
            
            # Predict emotion
            emotion_id = self.model.predict([features])[0]
            emotions = ['Neutral', 'Happy', 'Sad', 'Stressed']
            confidence = max(self.model.predict_proba([features])[0])
            
            return emotions[emotion_id], confidence
        
        return 'No Face', 0.0
    
    def run_detection(self):
        # Try different camera indices
        cap = None
        for i in range(3):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                print(f"Camera {i} opened successfully")
                break
            cap.release()
        
        if not cap or not cap.isOpened():
            print("Error: Could not open camera. Please check:")
            print("1. Camera is connected and not used by other apps")
            print("2. Camera permissions are granted")
            return
        
        print("Press 'q' to quit")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            emotion, confidence = self.predict_emotion(frame)
            
            # Display results
            cv2.putText(frame, f'Emotion: {emotion}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f'Confidence: {confidence:.2f}', (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Emotion Detector', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = EmotionDetector()
    detector.run_detection()