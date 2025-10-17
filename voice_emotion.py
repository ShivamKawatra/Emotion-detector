import librosa
import numpy as np
import sounddevice as sd
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

class VoiceEmotionDetector:
    def __init__(self):
        self.model = self._load_or_create_model()
        self.sample_rate = 22050
        self.duration = 3  # seconds
        
    def _load_or_create_model(self):
        if os.path.exists('voice_emotion_model.pkl'):
            with open('voice_emotion_model.pkl', 'rb') as f:
                return pickle.load(f)
        else:
            # Simple model with dummy training data
            model = RandomForestClassifier(n_estimators=10)
            X = np.random.rand(100, 13)  # 13 MFCC features
            y = np.random.randint(0, 4, 100)  # 4 emotions
            model.fit(X, y)
            with open('voice_emotion_model.pkl', 'wb') as f:
                pickle.dump(model, f)
            return model
    
    def extract_features(self, audio):
        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
        return np.mean(mfccs.T, axis=0)
    
    def record_audio(self):
        print("Recording...")
        audio = sd.rec(int(self.duration * self.sample_rate), 
                      samplerate=self.sample_rate, channels=1)
        sd.wait()
        return audio.flatten()
    
    def predict_emotion(self, audio):
        features = self.extract_features(audio)
        emotion_id = self.model.predict([features])[0]
        emotions = ['Neutral', 'Happy', 'Sad', 'Stressed']
        confidence = max(self.model.predict_proba([features])[0])
        return emotions[emotion_id], confidence
    
    def run_detection(self):
        print("Voice Emotion Detection Started. Press Ctrl+C to stop.")
        try:
            while True:
                audio = self.record_audio()
                emotion, confidence = self.predict_emotion(audio)
                print(f"Voice Emotion: {emotion}, Confidence: {confidence:.2f}")
        except KeyboardInterrupt:
            print("Detection stopped.")

if __name__ == "__main__":
    detector = VoiceEmotionDetector()
    detector.run_detection()