# Real-Time Emotion/Stress Detector

🎯 **AI-powered emotion and stress detection from webcam video and voice for meeting analysis**

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-latest-orange.svg)

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/emotion-detector.git
cd emotion-detector

# Install dependencies
pip install -r requirements.txt

# Run the demo
python run_demo.py
```

**Or start web dashboard directly:**
```bash
python dashboard.py
# Visit: http://localhost:5002
```

## 🎯 Features

- **Face Emotion Detection**: Real-time emotion recognition using MediaPipe face landmarks
- **Voice Emotion Analysis**: Speech emotion detection using MFCC features
- **Live Dashboard**: Web-based real-time visualization with emotion statistics
- **Stress Level Monitoring**: Automatic stress level assessment

## 💡 Usage

### Standalone Detection:
- Press 'q' to quit face detection
- Press Ctrl+C to stop voice detection

### Web Dashboard:
- Click "Start Detection" to begin real-time analysis
- View live video feed with emotion overlay
- Monitor emotion statistics and stress levels

## 🛠️ Tech Stack

- **Computer Vision**: OpenCV, MediaPipe
- **Audio Processing**: librosa, sounddevice
- **ML Models**: scikit-learn RandomForest
- **Web Interface**: Flask, SocketIO, Chart.js