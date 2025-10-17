#!/usr/bin/env python3
"""
Quick demo script for Real-Time Emotion/Stress Detector
"""

import sys
import subprocess

def main():
    print("Real-Time Emotion/Stress Detector Demo")
    print("=====================================")
    print("1. Face Emotion Detection (OpenCV)")
    print("2. Voice Emotion Detection")
    print("3. Web Dashboard (Recommended)")
    print("4. Exit")
    
    while True:
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == '1':
            print("Starting face emotion detection... Press 'q' to quit")
            subprocess.run([sys.executable, 'emotion_detector.py'])
        elif choice == '2':
            print("Starting voice emotion detection... Press Ctrl+C to stop")
            subprocess.run([sys.executable, 'voice_emotion.py'])
        elif choice == '3':
            print("Starting web dashboard at http://localhost:5002")
            subprocess.run([sys.executable, 'dashboard.py'])
        elif choice == '4':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please select 1-4.")

if __name__ == "__main__":
    main()