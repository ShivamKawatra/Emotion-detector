import cv2

def test_camera():
    print("Testing camera access...")
    
    for i in range(5):
        print(f"Trying camera index {i}...")
        cap = cv2.VideoCapture(i)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"✅ Camera {i} works! Frame shape: {frame.shape}")
                cv2.imshow(f'Camera {i} Test', frame)
                cv2.waitKey(2000)  # Show for 2 seconds
                cv2.destroyAllWindows()
            else:
                print(f"❌ Camera {i} opened but can't read frames")
        else:
            print(f"❌ Camera {i} failed to open")
        
        cap.release()
    
    print("\nIf no cameras work, try:")
    print("1. Close other apps using camera (Zoom, Teams, etc.)")
    print("2. Check camera permissions")
    print("3. Restart your computer")

if __name__ == "__main__":
    test_camera()