# Make sure you have installed the library. If you haven't, the required libraries are listed in requirements.txt
import cv2
from ultralytics import YOLO

def main():
    # --- CONFIGURATION ---
    # 1. Path to your trained model
    # Make sure 'best.pt' is in the SAME folder as this script
    model_path = 'best.pt' 
    
    # 2. Camera ID
    # 0 is usually the built-in webcam. Try 1 if you have an external USB cam.
    camera_id = 0
    
    # --- LOAD MODEL ---
    print(f"Loading model from: {model_path}...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Tip: Make sure 'best.pt' is in this folder!")
        return

    # --- START CAMERA ---
    print(f"Starting camera {camera_id}...")
    cap = cv2.VideoCapture(camera_id)
    
    # Set a reasonable resolution (optional, but good for speed)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        return

    print("✅ System Ready! Press 'q' to quit.")

    while True:
        # 1. Read a frame from the camera
        success, frame = cap.read()
        if not success:
            print("Failed to read frame.")
            break

        # 2. Run YOLO inference
        # conf=0.5 means only show detections that are 50% confident
        results = model(frame, conf=0.5, verbose=False)

        # 3. Visualize the results
        # .plot() draws the bounding boxes and labels for you
        annotated_frame = results[0].plot()

        # 4. Display the frame
        cv2.imshow("Waste Detection Test", annotated_frame)

        # 5. Quit logic (Press 'q')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Program ended.")

if __name__ == "__main__":
    main()