import cv2
import time
import math
from ultralytics import YOLO

# --- CONFIGURATION ---
MODEL_PATH = 'best1.pt'  # Ensure this is in the same folder
CAM_ID = 0              # 0 for default webcam, 1 for external

# Load the model
print(f"Loading model: {MODEL_PATH}...")
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Initialize Camera
cap = cv2.VideoCapture(CAM_ID)
cap.set(3, 1280) # Width
cap.set(4, 720)  # Height

# Create a window for sliders
WINDOW_NAME = "Advanced Waste Detector"
cv2.namedWindow(WINDOW_NAME)

# --- TRACKBAR FUNCTIONS ---
def nothing(x):
    pass

# Create Sliders (Trackbars)
# Confidence: How sure the AI needs to be (0-100%)
cv2.createTrackbar("Confidence", WINDOW_NAME, 50, 100, nothing)
# NMS (IoU): How much boxes can overlap before being removed (0-100%)
cv2.createTrackbar("NMS/IoU", WINDOW_NAME, 45, 100, nothing)

# Colors for different classes (BGR Format)
CLASS_COLORS = {
    'plastic': (0, 255, 255),   # Yellow
    'paper': (255, 0, 0),       # Blue
    'metal': (0, 0, 255),       # Red
    'glass': (0, 255, 0),       # Green
    'trash': (128, 128, 128)    # Gray
}

# FPS Calculation variables
prev_frame_time = 0
new_frame_time = 0

print("Starting Camera... Press 'q' to quit.")

while True:
    success, frame = cap.read()
    if not success:
        print("Failed to read from camera.")
        break

    # 1. Get Slider Values
    # We divide by 100 to get a float between 0.0 and 1.0
    conf_threshold = cv2.getTrackbarPos("Confidence", WINDOW_NAME) / 100
    iou_threshold = cv2.getTrackbarPos("NMS/IoU", WINDOW_NAME) / 100

    # 2. Run Inference
    # We pass the slider values dynamically to the model
    results = model(frame, conf=conf_threshold, iou=iou_threshold, verbose=False)

    # 3. Process Detections
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Bounding Box Coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Confidence & Class
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            class_name = model.names[cls]

            # Choose Color (Default to White if class not in list)
            # This handles partial matches like 'plastic_bottle' -> 'plastic'
            color = (255, 255, 255)
            for key in CLASS_COLORS:
                if key in class_name.lower():
                    color = CLASS_COLORS[key]
                    break

            # 4. Draw Custom Visuals (Cleaner than default)
            # Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            # Label Background
            label = f"{class_name} {int(conf*100)}%"
            t_size = cv2.getTextSize(label, 0, fontScale=0.6, thickness=1)[0]
            c2 = x1 + t_size[0] + 10, y1 - t_size[1] - 10
            cv2.rectangle(frame, (x1, y1), c2, color, -1) # Filled rectangle
            
            # Label Text
            cv2.putText(frame, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, [0,0,0], 2)

    # 5. FPS Counter
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 6. Show Result
    cv2.imshow(WINDOW_NAME, frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()