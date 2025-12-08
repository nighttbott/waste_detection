import cv2
from ultralytics import YOLO

# 1. Load your custom model
# Ensure 'best.pt' is in the same folder as this script!
model = YOLO("best2.pt")

# 2. Open the default camera (0 is usually the laptop webcam)
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

print("Press 'q' to quit.")

while True:
    # 3. Read a frame from the camera
    success, frame = cap.read()
    if not success:
        break

    # 4. Run YOLOv8 inference on the frame
    # conf=0.5 means only show detections with >50% confidence
    results = model(frame, conf=0.5)

    # 5. Visualize the results on the frame
    # .plot() draws the bounding boxes and labels automatically
    annotated_frame = results[0].plot()

    # 6. Display the resulting frame
    cv2.imshow("Waste Detection Test", annotated_frame)

    # 7. Press 'q' on keyboard to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()