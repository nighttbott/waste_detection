# COPY THIS CODE BLOCK PER BLOCK IN GOOGLE COLAB AND RUN IN BLOCK PER BLOCK

# BLOCK CODE 1
# Make sure you have change your run time to GPU
!nvidia-smi

# BLOCK CODE 2
!pip install ultralytics roboflow

# BLOCK CODE 3
from google.colab import drive
from ultralytics import YOLO
from roboflow import Roboflow
import os

# --- 1. Connect to Google Drive ---
# A popup will appear asking for permission. Click "Connect to Google Drive".
drive.mount('/content/drive')

# --- 2. Download Dataset from Roboflow ---
# PASTE YOUR ROBOFLOW EXPORT CODE HERE
# It looks like: rf = Roboflow(api_key="..."); project = ...; dataset = ...
# Example placeholder (REPLACE THIS BLOCK):
# rf = Roboflow(api_key="xyz123")
# project = rf.workspace("my-workspace").project("waste-sorter")
# dataset = project.version(1).download("yolov8")

# --- 3. Setup Save Location ---
# We save directly to your Drive so you don't lose progress
drive_save_path = '/content/drive/MyDrive/YOLO11_Waste_Project'
os.makedirs(drive_save_path, exist_ok=True)
print(f"✅ Training results will be saved to: {drive_save_path}")

# --- 4. Train the Model ---
model = YOLO("yolo11n.pt")

results = model.train(
    # Your path of data.yaml
    data="data.yaml",
    epochs=125,
    imgsz=640,
    batch=16,
    # This 'project' argument sends all files to your Drive
    project=drive_save_path,
    name="waste_sorter_run",
    exist_ok=True
)

print(f"🎉 DONE! Download your model here: {drive_save_path}/waste_sorter_run/weights/best.pt")