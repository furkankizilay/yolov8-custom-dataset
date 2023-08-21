# Import necessary libraries
from ultralytics import YOLO

# Initialize the YOLOv8 model
model = YOLO('yolov8n.pt')

# Perform predictions on the webcam feed (source = 0) and display results (show=True)
results = model.predict(source=0, show=True)

