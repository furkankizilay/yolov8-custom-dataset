# Import necessary libraries
from ultralytics import YOLO
from PIL import Image

# Initialize the YOLOv8 model
model = YOLO('yolov8n.pt')

# Load an input image using the PIL library
img = Image.open("images/download.jpg")

# Perform object detection on the input image and save the annotated image with predictions
result = model.predict(source=img, save=True)