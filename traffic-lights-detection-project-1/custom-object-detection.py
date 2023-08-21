from roboflow import Roboflow

rf = Roboflow(api_key="...")
project = rf.workspace("furkan-kzlay").project("traffic-lights-detection-project") 
model = project.version(1).model

# infer on a local image
print(model.predict("C:/Users/furka/Desktop/yolo/traffic-lights-detection-project-1/3.jpg", confidence=40, overlap=30).json())

# visualize your prediction
model.predict("C:/Users/furka/Desktop/yolo/traffic-lights-detection-project-1/3.jpg", confidence=40, overlap=30).save("prediction.jpg")
