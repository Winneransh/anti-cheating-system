import torch
from PIL import Image

def detect_cellphone(image_path):
    # Load YOLOv5 model (pre-trained on COCO dataset)
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    
    # Load image
    image = Image.open(image_path)
    
    # Run detection
    results = model(image)
    
    # Get detection results
    detections = results.pandas().xyxy[0]
    
    # Check for cell phone (class 67 in COCO dataset)
    cellphone_detections = detections[detections['class'] == 67]
    
    if len(cellphone_detections) > 0:
        print("Cell phone detected!")
        print(f"Confidence: {cellphone_detections['confidence'].values[0]:.2f}")
        return True
    else:
        print("No cell phone detected")
        return False

# Usage
image_path = "WIN_20250622_03_19_18_Pro.jpg"  # Replace with your image path
detect_cellphone(image_path)