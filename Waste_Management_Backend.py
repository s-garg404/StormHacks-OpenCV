# waste_management_backend.py
import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np

# -----------------------------
# CONFIGURATION
# -----------------------------
mobilenet_path = '/Users/siagarg/PycharmProjects/StormHacks OpenCV/best_garbage_model.pt'
class_labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
conf_threshold = 0.2

coco_to_dataset = {
    'bottle': ['glass', 'plastic'],
    'can': ['metal'],
    'cup': ['plastic'],
    'box': ['cardboard'],
    'book': ['paper'],
    'fork': ['trash'],
    'spoon': ['trash']
}

# -----------------------------
# LOAD MODELS ONCE
# -----------------------------
print("Loading YOLOv5...")
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
yolo_model.eval()

print("Loading MobileNetV2...")
mobilenet_model = models.mobilenet_v2(weights=None)
mobilenet_model.classifier[1] = nn.Linear(mobilenet_model.last_channel, len(class_labels))
mobilenet_model.load_state_dict(torch.load(mobilenet_path, map_location='cpu'))
mobilenet_model.eval()

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -----------------------------
# FUNCTION FOR SINGLE IMAGE
# -----------------------------
def classify_waste_image(image_path):
    """Run YOLO + MobileNetV2 pipeline on a single image and return predicted category."""
    frame = cv2.imread(image_path)
    results = yolo_model(frame)
    detections = results.xyxy[0].cpu().numpy()

    final_labels = []

    for *xyxy, conf, cls in detections:
        if conf < conf_threshold:
            continue
        coco_label = yolo_model.names[int(cls)]
        if coco_label not in coco_to_dataset:
            continue

        x1, y1, x2, y2 = map(int, xyxy)
        cropped_img = frame[y1:y2, x1:x2]
        if cropped_img.size == 0:
            continue

        input_tensor = preprocess(cropped_img).unsqueeze(0)
        with torch.no_grad():
            outputs = mobilenet_model(input_tensor)
            _, predicted_class = torch.max(outputs, 1)
            label = class_labels[predicted_class.item()]
            final_labels.append(label)

    if not final_labels:
        return "Unknown"
    # Return the most common label in the detections
    return max(set(final_labels), key=final_labels.count)
