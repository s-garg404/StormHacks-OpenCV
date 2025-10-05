import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np

# -----------------------------
# CONFIGURATION
# -----------------------------

# Path to your trained MobileNetV2 model (.pt file)
mobilenet_path = '/Users/siagarg/PycharmProjects/StormHacks OpenCV/best_garbage_model.pt'

# Dataset classes
class_labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
num_classes = len(class_labels)

# Bin tracking
bin_fullness = {label: 0 for label in class_labels}
max_capacity = 10  # Arbitrary max for demo

# Confidence threshold for YOLO detections
conf_threshold = 0.2

# Map COCO YOLO classes to your dataset classes
coco_to_dataset = {
    'bottle': ['glass', 'plastic'],
    'can': ['metal'],
    'cup': ['plastic'],
    'box': ['paper'],
    'book': ['paper'],
    'fork': ['trash'],
    'spoon': ['trash']
}

# -----------------------------
# LOAD MODELS
# -----------------------------

print("Loading YOLOv5...")
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
yolo_model.eval()


print("Loading MobileNetV2...")
# Initialize MobileNetV2 architecture
mobilenet_model = models.mobilenet_v2(weights=None)
mobilenet_model.classifier[1] = nn.Linear(mobilenet_model.last_channel, num_classes)
# Load trained weights
mobilenet_model.load_state_dict(torch.load(mobilenet_path, map_location='cpu'))
mobilenet_model.eval()

# Define ImageNet preprocessing
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -----------------------------
# START WEBCAM
# -----------------------------

cap = cv2.VideoCapture(0)
print("Starting Smart Waste Detection (press 'q' to quit)")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO detection
    results = yolo_model(frame)
    detections = results.xyxy[0].cpu().numpy()  # xyxy, confidence, class

    for *xyxy, conf, cls in detections:
        if conf < conf_threshold:
            continue  # Skip low-confidence detections

        coco_label = yolo_model.names[int(cls)]

        # Only process objects relevant to your dataset
        if coco_label not in coco_to_dataset:
            continue

        x1, y1, x2, y2 = map(int, xyxy)
        cropped_img = frame[y1:y2, x1:x2]

        if cropped_img.size == 0:
            continue

        # Preprocess for MobileNetV2
        input_tensor = preprocess(cropped_img).unsqueeze(0)  # add batch dimension

        # Classify using trained MobileNetV2
        with torch.no_grad():
            outputs = mobilenet_model(input_tensor)
            _, predicted_class = torch.max(outputs, 1)
            label = class_labels[predicted_class.item()]

        # Update bin fullness
        bin_fullness[label] += 1

        # Draw detection and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Show cropped object for debugging
        cv2.imshow('Cropped Object', cv2.resize(cropped_img, (224,224)))

    # Display bin fullness
    for i, (category, count) in enumerate(bin_fullness.items()):
        fullness_percentage = (count / max_capacity) * 100
        color = (0, 255, 0) if fullness_percentage < 80 else (0, 0, 255)
        cv2.putText(frame, f'{category}: {fullness_percentage:.1f}%',
                    (10, 30 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if fullness_percentage >= 80:
            cv2.putText(frame, f'{category} Bin Full!',
                        (10, 30 + (i + 1) * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Show webcam frame
    cv2.imshow('Smart Waste Detection', frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Program ended.")
