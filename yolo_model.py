

from ultralytics import YOLO
from PIL import Image
import torch

# Load YOLOv8 pretrained model
model = YOLO("yolov8n.pt")  # For testing, you can later use custom trained food model

# Common food-related class names in COCO model
FOOD_CLASSES = [
    'apple', 'banana', 'orange', 'broccoli', 'carrot', 'hot dog',
    'pizza', 'cake', 'sandwich', 'donut', 'bottle', 'bowl'
]

def detect_ingredients(image_path):
    results = model(image_path)
    detected = set()

    for r in results:
        for c in r.boxes.cls:
            label = model.names[int(c)]
            if label in FOOD_CLASSES:
                detected.add(label)
    return list(detected)


def detect_ingredients(image_path):
    results = model(image_path)
    detected = set()

    for r in results:
        for c in r.boxes.cls:
            label = model.names[int(c)]
            if label in FOOD_CLASSES:
                detected.add(label)
    return list(detected)
