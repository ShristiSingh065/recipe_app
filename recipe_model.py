from transformers import AutoProcessor, AutoModelForImageClassification,pipeline
from PIL import Image
import torch
import requests
from ultralytics import YOLO
model = YOLO("yolov8s.pt")

def detect_objects(image_path):
    results = model(image_path)
    names = model.names

    
    detections = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = f"{names[cls_id]} ({conf:.2f})"
            detections.append(label)
    return detections

def load_classification_model():
    processor = AutoProcessor.from_pretrained("Shresthadev403/food-image-classification")
    model = AutoModelForImageClassification.from_pretrained("Shresthadev403/food-image-classification")
    return processor, model
def load_text_generator():
    return pipeline( "text2text-generation",model="distilgpt2")
processor, model = load_classification_model()
text_generator = load_text_generator()


def predict_dish(image: Image.Image):
    try:
        image = Image.open(image).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class_idx = logits.argmax(-1).item()
            label = model.config.id2label[predicted_class_idx]
            return label.lower().replace(" ", "_")
    except Exception as e:
        print(f"❌ Error in dish prediction: {e}")
        return "unknown_dish"



def generate_recipe(dish, diet=None, cuisine=None, cook_time=None):

    filters = []
    if diet and diet != "Any":
        filters.append(f"{diet} diet")
    if cuisine and cuisine != "Any":
        filters.append(f"{cuisine} cuisine")
    if cook_time and cook_time != "Any":
        filters.append(f"ready in {cook_time}")
        filter_text = ", ".join(filters)
    
    prompt = f"""
    Create a step-by-step recipe for {dish}.
    Include:
    - Ingredients with quantities
    - Step-by-step instructions cooking steps
    Make sure it's a {filter_text} recipe."""
    try:
        result = text_generator(prompt.strip(), max_length=282, do_sample=False)
        return result[0]['generated_text']
    except Exception as e:
        print(f"❌ Error generating recipe: {e}")
        return "Sorry, couldn't generate a recipe at the moment."
        


