from transformers import AutoProcessor, AutoModelForImageClassification, pipeline
from PIL import Image
import torch
import pandas as pd


def load_classification_model():
    processor = AutoProcessor.from_pretrained("Shresthadev403/food-image-classification")
    model = AutoModelForImageClassification.from_pretrained("Shresthadev403/food-image-classification")
    return processor, model
def load_text_generator():
    print("🔁 Loading T5 model...")
    return pipeline("text2text-generation", model="zphang/t5-small-fr-en")
processor, model = load_classification_model()
text_generator = load_text_generator()


def predict_dish(image: Image.Image):
    image = Image.open(image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    label = model.config.id2label[predicted_class_idx]
    return label.lower().replace(" ", "_")


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
        result = text_generator(prompt.strip(), max_length=312, do_sample=False)
        return result[0]['generated_text']
    except Exception as e:
        print(f"❌ Error generating recipe: {e}")
        return "Sorry, couldn't generate a recipe at the moment."
