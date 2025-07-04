from transformers import AutoProcessor, AutoModelForImageClassification, pipeline
from PIL import Image
import torch


processor = AutoProcessor.from_pretrained("Shresthadev403/food-image-classification")
model = AutoModelForImageClassification.from_pretrained("Shresthadev403/food-image-classification")

# Text generator for recipe
#generator = pipeline("image-classification", model="nateraw/food101")
text_generator = pipeline("text2text-generation", model="google/flan-t5-small")
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
    Create a detailed recipe for {dish}.
    Include:
    - Ingredients with quantities
    - Step-by-step instructions
    Make sure it's a {filter_text} recipe."""
    result = text_generator(prompt.strip(), max_length=400, do_sample=True)[0]['generated_text']
    return result
