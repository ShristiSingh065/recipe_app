from transformers import AutoProcessor, AutoModelForImageClassification, pipeline
from PIL import Image

# 🔄 Use lightweight 85 MB food classifier
processor = AutoProcessor.from_pretrained("Shresthadev403/food-image-classification")
model = AutoModelForImageClassification.from_pretrained("Shresthadev403/food-image-classification")

# Text generator for recipe
generator = pipeline("text-generation", model="distilgpt2")

def predict_dish(image: Image.Image):
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    idx = outputs.logits.argmax(-1).item()
    return model.config.id2label[idx]

def generate_recipe(dish, diet=None, cuisine=None, cook_time=None):
    prompt = f"Recipe for {dish}"
    if diet:
        prompt += f", suitable for {diet}"
    if cuisine:
        prompt += f", in {cuisine} cuisine style"
    if cook_time:
        prompt += f", can be cooked in {cook_time}"
    prompt += ":"
    res = generator(prompt, max_length=150, temperature=0.7, do_sample=True)
    return res[0]["generated_text"]
