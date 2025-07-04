from transformers import AutoProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import pandas as pd

recipe_df = pd.read_csv("data/recipe_dataset_200_final.csv")

processor = AutoProcessor.from_pretrained("Shresthadev403/food-image-classification")
model = AutoModelForImageClassification.from_pretrained("Shresthadev403/food-image-classification")
   


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
    dish = dish.lower().replace("_", " ").strip()
    filtered_df = recipe_df[recipe_df['dish'].str.lower() == dish]
    if diet and diet != "Any":
        filtered_df = filtered_df[filtered_df['diet'].str.lower() == diet.lower()]
    if cuisine and cuisine != "Any":
        filtered_df = filtered_df[filtered_df['cuisine'].str.lower() == cuisine.lower()]
    if cook_time and cook_time != "Any":
        if cook_time == "<15 mins":
            filtered_df = filtered_df[filtered_df['cook_time'] <= 15]
        elif cook_time == "15-30 mins":
            filtered_df = filtered_df[(filtered_df['cook_time'] > 15) & (filtered_df['cook_time'] <= 30)]
        elif cook_time == ">30 mins":
            filtered_df = filtered_df[filtered_df['cook_time'] > 30]

    if filtered_df.empty:
        return "Sorry, no recipe found with the selected filters."

    recipe = filtered_df.iloc[0]
    return f"🍽️ **Dish**: {recipe['dish']}\n\n" + \
           f"📋 **Ingredients**:\n{recipe['ingredients']}\n\n" + \
           f"👨‍🍳 **Instructions**:\n{recipe['instructions']}"
    
    
