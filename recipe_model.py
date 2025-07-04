from transformers import AutoProcessor, AutoModelForImageClassification, pipeline
from PIL import Image
#import streamlit as st


processor = AutoProcessor.from_pretrained("Shresthadev403/food-image-classification")
model = AutoModelForImageClassification.from_pretrained("Shresthadev403/food-image-classification")

# Text generator for recipe
generator = pipeline("image-classification", model="julien-c/food101")
text_generator = pipeline("text2text-generation", model="flax-community/t5-recipe-generation")
def predict_dish(image: Image.Image):

    image = Image.open(image).convert("RGB")
    predictions = generator(image)
    top_prediction = predictions[0]['label'].replace(" ", "_")
    return top_prediction.lower()

def generate_recipe(dish, diet=None, cuisine=None, cook_time=None):
    prompt = f"Recipe for {dish}"
    filters = []
    if diet and diet != "Any":
        filters.append(diet)
    if cuisine and cuisine != "Any":
        filters.append(cuisine)
    if cook_time and cook_time != "Any":
        filters.append(f"ready in {cook_time}")
    filter_text = ", ".join(filters)
    prompt = f"generate a {filter_text} recipe for {dish}" if filter_text else f"generate recipe: {dish}"

    result = text_generator(prompt, max_length=300, do_sample=True)[0]['generated_text']
    return result

#with st.sidebar:
    st.header("History")
    for idx, rec in enumerate(st.session_state.get("history", [])):
         if st.button(f"🔄  {rec['title']}", key=f"hist_{idx}"):
            generate_recipe([rec])
            st.session_state.setdefault("history", []).extend(rec)


    