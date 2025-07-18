import streamlit as st
from PIL import Image
import torch
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    AutoTokenizer, AutoModelForSeq2SeqLM
)

st.title("AI CHEF 🍳")

# Load models (cached)
@st.cache_resource
def load_models():
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    t5_tokenizer = AutoTokenizer.from_pretrained("flax-community/t5-recipe-generation")
    t5_model = AutoModelForSeq2SeqLM.from_pretrained("flax-community/t5-recipe-generation")
    return blip_processor, blip_model, t5_tokenizer, t5_model

blip_processor, blip_model, t5_tokenizer, t5_model = load_models()

# BLIP: Get caption (description of food image)
def get_image_caption(img):
    inputs = blip_processor(img, return_tensors="pt")
    out = blip_model.generate(**inputs)
    caption = blip_processor.decode(out[0], skip_special_tokens=True)
    return caption

# T5: Generate recipe from caption
def generate_recipe(desc):
    prompt = f"Generate a recipe with ingredients and step-by-step instructions for: {desc}"
    inputs = t5_tokenizer(prompt, return_tensors="pt")
    outputs = t5_model.generate(**inputs, max_length=512)
    return t5_tokenizer.decode(outputs[0], skip_special_tokens=True)

# UI
uploaded_file = st.file_uploader("Upload food image 🍛", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing image..."):
        caption = get_image_caption(image)
    st.success(f"Detected Dish Description: **{caption}**")

    with st.spinner("Generating recipe..."):
        recipe = generate_recipe(caption)
    st.text_area("📋 Recipe", recipe, height=400)
