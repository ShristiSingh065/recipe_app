import streamlit as st
from PIL import Image
import torch
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    AutoTokenizer, AutoModelForSeq2SeqLM
)


st.set_page_config(page_title="🍽️ AI Recipe Generator", page_icon="🍲", layout="centered")

st.markdown(
    "<h1 style='text-align: center; color: #ff7043;'>🧠 AI Recipe Generator</h1>",
    unsafe_allow_html=True
)
st.markdown("<p style='text-align: center;'>Upload any food image & get its recipe instantly!</p>", unsafe_allow_html=True)
st.markdown("---")

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
    outputs = t5_model.generate(
        **inputs,
        max_length=512,
        num_beams=4,
        no_repeat_ngram_size=3,
        early_stopping=True,
        repetition_penalty=2.0
    )
    return t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
# Recipe parser
def parse_recipe(text):
    lines = text.split("\n")
    ingredients = []
    instructions = []
    is_ingredient = True

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if "step" in line.lower() or "instruction" in line.lower():
            is_ingredient = False
        elif is_ingredient:
            ingredients.append(line)
        else:
            instructions.append(line)

    return ingredients, instructions
# UI
uploaded_file = st.file_uploader("Upload food image 🍛", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing image..."):
        caption = get_image_caption(image)
    st.success(f"Detected Dish Description: **{caption}**")

    with st.spinner("👨‍🍳 Generating recipe..."):
        recipe = generate_recipe(caption)

    # Parse into ingredients & instructions
    ingredients, instructions = parse_recipe(recipe)
    st.markdown(" 🧂 Ingredients")
    if ingredients:
        for item in ingredients:
            st.markdown(f"- {item}")
    else:
        st.write("No ingredients found.")

    st.markdown("### 👨‍🍳 Instructions")
    if instructions:
        for i, step in enumerate(instructions, 1):
            st.markdown(f"**Step {i}:** {step}")
    else:
        st.write("No instructions found.")

    # Optional: Download raw recipe
    with st.expander("📄 View raw AI output"):
        st.text_area("Raw Output", recipe, height=300)

    st.download_button("⬇️ Download Recipe", data=recipe, file_name="recipe.txt", mime="text/plain")

st.markdown("---")
st.markdown("<p style='text-align: center; font-size: 0.9em; color: gray;'>Made with ❤️ using Streamlit & Transformers</p>", unsafe_allow_html=True)