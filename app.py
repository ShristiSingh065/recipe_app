import streamlit as st
from PIL import Image
from recipe_model import predict_dish, generate_recipe

st.set_page_config(page_title="AI Recipe Chef", layout="centered", page_icon="🍳")
st.title("👨‍🍳 AI Recipe Chef")
st.write("Upload any food image and get a real AI-generated recipe with filters.")

# Upload
file = st.file_uploader("Upload food image", type=["jpg","jpeg","png"])
diet = st.selectbox("Dietary Preference", ["", "vegetarian", "non-veg", "gluten-free", "keto", "pescatarian"])
cuisine = st.selectbox("Cuisine", ["", "Indian","Italian","Chinese","Mexican","Mediterranean"])
cook_time = st.selectbox("Cook Time", ["", "<15 mins","<30 mins","<45 mins","1+ hour"])

if file:
    img = Image.open(file)
    st.image(img, caption="Your Image", use_column_width=True)
    with st.spinner("Thinking like a chef..."):
        dish = predict_dish(img)
        st.success(f"Detected Dish: **{dish}**")
        recipe = generate_recipe(dish, diet, cuisine, cook_time)
        st.subheader("📝 Recipe:")
        st.write(recipe)

