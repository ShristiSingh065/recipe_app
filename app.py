import streamlit as st
from PIL import Image
from recipe_model import predict_dish, generate_recipe

st.set_page_config(page_title="AI Recipe Chef", layout="centered", page_icon="🍳")
st.title("👨‍🍳 AI Recipe Chef")
st.write("Upload any food image and get a real AI-generated recipe with filters.")

uploaded_image = st.file_uploader("Upload food image", type=["jpg", "jpeg", "png"])
st.sidebar.title("🔍 Filters")
diet = st.selectbox("Dietary Preference", ["vegetarian", "non-vegetarian"])
cuisine = st.selectbox("Cuisine", ["Indian", "Chinese", "Italian", "Mexican"])
cook_time = st.selectbox("Cook Time", ["<15 mins", "15-30 mins", ">30 mins"])

if uploaded_image:
    with st.spinner("Detecting dish..."):
        dish = predict_dish(uploaded_image)
    st.success(f"Detected Dish: {dish}")

    with st.spinner("Generating recipe..."):
        recipe = generate_recipe(dish, diet, cuisine, cook_time)
    st.subheader("📋 Ingredients & Instructions")
    st.text_area("Here's your recipe:", recipe, height=350)




# ---------- tiny helper to colour the diet tag ----------
def diet_badge(diet: str) -> str:
    colors = {
        "Vegetarian":   "#34c759",   # green
        "Vegan":        "#0a84ff",   # blue
        "Keto":         "#ff9f0a",   # orange
        "Gluten-Free":  "#ff375f",   # pink
        "Any":          "#8e8e93",   # grey
    }
    col = colors.get(diet, "#8e8e93")
    return (
        f"<span style='background:{col};color:white;"
        "border-radius:4px;padding:2px 6px;font-size:0.85rem;'>"
        f"{diet}</span>"
    )
#if file:
 #   img = Image.open(file)
  #  st.image(img, caption="Your Image", use_column_width=True)
   # with st.spinner("Thinking like a chef..."):
    #    dish = predict_dish(img)
     #   st.success(f"Detected Dish: **{dish}**")
      #  recipe = generate_recipe(dish, diet, cuisine, cook_time)
      #  st.subheader("📝 Recipe:")
      #  st.write(recipe)

