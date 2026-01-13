import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Rice Leaf Disease Detection")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("rice_leaf_disease_model.h5")

model = load_model()

class_names = [
    "Bacterial Leaf Blight",
    "Brown Spot",
    "Leaf Smut"
]

st.title("🌾 Rice Leaf Disease Detection")
st.write("Upload a rice leaf image")

uploaded_file = st.file_uploader("Choose image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((224, 224))
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    result = class_names[np.argmax(prediction)]

    st.success(f"✅ Predicted Disease: **{result}**")
