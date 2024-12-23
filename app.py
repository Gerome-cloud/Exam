import streamlit as st
import tensorflow as tf
import os
from PIL import Image, ImageOps
import numpy as np

@st.cache_resource
def load_model():
    model_path = 'weather_model.keras'  # Ensure this path is correct
    try:
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            return model
        else:
            st.error(f"Model file not found at {model_path}")
            return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

st.write("# Weather Prediction")

file = st.file_uploader("Choose weather photo from computer", type=["jpg", "png"])

def import_and_predict(image_data, model):
    size = (64, 64)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    if model is not None:
        image = Image.open(file)
        st.image(image, use_column_width=True)
        prediction = import_and_predict(image, model)
        class_names = ['Cloudy', 'Rain', 'Shine', 'Sunrise']
        result = class_names[np.argmax(prediction)]
        st.success(f"OUTPUT: {result}")
    else:
        st.error("Model not loaded. Please check the model path or model file integrity.")
