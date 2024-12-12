import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np


@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('Recog.keras')
    return model

model = load_model()

st.write("""
# Weather Recognition System
""")
file = st.file_uploader("Choose a weather photo from your computer", type=["jpg", "png"])

def import_and_predict(image_data, model):
    size = (64, 64)
    image = image_data.resize(size, Image.Resampling.LANCZOS)
    img = np.asarray(image)
    img_reshape = np.expand_dims(img, axis=0)
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_container_width=True)
    prediction = import_and_predict(image, model)
    class_names = ['Cloudy', 'Rain', 'Shine', 'Sunrise']
    result = f"OUTPUT: {class_names[np.argmax(prediction)]}"
    st.success(result)
