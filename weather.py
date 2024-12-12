import streamlit as st
import tensorflow as tf
import os
from PIL import Image, ImageOps
import numpy as np

# Function to load the model with error handling
def load_model(model_path):
    if os.path.exists(model_path):
        try:
            model = tf.keras.models.load_model(model_path)
            return model
        except Exception as e:
            st.error(f"Error loading the model: {e}")
            return None
    else:
        st.error(f"Model file not found at {model_path}")
        return None

# Load the model
model_path = 'weather_model.keras'
st.write("Checking model path and file...")
st.write("Current working directory:", os.getcwd())
st.write("Files in directory:", os.listdir(os.getcwd()))

model = load_model(model_path)

st.write("""
# Weather Prediction
Upload an image to predict the weather condition.
""")

# File uploader to upload a weather-related photo
file = st.file_uploader("Choose an image file (JPG or PNG)", type=["jpg", "png"])

def import_and_predict(image_data, model):
    try:
        size = (64, 64)
        # Resize and preprocess the image
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        img = np.asarray(image)
        img_reshape = img[np.newaxis, ...]  # Add batch dimension
        prediction = model.predict(img_reshape)
        return prediction
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None

if file is None:
    st.text("Please upload an image file.")
else:
    if model is not None:
        try:
            # Display uploaded image
            image = Image.open(file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Make prediction
            prediction = import_and_predict(image, model)

            if prediction is not None:
                # Define class names
                class_names = ['Cloudy', 'Rain', 'Shine', 'Sunrise']
                result = class_names[np.argmax(prediction)]
                st.success(f"Prediction: {result}")
            else:
                st.error("Prediction failed.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.error("Model not loaded. Please check the model path.")
