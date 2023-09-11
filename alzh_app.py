import streamlit as st
import os
import numpy as np
from PIL import Image
from keras.preprocessing import image
import matplotlib.pyplot as plt
from keras.models import load_model

# Load your model
model = load_model('mobilenet_mri_model.h5')

input_shape = (224, 224)
num_classes = 4

st.title('Dementia Prediction')

# Allow the user to upload an image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg"])

if uploaded_image is not None:
    # Load and preprocess the uploaded image
    img = Image.open(uploaded_image).convert('RGB')
    img = img.resize(input_shape)
    img_array = np.array(img, dtype='float32')
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Make predictions
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    cats = ['Mild Dementia', 'Moderate Dementia', 'Very mild Dementia', 'Non Demented']
    predicted_class = cats[predicted_class_index]

    # Display the image and prediction
    st.image(img, caption=f'Predicted Class: {predicted_class}', use_column_width=True)
