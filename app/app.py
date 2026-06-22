import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

from gradcam_utils import generate_gradcam_image

# Load model
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "models",
    "mobilenet_model.keras"
)

model = tf.keras.models.load_model(MODEL_PATH)

# Page config
st.set_page_config(
    page_title="Cat vs Dog Classifier",
    layout="wide"
)

st.title("Image Classification using CNN")
st.write(
    "Upload an image and the model will predict whether it is a Cat or Dog."
)

# File uploader
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Original image
    original_img = Image.open(uploaded_file).convert("RGB")

    # Display uploaded image
    st.image(
        original_img,
        caption="Uploaded Image",
        width=300
    )

    # Resize for model prediction
    img = original_img.resize((224, 224))

    img_array = np.array(
        img,
        dtype=np.float32
    )

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    # Prediction
    prediction = model.predict(img_array)

    score = float(prediction[0][0])


    if score > 0.5:
        confidence = score * 100

        confidence = score * 100

        st.success(
            f"Prediction: Dog"
        )

        st.progress(confidence / 100)

        st.write(f"Confidence: {confidence:.2f}%")

    else:
        confidence = (1 - score) * 100

        confidence = (1 - score) * 100

        st.success(
            f"Prediction: Cat"
        )

        st.progress(confidence / 100)

        st.write(f"Confidence: {confidence:.2f}%")

    # Generate Grad-CAM
    try:

        gradcam_img = generate_gradcam_image(
            img_array,
            original_img,
            model
        )

        st.subheader("Grad-CAM Visualization")

        st.write("Red and yellow regions indicate the areas that contributed most to the model's prediction. "
        "Blue regions had little influence on the final decision.")

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                original_img,
                caption="Original Image",
                use_container_width=True
            )

        with col2:
            st.image(
                gradcam_img,
                caption="Grad-CAM",
                use_container_width=True
            )

    except Exception as e:

        st.error(
            f"Grad-CAM Error: {str(e)}"
        )