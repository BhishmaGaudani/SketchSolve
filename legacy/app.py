from pathlib import Path

import streamlit as st
import numpy as np
from streamlit_drawable_canvas import st_canvas
from tensorflow import keras
from PIL import Image

st.set_page_config(page_title="Digit Recognizer", page_icon="✏️")

st.title("✏️ Handwritten Digit Recognizer")
st.write("Draw a single digit (0–9) in the box below, then click **Predict**.")


MODEL_PATH = Path(__file__).parent / "mnist_cnn_model.keras"


@st.cache_resource
def load_model():
    return keras.models.load_model(MODEL_PATH)

model = load_model()

canvas_result = st_canvas(
    fill_color="black",
    stroke_width=18,
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    return_image_data=True,  # required since streamlit-drawable-canvas 0.13
    key="canvas",
)

col1, col2 = st.columns([1, 1])
with col1:
    predict_clicked = st.button("Predict", use_container_width=True)
with col2:
    st.caption("Tip: draw big and centered, like the MNIST examples.")

if predict_clicked:
    if canvas_result.image_data is None:
        st.warning("Draw a digit first.")
    else:
        img = Image.fromarray((canvas_result.image_data[:, :, 0:3]).astype("uint8"))
        img = img.convert("L")
        img = img.resize((28, 28))
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        prediction = model.predict(img_array)
        predicted_digit = int(np.argmax(prediction))
        confidence = float(np.max(prediction) * 100)

        st.subheader(f"Predicted digit: {predicted_digit}")
        st.write(f"Confidence: {confidence:.2f}%")

        st.write("Confidence across all digits:")
        st.bar_chart(prediction[0])
