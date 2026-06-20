import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.title("MNIST Digit Recognition tool 24L2541")
st.header("Model Motto: If you want to lie, lie with 100 percent confidence (P.S. ifykyk accuracy) :)")
st.write("Upload a handwritten digit image (hands pls)")

model = tf.keras.models.load_model("best_model.h5")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    original = Image.open(uploaded_file).convert("L")
    st.image(original, caption="Uploaded Image", width=200)

    image_array = np.array(original)

    if np.mean(image_array) > 127:
        image_array = 255 - image_array
        st.write(" Auto inverted bg")

    resized = Image.fromarray(image_array).resize((28, 28))
    normalized = np.array(resized).astype(np.float32) / 255.0
    input_data = normalized.reshape(1, 784)

    probabilities = model.predict(input_data, verbose=0)
    predicted_digit = int(np.argmax(probabilities))
    confidence = float(np.max(probabilities))

    st.subheader(f"prediction= **{predicted_digit}**")
    st.write(f"confidence as  motto of the model = {confidence:.2%}")