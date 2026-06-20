# MNIST-Digit-Recognition-Tool


An interactive, end-to-end Machine Learning web application that trains a Deep Neural Network (DNN) on the classic MNIST dataset and deploys it via Streamlit. Users can upload custom images of handwritten digits to test the model's predictive ability in real-time.

> **Model Motto:** *If you want to lie, lie with 100 percent confidence (P.S. ifykyk accuracy) :)*

---

## 🚀 Features
- **Deep Learning Model:** A sequential neural network architecture trained, optimized, and serialized natively through TensorFlow/Keras.
- **Interactive UI:** A minimal, dark-mode friendly Streamlit dashboard layout designed for seamless user interaction.
- **Dynamic Preprocessing Pipeline:** Automatically converts user image uploads to grayscale, resizes them cleanly to standard `(28, 28)` dimensions, handles dynamic min/max pixel normalizations, and checks global pixel means to automatically invert bright/dark contrast backgrounds to match true MNIST characteristics.

---
