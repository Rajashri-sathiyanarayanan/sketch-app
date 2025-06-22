import streamlit as st
import numpy as np
import cv2
from PIL import Image

def pencil_sketch(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray
    blur = cv2.GaussianBlur(inv, (21, 21), 0)
    sketch = cv2.divide(gray, 255 - blur, scale=256.0)
    return sketch

st.set_page_config(page_title="Image to Sketch", layout="centered")
st.title("🎨 Convert Image to Pencil Sketch")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    sketch = pencil_sketch(image_bgr)
    
    st.subheader("Original Image")
    st.image(image, use_column_width=True)
    
    st.subheader("Pencil Sketch Output")
    st.image(sketch, use_column_width=True, channels="GRAY")
    
    # Download Button
    sketch_pil = Image.fromarray(sketch)
    st.download_button("Download Sketch", data=sketch_pil.tobytes(), file_name="sketch.png")
