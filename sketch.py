import streamlit as st
from PIL import Image
import cv2
import numpy as np
import io

# Function to convert image to pencil sketch
def pencil_sketch(image):
    img_gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    img_invert = cv2.bitwise_not(img_gray)
    img_smooth = cv2.GaussianBlur(img_invert, (21, 21), sigmaX=0, sigmaY=0)
    sketch = cv2.divide(img_gray, 255 - img_smooth, scale=256)
    return sketch

# Page config
st.set_page_config(page_title="Pencil Sketch App ✏️", layout="wide")

# Title and description
st.title("🎨 Sketcher")
st.markdown("""
Upload your image and convert it into a beautiful pencil sketch.  
Try adjusting the smoothness level in the sidebar!
""")

# Sidebar controls
st.sidebar.header("Settings")
smoothness = st.sidebar.slider("Smoothness (Gaussian Blur)", 1, 51, 21, step=2)

# Image upload
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Load image with PIL
    image = Image.open(uploaded_file).convert("RGB")

    # Show original image
    st.subheader("Original Image")
    st.image(image, use_column_width=True)

    # Convert to pencil sketch with adjustable smoothness
    img_gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    img_invert = cv2.bitwise_not(img_gray)
    img_smooth = cv2.GaussianBlur(img_invert, (smoothness, smoothness), sigmaX=0, sigmaY=0)
    sketch = cv2.divide(img_gray, 255 - img_smooth, scale=256)

    # Show sketch and original side by side
    st.subheader("Pencil Sketch")
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Original", use_column_width=True)
    with col2:
        st.image(sketch, caption="Sketch", use_column_width=True, clamp=True)

    # Download button for sketch
    buf = cv2.imencode('.png', sketch)[1].tobytes()
    st.download_button(
        label="Download Sketch as PNG",
        data=buf,
        file_name="pencil_sketch.png",
        mime="image/png"
    )
else:
    st.info("Please upload an image to get started.")
