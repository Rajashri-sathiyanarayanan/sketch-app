import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("Sketchers ✏️")
st.write("Do you want to convert your pictures to realistic pencil sketches? Hurray! You are at the right destination then!")

def dodgeV2(x, y):
    return cv2.divide(x, 255 - y, scale=256)

def pencilsketch(inp_img):
    img_gray = cv2.cvtColor(inp_img, cv2.COLOR_BGR2GRAY)
    img_invert = cv2.bitwise_not(img_gray)
    img_smoothing = cv2.GaussianBlur(img_invert, (21, 21), sigmaX=0, sigmaY=0)
    final_img = dodgeV2(img_gray, img_smoothing)
    return final_img

file_image = st.sidebar.file_uploader("Upload your image", type=['jpeg', 'jpg', 'png'])

if file_image is None:
    st.write("You haven't uploaded any image yet!")
else:
    image = Image.open(file_image)
    img_array = np.array(image.convert('RGB'))  # Convert PIL Image to RGB then to NumPy array
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV

    st.write("**Input Photo**")
    st.image(image, use_column_width=True)

    sketch = pencilsketch(img_array)

    st.write("**Output Sketch**")
    st.image(sketch, use_column_width=True, clamp=True)
