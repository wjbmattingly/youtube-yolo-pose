import streamlit as st
from super_gradients.training import models
from super_gradients.common.object_names import Models
from PIL import Image
import torch
import io
import numpy as np

# Load the model
@st.cache_resource
def load_model():
    yolo_nas_pose = models.get("yolo_nas_pose_s", pretrained_weights="coco_pose")
    yolo_nas_pose.to('cuda' if torch.cuda.is_available() else 'cpu')
    return yolo_nas_pose

yolo_nas_pose = load_model()

# Modified prediction function
def make_prediction(uploaded_file, confidence=0.55):
    """
    Make a prediction using the fixed model and device, and return the image with predictions.

    Args:
    - uploaded_file: Streamlit UploadedFile object.
    - confidence (float, optional): Confidence threshold. Defaults to 0.75.

    Returns:
    - PIL.Image: Image with predictions.
    """
    # Convert the UploadedFile to a PIL Image
    image = Image.open(io.BytesIO(uploaded_file.getvalue()))

    np_image = np.array(image)


    yolo_nas_pose.predict(np_image, conf=confidence).save("")

    predictions = Image.open("pred_0.jpg")
    return predictions


# Streamlit UI
st.title("Streamlit YOLO Pose Demo")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Display original image
    original_image = Image.open(uploaded_file)
    predicted_image = make_prediction(uploaded_file, confidence=0.55)

    # Display images side by side
    col1, col2 = st.columns(2)
    with col1:
        st.image(original_image, caption="Original Image")
    with col2:
        st.image(predicted_image, caption="Prediction")
