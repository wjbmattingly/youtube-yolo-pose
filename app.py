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

def make_prediction(uploaded_file, confidence=0.55):
    image = Image.open(io.BytesIO(uploaded_file.getvalue()))
    np_image = np.array(image)
    st.write(np_image.shape)
    st.write(np_image.dtype)

    predictions = yolo_nas_pose.predict(np_image, conf=confidence)

    if hasattr(predictions, '_images_prediction_lst') and len(predictions._images_prediction_lst) > 0:
        # Use the 'draw' method to draw the predictions on the image
        predicted_image = predictions._images_prediction_lst[0].draw()
        return predicted_image
    else:
        raise image



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
