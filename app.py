import streamlit as st
import requests
from PIL import Image
import io
import cv2
import numpy as np

def predict_image(image_data, api_endpoint, api_key):
    """
    Sends the image to Azure Custom Vision API for prediction.

    :param image_data: Binary image data
    :param api_endpoint: API endpoint for Azure Custom Vision
    :param api_key: API key for Azure Custom Vision
    :return: Prediction results as JSON
    """
    headers = {
        'Prediction-Key': api_key,
        'Content-Type': 'application/octet-stream'
    }
    response = requests.post(api_endpoint, headers=headers, data=image_data)

    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error: {response.status_code} - {response.text}")
        return None

# Streamlit app configuration
st.title("Crop Disease Detection")
st.write("Upload an image of a crop to detect potential diseases.")

# Azure Custom Vision API configuration
# Ensure the endpoint is correct for binary image data
api_endpoint = "https://centralindia.api.cognitive.microsoft.com/customvision/v3.0/Prediction/548ae86e-e8b1-42de-9c52-f299d46e5f1d/classify/iterations/Iteration2/image"
api_key = "c91abc22bc144b30b410f78e88b5cc76"

# File uploader for image upload
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Convert image to OpenCV format for additional processing
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Perform some OpenCV operations (e.g., edge detection)
    edges = cv2.Canny(image_cv, 100, 200)

    # Convert edges back to PIL Image for Streamlit display
    edges_pil = Image.fromarray(edges)
    # st.image(edges_pil, caption="Edge Detection Result", use_container_width=True, channels="GRAY")

    # Convert image to binary format
    image_bytes = io.BytesIO()
    if image.format != 'JPEG':
        image = image.convert('RGB')  # Ensure compatibility by converting to RGB
    image.save(image_bytes, format='JPEG')
    image_data = image_bytes.getvalue()

    # Predict using the Azure API
    with st.spinner("Analyzing the image..."):
        predictions = predict_image(image_data, api_endpoint, api_key)

    # Display the top prediction
    if predictions:
        st.subheader("Top Prediction:")
        top_prediction = max(predictions.get("predictions", []), key=lambda x: x['probability'])
        st.write(f"{top_prediction['tagName']}: {top_prediction['probability'] * 100:.2f}%")
