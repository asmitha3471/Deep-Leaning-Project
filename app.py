import os
import random
import cv2
import numpy as np
import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from image_preprocessing import preprocess_image  # Import preprocessing function

# Set the path to preprocessed images
IMAGE_FOLDER = "mpii_images_compressed"  # Use the compressed images folder

# Load the MoveNet TFLite model
MODEL_PATH = "MoveNet.tflite"

try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

def load_random_image():
    """Load a random image from the preprocessed dataset folder."""
    if not os.path.exists(IMAGE_FOLDER):
        st.error(f"Dataset folder not found at {IMAGE_FOLDER}")
        return None

    image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith((".jpg", ".png"))]
    
    if not image_files:
        st.error("No images found in dataset.")
        return None

    random_image = random.choice(image_files)
    return os.path.join(IMAGE_FOLDER, random_image)

def detect_keypoints(image_path):
    try:
        # Load and preprocess the image
        original_image = cv2.imread(image_path)  # ✅ Load original image
        if original_image is None:
            raise ValueError("Image not found or cannot be loaded.")

        original_height, original_width, _ = original_image.shape  # ✅ Get original dimensions
        input_image, pad_x, pad_y, scale = preprocess_image(image_path)

        # Get model input details
        input_details = interpreter.get_input_details()
        input_dtype = input_details[0]['dtype']

        # Normalize based on input type
        if input_dtype == np.uint8:
            input_image = np.array(input_image, dtype=np.uint8)
        else:
            input_image = np.array(input_image, dtype=np.float32) / 127.5 - 1

        # Add batch dimension
        input_tensor = np.expand_dims(input_image, axis=0)

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()

        # Get output keypoints
        output_details = interpreter.get_output_details()
        keypoints = interpreter.get_tensor(output_details[0]['index'])[0][0]

        # Scale keypoints back to original image size
        keypoints[:, 0] = ((keypoints[:, 0] * 256 - pad_x) / scale * original_width / 256).astype(int)
        keypoints[:, 1] = ((keypoints[:, 1] * 256 - pad_y) / scale * original_height / 256).astype(int)

        return original_image, keypoints
    except Exception as e:
        st.error(f"Error in keypoint detection: {e}")
        return None, None

def draw_pose(image, keypoints):
    try:
        original_height, original_width, _ = image.shape  # Get image dimensions

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create a figure
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(image_rgb)
        ax.axis("off")  # Hide axes

        for keypoint in keypoints:
            x, y = keypoint[0], keypoint[1]

            # **Scale if keypoints are normalized (0-1 range)**
            if 0 <= x <= 1 and 0 <= y <= 1:
                x = int(x * original_width)
                y = int(y * original_height)

            # Ensure keypoints stay within image bounds
            x = max(0, min(x, original_width - 1))
            y = max(0, min(y, original_height - 1))

            ax.scatter(x, y, s=50, c="yellow", marker="o")  # Draw keypoints

        return fig
    except Exception as e:
        st.error(f"Error in drawing keypoints: {e}")
        return None
    
# Streamlit UI
st.title("MoveNet Pose Estimation on MPII Dataset")
st.write("Detecting human pose keypoints using MoveNet.")

image_path = load_random_image()

if image_path:
    st.image(image_path, caption="Input Image", use_container_width=True)
    image, keypoints_pred = detect_keypoints(image_path)

    if image is None:
        st.write("❌ Error in image processing.")
    elif keypoints_pred is None:
        st.write("⚠ No pose detected.")
    else:
        pose_fig = draw_pose(image, keypoints_pred)  # Get Matplotlib figure
        if pose_fig:
            st.pyplot(pose_fig)  # ✅ Display figure only if valid