import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import streamlit as st
import os
import random

# Load MoveNet model
model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")

# Set local dataset path
DATASET_PATH = "./images"  # Ensure correct path

POSE_LABELS = {
    "standing": "Standing",
    "sitting": "Sitting",
    "crouching": "Crouching",
    "lying_down": "Lying Down",
    "running": "Running",
}

def load_random_image():
    """Load a random image from the LSP dataset."""
    if not os.path.exists(DATASET_PATH):  
        st.error(f"Dataset not found at {DATASET_PATH}. Please check the path.")  
        return None

    image_files = [f for f in os.listdir(DATASET_PATH) if f.endswith(".jpg")]
    if not image_files:
        st.error("No images found in dataset.")
        return None

    random_image = random.choice(image_files)
    image_path = os.path.join(DATASET_PATH, random_image)

    image = Image.open(image_path)
    return image

def preprocess_image(image):
    """Preprocess an image for MoveNet."""
    image = image.convert("RGB")  # Ensure 3 channels
    img_width, img_height = image.size  # Get original image size
    image_resized = image.resize((192, 192))  # Resize to MoveNet input size
    image_array = np.array(image_resized).astype(np.int32)  # Convert to int32 (MoveNet requirement)
    
    image_tensor = tf.convert_to_tensor(image_array)
    image_tensor = tf.expand_dims(image_tensor, axis=0)  # Add batch dimension
    image_tensor = tf.cast(image_tensor, tf.int32)  # Ensure int32 format

    return image_tensor, img_width, img_height

def detect_pose(image_tensor):
    """Detect pose keypoints from an image using MoveNet."""
    outputs = model.signatures["serving_default"](image_tensor)
    keypoints = outputs["output_0"].numpy()[0, 0, :, :]  # Shape: (17, 3)
    return keypoints

import numpy as np

def calculate_angle(p1, p2, p3):
    """Returns the angle (in degrees) between three points."""
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    return np.degrees(angle)

import numpy as np

def calculate_angle(p1, p2, p3):
    """Returns the angle (in degrees) between three points."""
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    return np.degrees(angle)

import numpy as np

def calculate_angle(p1, p2, p3):
    """Returns the angle (in degrees) between three points."""
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    return np.degrees(angle)

def classify_sport_pose(keypoints):
    # Extract keypoints
    head = keypoints[0]  
    shoulders = (keypoints[5], keypoints[6])  
    hips = (keypoints[11], keypoints[12])  
    knees = (keypoints[13], keypoints[14])  
    ankles = (keypoints[15], keypoints[16])  
    
    # Calculate angles
    torso_angle = calculate_angle(head, hips[0], ankles[0])  
    knee_angle = calculate_angle(hips[0], knees[0], ankles[0])  
    shoulder_angle = calculate_angle(shoulders[0], shoulders[1], hips[0])  

    # Ground detection (feet touching the ground)
    feet_on_ground = ankles[0][1] > 0.9  # Adjust threshold if needed

    ### **Sports Pose Classification**
    
    # 🏃 **Running / Sprinting**
    if torso_angle > 70 and knee_angle < 160 and not feet_on_ground:
        return "Running / Sprinting"

    # 🏀 **Jumping / Dunking**
    if not feet_on_ground and knee_angle > 140:
        return "Jumping / Dunking"

    # ⚽ **Kicking**
    if abs(knees[0][0] - ankles[0][0]) > 50 and knee_angle < 90:
        return "Kicking"

    # 🤿 **Diving**
    if torso_angle < 30 and not feet_on_ground:
        return "Diving"

    # 🤸 **Flipping / Tumbling**
    if not feet_on_ground and knee_angle < 90:
        return "Flipping / Tumbling"

    # ⚾ **Throwing / Pitching**
    if abs(shoulders[0][0] - shoulders[1][0]) > 50 and abs(shoulders[0][1] - shoulders[1][1]) > 30:
        return "Throwing / Pitching"

    # 🚴 **Cycling**
    if knee_angle < 120 and feet_on_ground and torso_angle > 45:
        return "Cycling"

    # 🏋️ **Weightlifting**
    if torso_angle > 80 and abs(shoulders[0][1] - shoulders[1][1]) < 20:
        return "Weightlifting"

    # 🥅 **Goalkeeping**
    if abs(shoulders[0][1] - shoulders[1][1]) > 50 and abs(hips[0][1] - hips[1][1]) > 30:
        return "Goalkeeping"

    return "Unknown Pose"


def draw_pose(image, keypoints, img_width, img_height):
    """Draw detected pose keypoints on the image."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image)
    ax.axis("off")

    # Scale keypoints to original image size
    for kp in keypoints:
        x, y, confidence = kp
        x *= img_width  # Scale x-coordinate
        y *= img_height  # Scale y-coordinate
        if confidence > 0.1:  # Adjust confidence threshold if needed
            ax.add_patch(patches.Circle((x, y), radius=5, color="red"))

    return fig  # Return the plotted figure

# Streamlit UI
st.title("MoveNet Pose Estimation with LSP Dataset")
st.write("Loading a random image from the LSP dataset and detecting keypoints.")

image = load_random_image()

if image:
    st.image(image, caption="LSP Dataset Image", use_container_width=True)

    image_tensor, img_width, img_height = preprocess_image(image)  # Preprocess image before passing to the model

    # Detect pose keypoints
    keypoints_pred = detect_pose(image_tensor)

    # Display results
    st.write("### Predicted Keypoints:")
    if np.all(keypoints_pred[:, 2] < 0.1):  # If confidence is too low for all keypoints
        st.write("No pose detected.")
    else:
        st.pyplot(draw_pose(image, keypoints_pred, img_width, img_height))

        # Classify Pose
        predicted_pose = classify_sport_pose(keypoints_pred)
        st.write(f"### Pose Detected: {predicted_pose}")