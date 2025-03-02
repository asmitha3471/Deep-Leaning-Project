import tensorflow as tf
import numpy as np
import cv2
import streamlit as st
import random
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

# Load the TFLite model
MODEL_PATH = "MoveNet.tflite"

try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# Path to the images folder
DATASET_PATH = "./images"

def load_random_image():
    """Load a random image from the dataset folder."""
    if not os.path.exists(DATASET_PATH):
        st.error(f"Dataset folder not found at {DATASET_PATH}")
        return None

    image_files = [f for f in os.listdir(DATASET_PATH) if f.endswith((".jpg", ".png"))]
    
    if not image_files:
        st.error("No images found in dataset.")
        return None

    random_image = random.choice(image_files)
    image_path = os.path.join(DATASET_PATH, random_image)

    return image_path

def detect_keypoints(image_path):
    """Detect pose keypoints from an image using MoveNet."""
    image = cv2.imread(image_path)
    
    if image is None:
        st.error(f"Failed to load image: {image_path}")
        return None, None

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_height, original_width, _ = image.shape

    # MoveNet model options (Thunder = 256x256, Lightning = 192x192)
    INPUT_SIZE = 256  # Change to 192 if using Lightning

    # Resize image
    input_image = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))

    # Normalize the image to uint8 (0-255 range)
    input_image = np.array(input_image, dtype=np.uint8)  # Ensure uint8 format
    
    # Add a batch dimension
    input_tensor = np.expand_dims(input_image, axis=0)

    # Get input and output tensor indexes
    input_tensor_index = interpreter.get_input_details()[0]['index']
    output_tensor_index = interpreter.get_output_details()[0]['index']

    # Run inference
    interpreter.set_tensor(input_tensor_index, input_tensor)
    interpreter.invoke()

    # Get the keypoints
    keypoints = interpreter.get_tensor(output_tensor_index)[0][0]  # Shape: (17, 3)

    # Extract confidence scores
    confidences = keypoints[:, 2]

    # Lowered Confidence Threshold (from 0.3 → 0.1)
    CONFIDENCE_THRESHOLD = 0.1

    # Check if all keypoints have very low confidence
    if np.all(confidences < CONFIDENCE_THRESHOLD):
        st.warning("⚠ No pose detected with sufficient confidence. Try another image.")
        return image, None

    # Scale keypoints back to original image size
    keypoints[:, 0] = np.clip((keypoints[:, 0] * original_width).astype(int), 0, original_width - 1)
    keypoints[:, 1] = np.clip((keypoints[:, 1] * original_height).astype(int), 0, original_height - 1)

    return image, keypoints

def calculate_pose_details(keypoints):
    """Calculate pose details based on keypoints."""
    # Filter keypoints with high confidence
    confident_keypoints = [keypoint for keypoint in keypoints if keypoint[2] > 0.1]

    # Number of detected keypoints with sufficient confidence
    num_confident_keypoints = len(confident_keypoints)

    # Average confidence of the detected keypoints
    average_confidence = np.mean([keypoint[2] for keypoint in confident_keypoints]) if confident_keypoints else 0

    # Extract keypoints for shoulders, hips, and knees
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]
    left_hip = keypoints[11]
    right_hip = keypoints[12]
    left_knee = keypoints[13]
    right_knee = keypoints[14]

    # Refined criteria for pose type detection
    is_standing = False
    is_sitting = False

    # Standing Pose Criteria
    if abs(left_shoulder[1] - right_shoulder[1]) < 30 and abs(left_hip[1] - right_hip[1]) < 30:
        # Knees should be lower than hips for standing
        if left_knee[1] > left_hip[1] and right_knee[1] > right_hip[1]:
            is_standing = True

    # Sitting Pose Criteria (example, adjust as necessary)
    if left_knee[1] <= left_hip[1] and right_knee[1] <= right_hip[1]:
        is_sitting = True

    # Default to "Unknown" if neither criteria is met
    if is_standing:
        pose_type = "Standing"
    elif is_sitting:
        pose_type = "Sitting"
    else:
        pose_type = "Unknown"

    return num_confident_keypoints, average_confidence, pose_type

def draw_pose(image, keypoints):
    """Draw keypoints and skeleton on the image."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image)
    ax.axis("off")

    # Define keypoint connections (skeleton)
    connections = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Head and eyes
        (5, 6), (5, 7), (6, 8),  # Shoulders to elbows
        (7, 9), (8, 10),  # Elbows to wrists
        (5, 11), (6, 12), (11, 12),  # Shoulders to hips
        (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
    ]

    # Keypoint labels (optional, for reference)
    keypoint_names = [
        "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
        "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
        "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
        "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
    ]

    # Draw connections (lines between keypoints)
    for i, j in connections:
        if keypoints[i, 2] > 0.1 and keypoints[j, 2] > 0.1:  # Only draw lines if both keypoints have confidence > 0.1
            ax.add_line(Line2D([keypoints[i, 0], keypoints[j, 0]],
                               [keypoints[i, 1], keypoints[j, 1]], linewidth=3, color="cyan"))

    # Draw keypoints
    for i, (x, y, confidence) in enumerate(keypoints):
        if confidence > 0.1:  # Lowered threshold to 0.1
            ax.add_patch(Circle((x, y), radius=5, color="red", fill=True))
            ax.text(x + 5, y - 10, keypoint_names[i], fontsize=7, color="white",
                    bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.3'))

    return fig

# Streamlit UI
st.title("MoveNet Pose Estimation")
st.write("Loading a random image and detecting keypoints.")

image_path = load_random_image()

if image_path:
    st.image(image_path, caption="Input Image", use_container_width=True)

    image, keypoints_pred = detect_keypoints(image_path)

    if image is None:
        st.write("❌ Error in image processing.")
    elif keypoints_pred is None:
        st.write("⚠ No pose detected.")
    else:
        st.pyplot(draw_pose(image, keypoints_pred))

        # Calculate and display pose details
        num_confident_keypoints, average_confidence, pose_type = calculate_pose_details(keypoints_pred)

        st.subheader("Pose Details:")
        st.write(f"Number of detected keypoints with sufficient confidence: {num_confident_keypoints}")
        st.write(f"Average confidence of detected keypoints: {average_confidence:.2f}")
        st.write(f"Pose Type: {pose_type}")