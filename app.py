import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import time

# Load MoveNet model
model = tf.saved_model.load('https://tfhub.dev/google/movenet/singlepose/lightning/4')

def detect_pose(image):
    # Convert image to tensor
    image = tf.image.resize_with_pad(image, 192, 192)  # Resize image
    image = tf.cast(image, dtype=tf.int32)  # Convert to integer
    image = tf.expand_dims(image, axis=0)  # Add batch dimension
    # Run model
    outputs = model.signatures["serving_default"](image)
    keypoints = outputs['output_0'].numpy()[0, 0, :, :]

    return keypoints  # Return keypoints

def draw_pose(image_path, keypoints):
    image = Image.open(image_path)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image)
    ax.axis("off")

    # Draw keypoints
    for kp in keypoints:
        x, y, confidence = kp
        x *= image.width
        y *= image.height
        if confidence > 0.5:  # Draw only high-confidence keypoints
            ax.add_patch(patches.Circle((x, y), radius=5, color='red'))

    plt.show()

# Load and process an image
image_path = "person.jpg"  # Replace with the path to an input image
image = Image.open(image_path)
image_tensor = tf.convert_to_tensor(np.array(image))

# Detect pose
keypoints = detect_pose(image_tensor)

# Draw the pose on the image
draw_pose(image_path, keypoints)
