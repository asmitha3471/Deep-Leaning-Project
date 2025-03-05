import cv2
import os
import numpy as np

# Define input and output folders
INPUT_FOLDER = "mpii_images"  # Folder containing original images
OUTPUT_FOLDER = "mpii_images_compressed"  # Folder to save processed images

# Create output folder if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def compress_and_save_image(image_path, output_path, size=(256, 256), quality=50):
    """
    Resizes and compresses an image for storage.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Error loading image: {image_path}")

        resized_img = cv2.resize(img, size)
        cv2.imwrite(output_path, resized_img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return output_path
    except Exception as e:
        print(f"Compression Error: {e}")
        return None

# Compress all images in the input folder
for img_name in os.listdir(INPUT_FOLDER):
    img_path = os.path.join(INPUT_FOLDER, img_name)
    output_path = os.path.join(OUTPUT_FOLDER, img_name)
    compress_and_save_image(img_path, output_path)

print("Image compression complete!")

def preprocess_image(image_path, target_size=256):
    """
    Preprocesses an image for MoveNet model inference:
    - Loads the image.
    - Converts it to RGB.
    - Resizes while maintaining aspect ratio.
    - Pads to 256x256.
    - Normalizes for model input.
    Returns the processed image, padding values, and scale factor.
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Error loading image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape

        # Resize while maintaining aspect ratio
        scale = target_size / max(height, width)
        new_size = (int(width * scale), int(height * scale))
        resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)

        # Pad to (256, 256)
        pad_x = (target_size - new_size[0]) // 2
        pad_y = (target_size - new_size[1]) // 2
        padded_image = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        padded_image[pad_y:pad_y+new_size[1], pad_x:pad_x+new_size[0]] = resized_image

        return padded_image, pad_x, pad_y, scale
    except Exception as e:
        print(f"Preprocessing Error: {e}")
        return None, None, None, None