import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm

# Input/output folders (FIXED with raw strings)
input_dir = r'C:\Users\saksa\Desktop\NUM\ITSATSUN\Year4\AI\Dataset\HandwrittenKhmerDigit\train'
output_dir = r'C:\Users\saksa\Desktop\NUM\ITSATSUN\Year4\AI\Dataset\HandwrittenKhmerDigit\augment'
os.makedirs(output_dir, exist_ok=True)

# Data augmentation configuration (More subtle and less aggressive parameters)
augmenter = ImageDataGenerator(
    rotation_range=5,          # Very small rotation (max ±5 degrees)
    zoom_range=0.02,           # Minimal zoom (±2%)
    width_shift_range=0.02,    # Slight horizontal shift (±2% of width)
    height_shift_range=0.02,   # Slight vertical shift (±2% of height)
    shear_range=0.02           # Minimal shearing
)

AUG_PER_IMAGE = 3  # Reduced number of augmented images per original

# Process each class folder
for class_name in os.listdir(input_dir):
    class_input_path = os.path.join(input_dir, class_name)
    class_output_path = os.path.join(output_dir, class_name)
    os.makedirs(class_output_path, exist_ok=True)

    # Iterate through each file in the class folder
    for fname in tqdm(os.listdir(class_input_path), desc=f"Class {class_name}"):
        img_path = os.path.join(class_input_path, fname)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Skip invalid files
        if img is None:
            print(f"Warning: Could not read image {img_path}. Skipping...")
            continue

        # Resize the image with high-quality interpolation
        img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_CUBIC)
        img = img.reshape((1, 28, 28, 1))

        # Save the original image to the output folder
        original_save_path = os.path.join(class_output_path, fname)
        cv2.imwrite(original_save_path, img.reshape(28, 28))

        # Generate augmented images
        i = 0
        for batch in augmenter.flow(img, batch_size=1):
            aug_img = batch[0].reshape(28, 28) * 255  # Rescale pixel values back to 0-255
            aug_img = aug_img.astype(np.uint8)
            out_fname = f"{fname.split('.')[0]}_aug{i}.png"
            out_path = os.path.join(class_output_path, out_fname)
            cv2.imwrite(out_path, aug_img)

            i += 1
            if i >= AUG_PER_IMAGE:  # Stop after generating the required number of augmentations
                break