import os
import cv2
import numpy as np
from skimage.filters import threshold_otsu
# Define the input and output directories
input_dir = os.path.expanduser('~/IAM/image')
output_dir = os.path.expanduser('~/IAM_otsu_masked/image')

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Loop over all files in the input directory
for filename in os.listdir(input_dir):
    if filename.lower().endswith('.jpg'):
        # Construct full file path
        img_path = os.path.join(input_dir, filename)

        # Read the image in grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"Error: Could not read image {img_path}")
            continue

        # Create a mask for non-white pixels
        mask = img != 255
        tobin = img[mask]  # Extract non-white pixels

        # Check if tobin has content to apply Otsu threshold
        if tobin.size == 0:
            print(f"No content to threshold in {img_path}")
            continue

        # Compute the Otsu threshold on masked content
        thresh = threshold_otsu(tobin)

        # Apply threshold within masked area
        binarized_img = np.ones(img.shape, dtype=np.uint8) * 255  # Start with all-white image
        binarized_img[mask] = np.where(img[mask] > thresh, 255, 0)  # Binarize only within masked area

        # Construct output file path
        output_path = os.path.join(output_dir, filename)

        # Save the binarized image
        cv2.imwrite(output_path, binarized_img)

        print(f"Binarized and saved: {output_path}")

print("Processing complete.")