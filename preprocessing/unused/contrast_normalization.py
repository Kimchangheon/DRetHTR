import os
import cv2
import numpy as np


def contrast_normalize(img_path, output_path):
    """
    Performs contrast normalization on an image.

    Args:
        img_path: Path to the input image.
        output_path: Path to save the normalized image.
    """
    try:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Error: Could not read image at {img_path}")
            return

        # Calculate the percentiles
        lower_percentile = 5
        upper_percentile = 70

        # Handle single-value images
        if np.all(img == img[0, 0]):
            print(f"Warning: Image {img_path} has only one unique value. Saving as is.")
            cv2.imwrite(output_path, img)
            return

        lower_bound = np.percentile(img, lower_percentile)
        upper_bound = np.percentile(img, upper_percentile)

        # Stretch the histogram
        normalized_img = np.interp(img, (lower_bound, upper_bound), (0, 255)).astype(np.uint8)

        # Clip values outside the range
        normalized_img[normalized_img < 0] = 0
        normalized_img[normalized_img > 255] = 255

        cv2.imwrite(output_path, normalized_img)

    except Exception as e:
        print(f"Error processing image {img_path}: {e}")


def process_directory(input_directory, output_directory):
    """
    Processes all images in the input directory and saves the normalized images
    to the output directory.

    Args:
        input_directory: Path to the directory containing the input images.
        output_directory: Path to the directory where normalized images will be saved.
    """
    os.makedirs(output_directory, exist_ok=True)

    for filename in os.listdir(input_directory):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            input_path = os.path.join(input_directory, filename)
            output_path = os.path.join(output_directory, filename)
            contrast_normalize(input_path, output_path)
            print(f"Processed: {filename}")


if __name__ == "__main__":
    input_directory = os.path.expanduser('~/READ2016/image')
    output_directory = os.path.expanduser('~/READ2016_contrast_normalized/image')

    process_directory(input_directory, output_directory)