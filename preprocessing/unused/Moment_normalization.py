import os
from PIL import Image
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
from skimage.morphology import disk
from scipy.signal import windows
import cv2

def calculate_moments(image, order):
    """Calculates geometric moments of an image.

    Args:
        image (np.array): grayscale image.
        order (tuple): order of the moment (p, q).

    Returns:
        float: computed moment.
    """
    p, q = order
    rows, cols = image.shape
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    moment = np.sum(np.power(x, p) * np.power(y, q) * image)
    return moment

def calculate_central_moments(image, order):
    """Calculates central moments of an image.

    Args:
        image (np.array): grayscale image.
        order (tuple): order of the moment (p, q).

    Returns:
       float: computed central moment.
    """
    m00 = calculate_moments(image, (0, 0))
    if m00 == 0:
        return 0
    m10 = calculate_moments(image, (1, 0))
    m01 = calculate_moments(image, (0, 1))
    x_bar = m10 / m00
    y_bar = m01 / m00
    p, q = order
    rows, cols = image.shape
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    central_moment = np.sum(np.power(x - x_bar, p) * np.power(y - y_bar, q) * image)
    return central_moment

def calculate_nu_moment(image):
    """Calculates the nu moment of an image.

    Args:
        image (np.array): grayscale image.

    Returns:
        float: computed nu moment.
    """
    m00 = calculate_moments(image, (0, 0))
    if m00 == 0:
        return 0
    m01 = calculate_moments(image, (0, 1))
    y_bar = m01 / m00
    rows, cols = image.shape
    y = np.arange(rows)
    nu_moment = np.sum(np.abs(y - y_bar) * np.sum(image, axis=1))
    return nu_moment

def dilate_image(image, radius):
    """Dilates an image using a disk-shaped structuring element."""
    selem = disk(radius)
    dilated_image = binary_dilation(image, selem)
    # Ensure the output is in the same format as the input
    return dilated_image.astype(image.dtype)

def erode_image(image, radius):
    """Erodes an image using a disk-shaped structuring element."""
    selem = disk(radius)
    eroded_image = binary_erosion(image, selem)
    # Ensure the output is in the same format as the input
    return eroded_image.astype(image.dtype)

def calculate_stroke_thickness(image, rho=1):
    """Calculates the stroke thickness of an image.

    Args:
        image (np.array): grayscale image.
        rho (int): radius of structuring element.

    Returns:
        float: stroke thickness of the image.
    """
    # Ensure the image is binary (0 and 1) for dilation and erosion
    binary_image = (image > 0).astype(float)

    dilated_image = dilate_image(binary_image, rho)
    eroded_image = erode_image(binary_image, rho)
    gradient_image = dilated_image - eroded_image

    m00_f_prime = calculate_moments(binary_image, (0, 0))
    m00_g_f = calculate_moments(gradient_image, (0, 0))

    if m00_g_f == 0:
        return 0
    tau = 2 * rho * (m00_f_prime / m00_g_f)

    return tau

def normalize_stroke_thickness(image, target_thickness, epsilon=0.5, max_iterations=10):
    """Normalizes stroke thickness of an image.

    Args:
        image (np.array): grayscale image.
        target_thickness (float): target stroke thickness.
        epsilon (float): threshold for convergence.
        max_iterations (int): maximum number of iterations.

    Returns:
        np.array: stroke thickness normalized image.
    """
    # Ensure the image is binary (0 and 1) for dilation and erosion
    normalized_image = (image > 0).astype(float)

    for _ in range(max_iterations):
        tau = calculate_stroke_thickness(normalized_image)
        r = target_thickness - tau

        if np.abs(r) < epsilon or tau == 0:
            break
        if r > 0:
            # Use dilation only on the foreground
            normalized_image = dilate_image(normalized_image, int(np.ceil(r)))
        else:
            # Use erosion only on the foreground
            normalized_image = erode_image(normalized_image, int(np.ceil(-r)))

    return normalized_image
def reestimate_height(image, beta):
    """Re-estimates the height of an image.

   Args:
       image (np.array): grayscale image.
       beta (float): parameter for height re-estimation.

   Returns:
       float: re-estimated height of the image.
   """
    m00 = calculate_moments(image, (0, 0))
    if m00 == 0:
        return image.shape[0]
    nu_moment = calculate_nu_moment(image)
    h_prime = beta * (nu_moment / m00)
    return h_prime

def segment_image(image, h_prime, gamma1, gamma2, H):
    """Segments an image into slices using a sliding window."""
    slices = []
    height, width = image.shape
    window_width = int(gamma1 * h_prime)
    window_shift = int(gamma2 * h_prime)

    # Ensure window_width does not exceed the image width
    window_width = min(window_width, width)

    for i in range(0, max(1, width - window_width + 1), max(1, window_shift)):
        slice_start = i
        slice_end = min(i + window_width, width)
        slice_image = image[:, slice_start:slice_end]

        # Apply cosine window with proper broadcasting
        window = windows.cosine(slice_image.shape[1])
        window = window.reshape(1, -1)  # Reshape window to (1, window_width)
        slice_image = slice_image * window

        slices.append(slice_image)

    return slices
def normalize_slice(slice_image, H2, gamma1, alpha):
    """Normalizes a slice of an image.

    Args:
        slice_image (np.array): grayscale slice of the image.
        H2 (int): target height of normalized slice.
        gamma1 (float): parameter for normalized slice width.
        alpha (float): parameter for size normalization.

    Returns:
        tuple: normalized slice and feature vector.
    """
    H1, W1 = slice_image.shape
    m00 = calculate_moments(slice_image, (0, 0))
    if m00 == 0:
        normalized_slice = np.zeros((H2, int(gamma1 * H2)))
        features = [0, 0, 0, 0]
        return normalized_slice, features

    mu20 = calculate_central_moments(slice_image, (2, 0))
    mu02 = calculate_central_moments(slice_image, (0, 2))

    m10 = calculate_moments(slice_image, (1, 0))
    m01 = calculate_moments(slice_image, (0, 1))

    x_bar = m10 / m00 if m00 > 0 else 0
    y_bar = m01 / m00 if m00 > 0 else 0

    delta_x = alpha * np.sqrt(mu20 / m00) if m00 > 0 else 0
    delta_y = alpha * np.sqrt(mu02 / m00) if m00 > 0 else 0

    W2 = int(gamma1 * H2)
    normalized_slice = np.zeros((H2, W2), dtype=slice_image.dtype)
    rows, cols = np.indices(normalized_slice.shape)
    x_coords = ((cols / W2) - 0.5) * delta_x + x_bar
    y_coords = ((rows / H2) - 0.5) * delta_y + y_bar

    # Use linear interpolation for mapping
    for y_idx in range(H2):
        for x_idx in range(W2):
            x = x_coords[y_idx, x_idx]
            y = y_coords[y_idx, x_idx]

            if 0 <= y < H1 and 0 <= x < W1:
                x_floor, x_ceil = int(np.floor(x)), int(np.ceil(x))
                y_floor, y_ceil = int(np.floor(y)), int(np.ceil(y))

                if x_floor == x_ceil and y_floor == y_ceil:
                    normalized_slice[y_idx, x_idx] = slice_image[y_floor, x_floor]

                elif x_floor == x_ceil:
                    if 0 <= y_floor < H1 and 0 <= y_ceil < H1:
                        y_frac = y - y_floor
                        normalized_slice[y_idx, x_idx] = (
                                slice_image[y_floor, x_floor] * (1 - y_frac) +
                                slice_image[y_ceil, x_floor] * y_frac
                        )
                elif y_floor == y_ceil:
                    if 0 <= x_floor < W1 and 0 <= x_ceil < W1:
                        x_frac = x - x_floor
                        normalized_slice[y_idx, x_idx] = (
                                slice_image[y_floor, x_floor] * (1 - x_frac) +
                                slice_image[y_floor, x_ceil] * x_frac
                        )
                else:
                    if 0 <= y_floor < H1 and 0 <= y_ceil < H1 and 0 <= x_floor < W1 and 0 <= x_ceil < W1:
                        x_frac = x - x_floor
                        y_frac = y - y_floor
                        normalized_slice[y_idx, x_idx] = (
                                (slice_image[y_floor, x_floor] * (1 - x_frac) + slice_image[
                                    y_floor, x_ceil] * x_frac) * (1 - y_frac) +
                                (slice_image[y_ceil, x_floor] * (1 - x_frac) + slice_image[
                                    y_ceil, x_ceil] * x_frac) * y_frac
                        )

    features = [x_bar / W1, y_bar / H1, 2 * np.sqrt(mu20 / m00) / W1, 2 * np.sqrt(mu02 / m00) / H1] if m00 > 0 else [0,
                                                                                                                     0,
                                                                                                                     0,
                                                                                                                     0]
    return normalized_slice, features

def process_image(image_path, output_dir, target_thickness=10, beta=1.0, gamma1=2.0, gamma2=0.5, H2=32, alpha=4.0, epsilon=0.5):
    """Processes a single image."""
    try:
        image = Image.open(image_path).convert('L')
        image = np.array(image, dtype=np.float64) / 255.0
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return

    print("Original Image:")
    print(f"  Data Type: {image.dtype}")
    print(f"  Min/Max: {image.min()}, {image.max()}")

    normalized_image = normalize_stroke_thickness(image, target_thickness, epsilon)

    print("After Stroke Thickness Normalization:")
    print(f"  Data Type: {normalized_image.dtype}")
    print(f"  Min/Max: {normalized_image.min()}, {normalized_image.max()}")

    h_prime = reestimate_height(normalized_image, beta)
    H = image.shape[0]
    slices = segment_image(normalized_image, h_prime, gamma1, gamma2, H)

    all_features = []
    for i, slice_image in enumerate(slices):
        print(f"Slice {i}:")
        print(f"  Data Type: {slice_image.dtype}")
        print(f"  Min/Max: {slice_image.min()}, {slice_image.max()}")

        normalized_slice, features = normalize_slice(slice_image, H2, gamma1, alpha)

        print(f"Normalized Slice {i}:")
        print(f"  Data Type: {normalized_slice.dtype}")
        print(f"  Min/Max: {normalized_slice.min()}, {normalized_slice.max()}")

        all_features.append(np.concatenate((normalized_slice.flatten(), features)))

        output_path = os.path.join(output_dir, os.path.basename(image_path).replace(".png", f"_slice_{i}.png"))
        normalized_slice_img = Image.fromarray((normalized_slice * 255).astype(np.uint8))
        normalized_slice_img.save(output_path)

        print(f"Saved slice {i} to {output_path}")

    return all_features

def main(input_directory, output_directory):
    """Main function to process all images in the input directory.

    Args:
        input_directory (str): path to the input directory.
        output_directory (str): path to the output directory.
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    all_features = []
    for filename in os.listdir(input_directory):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(input_directory, filename)
            features = process_image(image_path, output_directory)
            if features is not None:
                all_features.extend(features)

    # PCA will go here on the all_features
    # print(all_features)
    # print(f"Total features extracted: {len(all_features)}")

if __name__ == "__main__":
    input_directory = os.path.expanduser('./IAM_deslanted/image')
    output_directory = os.path.expanduser('./IAM_deslanted_moment_normalized/image')

    main(input_directory, output_directory)