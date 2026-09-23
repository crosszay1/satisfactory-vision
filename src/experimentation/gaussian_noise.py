import math
import random
import numpy as np
import cv2


def add_gaussian_noise(sigma, mean, image_path):
    """
    Sigma: Deviation from the sigma, the width of the graph
    Mean: Average value of the Gausian nosise, if graphed, it is the vertex
    Image: Image to add noise to
    """
    image = cv2.imread(image_path)
    if not image:
        raise ValueError("Image not found")

    height, width, channels = image.shape  # Get shape

    """
    Create Gaussian noise (method 1, numpy)
    noise = np.random.normal(mean, sigma, dimensions).astype(np.uint8)
    """

    # using method 1 here
    noise = np.random.normal(
        mean, sigma, (height, width, channels)
    )  # Insert height, width, image.shape

    noisy_image = image + noise

    noisy_image_clipped = np.clip(noisy_image, 0, 255)  # Stay within bounds of 0-255

    # Convert back to unsigned 8-bit integer type
    noisy_image_clipped_u8 = noisy_image_clipped.astype(np.uint8)

    return noisy_image_clipped_u8


if __name__ == "__main__":
    noisy_img = add_gaussian_noise(
        sigma=10, mean=10, image_path="src/experimentation/rainbow.jpg"
    )
    cv2.imwrite("src/experimentation/rainbow_noisy.jpg", noisy_img)
