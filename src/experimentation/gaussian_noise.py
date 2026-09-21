import math
import random
import numpy as np
import cv2

def add_gaussian_noise(sigma, mean, image_path):
    """
    Sigma: Average value of the Gausian nosise, if graphed, it is the vertex
    Mean: Deviation from the sigma
    Image: Image to add noise to
    """
    image = cv2.imread(image_path)
    if not image:
        raise ValueError("Image not found")
    