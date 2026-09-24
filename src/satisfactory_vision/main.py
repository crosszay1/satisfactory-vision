import math
import random
import numpy as np
import cv2


def add_gaussian_noise(img, sigma=25/255, mean=0.0, rng=None):
    rng = rng or np.random.default_rng() # Use default random number generator if none is provided
    noisy = img + rng.normal(mean, sigma, img.shape).astype(np.float32)
    return np.clip(noisy, 0.0, 1.0) # Clip to 0-1

"""
Alright, let's try GAN
"""


