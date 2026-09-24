import math
import random
import numpy as np
import cv2
import torch
from torch import nn

def add_gaussian_noise(sigma, mean, image_path):
    """
    Sigma: Deviation from the sigma, the width of the graph
    Mean: Average value of the Gausian nosise, if graphed, it is the vertex
    Image: Image to add noise to
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found")
    
    height, width, channels = image.shape # Get shape

    """
    Create Gaussian noise (method 1, numpy)
    noise = np.random.normal(mean, sigma, dimensions).astype(np.uint8)
    """

    # using method 1 here
    noise = np.random.normal(mean, sigma, (height, width, channels)) #Insert height, width, image.shape

    noisy_image = image + noise

    noisy_image_clipped = np.clip(noisy_image, 0, 255) # Stay within bounds of 0-255
    # ^^^ When we add gaussian noise to an image, some pixel values may exceed the valid range of 0-255 for 8-bit images. 

    # Convert back to unsigned 8-bit integer type
    noisy_image_clipped_u8 = noisy_image_clipped.astype(np.uint8)

    return noisy_image_clipped_u8



class vae(nn.Module):
    def __init__(self):
        pass
    class Encoder(nn.Module):
        def __init__(self, latent_dim=2): # two dimensional latent space. output will be a 2d tensor
            """
            latent_dim: # of dimensions in the latent space
            """
            pass
            super().__init__()


def main():
    print(add_gaussian_noise(sigma=25, mean=25, image_path="src/experimentation/rainbow.jpg"))  