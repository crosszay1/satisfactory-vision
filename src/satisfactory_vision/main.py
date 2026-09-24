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

def discriminator(img): -> int
    pass

def satisfactory_to_rl(img): # Outputs image
    pass
def rl_to_satisfactory(img): # Outputs image
    pass

# Rough format
imgs = ["path", "path"]
for img in imgs:
    img = cv2.imread(img)
    img = satisfactory_to_rl(img)
    img = add_gaussian_noise(img)
    img = rl_to_satisfactory(img)

    discrim_output_int = discriminator(img)
    discrim_output_bool = True if discrim_output >= 0.5 else False

    if discrim_output_bool == False && imageWasSatisfactory(): # See if discriminator was right
        # If image was satisfactory, and discrim correctly identified reward generator, punish discrim
    elif discrim_output_bool == True && not imageWasSatisfactory(): # See if discriminator was right
        # If image was not satisfactory, but discrim incorrectly identified it as satisfactory, punish discrim
    elif discrim_output_bool == True && imageWasSatisfactory(): # See if discriminator was right
        # If image was satisfactory, and discrim correctly identified reward generator, reward discrim, punish gen
    elif discrim_output_bool == False && not imageWasSatisfactory(): # See if discriminator was right
        # If image was not satisfactory, and discrim correctly identified it as not satisfactory, reward discrim, punish gen



    
