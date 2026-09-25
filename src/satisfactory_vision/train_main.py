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

def discriminator(img) -> int:
    pass

def satisfactory_to_rl(img): # Outputs image
    pass
def rl_to_satisfactory(img): # Outputs image
    pass

def calculate_mean_squared_error(img1, img2):
    """
    Calculate the mean squared error between two images.
    """
    return np.mean((img1 - img2) ** 2)

# Rough format
imgs = ["path", "path"]
for img in imgs:
    img = cv2.imread(img)
    satis_img = satisfactory_to_rl(img)
    satis_img_noisy = add_gaussian_noise(img)
    rl_img = rl_to_satisfactory(satis_img_noisy)

    # If they differ too much, SMITE them
    mse = calculate_mean_squared_error(satis_img, rl_img) #MSE = mean squared error
    if mse > someNumber:
        # Punish the generator
        pass
    if mse < someNumber:
        # Reward the generator
        pass
    

    discrim_output_int = discriminator(img)
    discrim_output_bool = True if discrim_output >= 0.5 else False

    imgIsReal = imageWasSatisfactory() # This is a function that checks if the image was satisfactory or not
    if discrim_output_bool == False && imgIsReal: # See if discriminator was right
        # If image was satisfactory, and discrim correctly identified reward generator, punish discrim
    elif discrim_output_bool == True && not imgIsReal: # See if discriminator was right
        # If image was not satisfactory, but discrim incorrectly identified it as satisfactory, punish discrim
    elif discrim_output_bool == True && imgIsReal: # See if discriminator was right
        # If image was satisfactory, and discrim correctly identified reward generator, reward discrim, punish gen
    elif discrim_output_bool == False && not imgIsReal: # See if discriminator was right
        # If image was not satisfactory, and discrim correctly identified it as not satisfactory, reward discrim, punish gen



    
