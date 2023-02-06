import numpy as np
import cv2

def crop_square(image):
    width = image.shape[1]
    height = image.shape[0]
    new_dim = np.floor(min(width, height) / 32) * 32
    height_start = int(height / 2 - new_dim / 2)
    width_start = int(width / 2 - new_dim / 2)
    height_end = int(height_start + new_dim)
    width_end = int(width_start + new_dim)
    new_img = image[height_start:height_end, width_start:width_end,:]
    return new_img
    
def downsizer (image):
    cropped = crop_square(image)
    down_sampled = cv2.resize(cropped,(32,32),interpolation=cv2.INTER_CUBIC)
    return down_sampled
    