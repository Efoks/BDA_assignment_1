from skimage import io, filters, util
import numpy as np

def apply_gaussian_blur_to_sub_image(sub_image):
    result = filters.gaussian(sub_image, sigma=2, channel_axis=-1)
    return result

def apply_conversion_to_grayscale(sub_image):
    grayscale_sub_image = np.dot(sub_image[..., :3], [0.299, 0.587, 0.114])
    return grayscale_sub_image

def apply_gaussian_noise_to_sub_image(sub_image):
    result = util.random_noise(sub_image, mode='gaussian', var=0.01)
    return result
