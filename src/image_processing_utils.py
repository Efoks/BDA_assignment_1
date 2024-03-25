from skimage import io, filters, util
import numpy as np


# Var is a dummy argument used only in noise function
def apply_gaussian_blur_to_sub_image(sub_image, var = None):
    result = filters.gaussian(sub_image, sigma=1, channel_axis=-1)
    return result

def apply_conversion_to_grayscale(sub_image, var = None):
    grayscale_sub_image = np.dot(sub_image[..., :3], [0.299, 0.587, 0.114])
    return grayscale_sub_image

def apply_gaussian_noise_to_sub_image(sub_image, var):
    base_variance = 0.00001
    var_prop = var / np.prod(sub_image.shape)
    variance = base_variance + (var_prop * base_variance)
    result = util.random_noise(sub_image, mode='gaussian', var=variance)
    return result
