import cupy as cp
from cupyx.scipy.ndimage import convolve
import os
from skimage import io
from src import config as cfg
import matplotlib.pyplot as plt


def create_gaussian_kernel(size, sigma):
    """Create a 2D Gaussian kernel with CuPy."""
    ax = cp.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    xx, yy = cp.meshgrid(ax, ax)
    kernel = cp.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / cp.sum(kernel)


def apply_gaussian_blur_gpu_rgb(image, kernel_size=5, sigma=1):
    """Apply Gaussian blur to an RGB image using CuPy."""
    if not isinstance(image, cp.ndarray):
        image = cp.asarray(image)

    # Create the Gaussian kernel
    kernel = create_gaussian_kernel(kernel_size, sigma)

    # Initialize an empty array for the blurred image
    blurred_image = cp.zeros_like(image)

    # Apply the convolution separately for each channel
    for i in range(3):  # Assuming the last dimension is the channel
        blurred_image[:, :, i] = convolve(image[:, :, i], kernel, mode='reflect')

    return blurred_image


if __name__ == '__main__':
    images = os.listdir(cfg.DATA_DIR)
    test_image = io.imread(os.path.join(cfg.DATA_DIR, images[0]))

    image_gpu = cp.asarray(test_image)
    blurred_image = apply_gaussian_blur_gpu_rgb(image_gpu, kernel_size=5, sigma=2)
    blurred_image = cp.asnumpy(blurred_image)

    fig, axes = plt.subplots(1,
                             2,
                             figsize=(10, 6))
    ax = axes.ravel()

    ax[0].imshow(test_image)
    ax[0].set_title("Original")
    ax[1].imshow(blurred_image, cmap=plt.cm.gray)
    ax[1].set_title("Gaussian Blur")

    fig.tight_layout()
    plt.show()