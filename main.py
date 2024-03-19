import os
from src import config as cfg
import matplotlib.pyplot as plt
from skimage import io, filters
from skimage import color
import numpy as np

def add_noise(image, percentage_of_black=0.1, black_threshold=0.1):
    grayscale = color.rgb2gray(image)
    n_black_pixes = np.sum(grayscale < (black_threshold * 255))
    noise_amount = percentage_of_black * n_black_pixes

    # Normalizing this to get std for Gaussian noise
    noise_std = np.sqrt((noise_amount / image.size) * (255 ** 2))

    new_image = np.copy(image).astype(np.float64)
    for i in range(3):
        new_image[:, :,             i] += np.random.normal(scale=noise_std, size=image.shape[:2])

    new_image = np.clip(new_image, 0, 255).astype(np.uint8)

    return new_image

def main():

    # Read the files in the data directory
    files = os.listdir(cfg.DATA_DIR)

    # Print the file number
    print(len(files))

    # Get the first image
    test_image = io.imread(os.path.join(cfg.DATA_DIR, files[0]))
    # Image conversions that will be used for general testing of parralelization and its results.
    grayscale = color.rgb2gray(test_image)
    blur = filters.gaussian(test_image, sigma=2, channel_axis=-1)
    noise = add_noise(test_image)

    # Plot the images
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    ax = axes.ravel()

    ax[0].imshow(test_image)
    ax[0].set_title("Original")
    ax[1].imshow(grayscale, cmap=plt.cm.gray)
    ax[1].set_title("Grayscale")
    ax[2].imshow(blur)
    ax[2].set_title("Gaussian Blur")
    ax[3].imshow(noise)
    ax[3].set_title("Gaussian Noise")

    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()

