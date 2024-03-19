import numpy as np
from multiprocessing import Pool, cpu_count
from src import config as cfg
from skimage import io, filters
import os
from src import funcs as f
import matplotlib.pyplot as plt

def divide_image_into_strips(image, thread_count):
    sub_img_height = image.shape[0] // thread_count
    sub_images = [(image[i * sub_img_height:(i + 1) * sub_img_height, :]) for i in range(thread_count)]

    # Check if the last part of the image is not None, because the division by threads
    # can leave a remainder
    if image.shape[0] % thread_count != 0:
        sub_images.append(image[thread_count * sub_img_height:, :])

    return sub_images

def gaussian_blur(img):
    # Divide the image into strips for each CPU
    # For testing use only 6 cores (12 threads) from 8 (16 threads) available
    thread_count = 12

    sub_images = divide_image_into_strips(img, thread_count)

    # Process each strip in parallel
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(f.apply_gaussian_blur_to_sub_image, sub_images)

    full_image = np.concatenate(results, axis=0)

    print(f"Thread count: {thread_count}\nImg shape: {img.shape}\nSub Images count: {len(sub_images)}\nFull Img shape: {full_image.shape}")
    return full_image

def grayscale(img):
    thread_count = 12

    sub_images = divide_image_into_strips(img, thread_count)

    with Pool(processes=cpu_count()) as pool:
        results = pool.map(f.apply_conversion_to_grayscale, sub_images)

    full_image = np.concatenate(results, axis=0)
    return full_image

def noise(img):
    thread_count = 12

    sub_images = divide_image_into_strips(img, thread_count)

    with Pool(processes=cpu_count()) as pool:
        results = pool.map(f.apply_gaussian_noise_to_sub_image, sub_images)

    full_image = np.concatenate(results, axis=0)
    return full_image


if __name__ == '__main__':
    images = os.listdir(cfg.DATA_DIR)
    test_image = io.imread(os.path.join(cfg.DATA_DIR, images[0]))
    new_image_blur = gaussian_blur(test_image)
    new_image_grayscale = grayscale(test_image)
    new_image_noise = noise(test_image)

    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    ax = axes.ravel()

    ax[0].imshow(test_image)
    ax[0].set_title("Original")
    ax[1].imshow(new_image_blur, cmap=plt.cm.gray)
    ax[1].set_title("Gaussian Blur")
    ax[2].imshow(new_image_grayscale, cmap=plt.cm.gray)
    ax[2].set_title("Grayscale")
    ax[3].imshow(new_image_noise, cmap=plt.cm.gray)
    ax[3].set_title("Gaussian Noise")

    fig.tight_layout()
    plt.show()
