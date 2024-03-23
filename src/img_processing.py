import numpy as np
from multiprocessing import Pool, cpu_count
from skimage import io
import config as cfg
import matplotlib.pyplot as plt
import os
import image_processing_utils as f

class ImageProcessor:
    def __init__(self, image_path, thread_count = 12):
        self.image = io.imread(image_path)
        self.thread_count = thread_count

    @staticmethod
    def divide_image_into_strips(image, thread_count):
        sub_img_height = image.shape[0] // thread_count
        sub_images = [(image[i * sub_img_height:(i + 1) * sub_img_height, :]) for i in range(thread_count)]
        if image.shape[0] % thread_count != 0:
            sub_images.append(image[thread_count * sub_img_height:, :])
        return sub_images

    @staticmethod
    def process_image(img, processing_function, thread_count):
        sub_images = ImageProcessor.divide_image_into_strips(img, thread_count)
        with Pool(processes=cpu_count()) as pool:
            results = pool.map(processing_function, sub_images)
        return np.concatenate(results, axis=0)

    def apply_gaussian_blur(self):
        self.image = self.process_image(self.image, f.apply_gaussian_blur_to_sub_image, self.thread_count)
        return self

    def apply_grayscale(self):
        self.image = self.process_image(self.image, f.apply_conversion_to_grayscale, self.thread_count)
        return self

    def apply_gaussian_noise(self):
        self.image = self.process_image(self.image, f.apply_gaussian_noise_to_sub_image, self.thread_count)
        return self

    def show_image(self):
        plt.imshow(self.image, cmap='gray')
        plt.title("Processed Image")
        plt.show()

if __name__ == '__main__':
    images = os.listdir(cfg.DATA_DIR)
    processor = ImageProcessor(os.path.join(cfg.DATA_DIR, images[0]))
    processor.apply_gaussian_blur().apply_grayscale().apply_gaussian_noise().show_image()
