import numpy as np
from skimage import io
import os
from src import config as cfg
import time
from src import image_processing_utils as f
from skimage.util import img_as_ubyte


class SequentialImageProcessor:
    def __init__(self, data_dir, process_functions, results_dir=cfg.RESULTS_DIR):
        self.data_dir = data_dir
        self.process_functions = process_functions
        self.results_dir = results_dir

        self.total_images_processed = 0
        self.start_time = None
        self.end_time = None

        self.black_pixels_count = None

    def process_image(self, img, processing_function):
        return processing_function(img, self.black_pixels_count)

    def apply_gaussian_blur(self, img):
        return self.process_image(img, f.apply_gaussian_blur_to_sub_image)

    def apply_grayscale(self, img):
        grayscale_img = self.process_image(img, f.apply_conversion_to_grayscale)
        self.black_pixels_count = np.sum(grayscale_img < 0.1)  # black pixel sum
        return grayscale_img

    def apply_gaussian_noise(self, img):
        if self.black_pixels_count is None:
            self.apply_grayscale(img)
        noise_img = self.process_image(img, f.apply_gaussian_noise_to_sub_image)
        self.black_pixels_count = None
        return noise_img

    def process_all_images(self):
        self.start_time = time.time()
        images = os.listdir(self.data_dir)
        print(f"Total images to process: {len(images)}")
        for img_name in images:
            img_path = os.path.join(self.data_dir, img_name)
            img = io.imread(img_path)
            for func in self.process_functions:
                img = self.__getattribute__(func)(img)
            self.write_image(img, img_name)
            self.total_images_processed += 1
        self.end_time = time.time()

    def write_image(self, img, img_name):
        result_path = os.path.join(self.results_dir, img_name)
        if img.dtype != np.uint8:
            img_to_save = img_as_ubyte(img)
        else:
            img_to_save = img
        io.imsave(result_path, img_to_save)

    def display_metrics(self):
        total_time = self.end_time - self.start_time
        print(f"Total execution time: {total_time:.2f} seconds")
        print(f"Images per second: {self.total_images_processed / total_time:.2f}")
        print(
            f"Image processing techniques per second: {len(self.process_functions) * self.total_images_processed / total_time:.2f}")

if __name__ == '__main__':
    processor = SequentialImageProcessor(cfg.DATA_DIR,
                                         ['apply_gaussian_blur',
                                          'apply_grayscale',
                                          'apply_gaussian_noise'])

    processor.process_all_images()