import numpy as np
from multiprocessing import Pool, cpu_count
from threading import Thread
from queue import Queue
from skimage import io
from src import config as cfg
import matplotlib.pyplot as plt
import os
from src import image_processing_utils as f
import time
import logging
from functools import reduce, partial
from skimage.util import img_as_ubyte


class ImageProcessor:
    def __init__(self, workers):
        self.cpu_workers = workers
        self.pool = Pool(processes=workers)  # Pool is now instantiated once
        self.black_pixels_count = None

    @staticmethod
    def divide_image_into_strips(image, cpu_workers):
        sub_img_height = image.shape[0] // cpu_workers
        sub_images = [(image[i * sub_img_height:(i + 1) * sub_img_height, :]) for i in range(cpu_workers)]
        if image.shape[0] % cpu_workers != 0:
            sub_images.append(image[cpu_workers * sub_img_height:, :])
        return sub_images

    def process_image(self, img, processing_function):
        sub_images = self.divide_image_into_strips(img, self.cpu_workers)
        func_with_black_pixels_count = partial(processing_function, var=self.black_pixels_count)
        results = self.pool.map(func_with_black_pixels_count, sub_images)
        return np.concatenate(results, axis=0)

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

    def show_image(self, img):
        plt.imshow(img, cmap='gray')
        plt.title("Processed Image")
        plt.show()

    def close_pool(self):
        self.pool.close()


class ImageProcessingPipeline:
    def __init__(self, data_dir, process_functions, batch_size, cpu_workers_count, results_dir=cfg.RESULTS_DIR):
        self.data_dir = data_dir
        self.process_functions = process_functions
        self.batch_size = batch_size
        self.cpu_workers_count = cpu_workers_count
        self.processor = ImageProcessor(cpu_workers_count)
        self.results_dir = results_dir

        self.batch_times = []
        self.total_images_processed = 0
        self.start_time = None
        self.end_time = None

    def process_all_images(self):
        self.start_time = time.time()

        images = os.listdir(self.data_dir)
        print(f"Total images to process: {len(images)}")
        processed_images = []

        for i in range(0, len(images), self.batch_size):
            batch_start_time = time.time()
            print(f"Processing batch {i // self.batch_size + 1}")

            batch = images[i:i + self.batch_size]
            batch_processed_images = self.process_batch(batch)
            processed_images.extend(batch_processed_images)
            print(f"Processed {len(batch_processed_images)} images")

            batch_end_time = time.time()
            self.batch_times.append(batch_end_time - batch_start_time)
            self.total_images_processed += len(batch_processed_images)

        self.end_time = time.time()

        self.processor.close_pool()
        return processed_images

    def process_batch(self, batch):
        batch_processed_images = []
        for img_name in batch:
            img_path = os.path.join(self.data_dir, img_name)
            img = io.imread(img_path)
            processed_img = reduce(lambda x, func: self.processor.__getattribute__(func)(x), self.process_functions, img)
            batch_processed_images.append(processed_img)
            self.write_image(processed_img, img_name)
            break

        return batch_processed_images

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
        print(f"Average time per batch: {np.mean(self.batch_times):.2f} seconds")


if __name__ == '__main__':
    pipeline = ImageProcessingPipeline(cfg.DATA_DIR,
                                       ['apply_gaussian_blur', 'apply_grayscale', 'apply_gaussian_noise'],
                                       batch_size=250,
                                       cpu_workers_count=6
                                       )

    processed_images = pipeline.process_all_images()
    pipeline.display_metrics()

