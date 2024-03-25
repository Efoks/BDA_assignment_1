import cupy as cp
from cupyx.scipy.ndimage import convolve
import os
from skimage import io
from src import config as cfg
import matplotlib.pyplot as plt
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import logging
from functools import reduce
import numpy as np
from skimage.util import img_as_ubyte

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ImageProcessor:
    def __init__(self, img, kernel_size=2, sigma=0.2):
        """
        Initializes the ImageProcessor with an image and optional kernel size and sigma values.

        Args:
            img (cp.ndarray or any format convertible to cp.ndarray): The image to process.
            kernel_size (int, optional): The size of the Gaussian kernel. Default is 2.
            sigma (float, optional): The sigma value for the Gaussian kernel. Default is 0.2.
        """
        self.image = cp.asarray(img) if not isinstance(img, cp.ndarray) else img
        self.kernel_size = kernel_size
        self.sigma = sigma

        self._black_pixels_count = None
        self._gaussian_kernel = self.create_gaussian_kernel(self.kernel_size, self.sigma)

    def create_gaussian_kernel(self, size, sigma):
        """
        Creates a Gaussian kernel given a size and sigma.

        Args:
            size (int): The size of the kernel.
            sigma (float): The sigma value for the kernel.

        Returns:
            cp.ndarray: The Gaussian kernel as a CuPy array.
        """
        ax = cp.linspace(-(size - 1) / 2., (size - 1) / 2., size)
        xx, yy = cp.meshgrid(ax, ax)
        kernel = cp.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        return kernel / cp.sum(kernel)

    def apply_gaussian_blur(self):
        """
        Applies Gaussian blur to the image.
        Description:
        https://en.wikipedia.org/wiki/Gaussian_blur

        Returns:
            self: The ImageProcessor instance to allow method chaining.
        """
        if self.image.ndim == 2:  # Grayscale image
            self.image = convolve(self.image, self._gaussian_kernel, mode='reflect')
            return self
        elif self.image.ndim == 3:  # Color image
            blurred = cp.zeros_like(self.image)
            for i in range(self.image.shape[2]):  # Apply the filter to each channel
                blurred[:, :, i] = convolve(self.image[:, :, i], self._gaussian_kernel, mode='reflect')
            self.image = blurred
            return self

    def apply_grayscale(self):
        """
        Converts the image to grayscale.
        Description: of the formula used:
        https://en.wikipedia.org/wiki/Luma_(video)gauss

        Returns:
            self: The ImageProcessor instance to allow method chaining.
        """
        weights = cp.array([0.299, 0.587, 0.114], dtype=cp.float64)
        self.image = cp.tensordot(self.image, weights, axes=([-1], [0]))
        self._black_pixels_count = cp.sum(self.image < 10)  # black pixel sum, threshold set to 10 (can be adjusted if needed)
        return self

    def apply_gaussian_noise(self):
        """
        Applies Gaussian noise to the image based on the number of black pixels.
        Description:
        https://en.wikipedia.org/wiki/Gaussian_noise

        Returns:
            self: The ImageProcessor instance to allow method chaining.
        """
        if self._black_pixels_count is None:
            # Making a copy, because need to keep the original image for further processing
            base_image = self.image
            self.apply_grayscale()
            self.image = base_image

        rows = cp.random.randint(0, int(self.image.shape[0]), (int(self._black_pixels_count * 0.1), 3))
        cols = cp.random.randint(0, int(self.image.shape[1]), (int(self._black_pixels_count * 0.1), 3))

        channel_indices = cp.arange(self.image.shape[2])
        noise = cp.random.randint(0, 256, (int(self._black_pixels_count * 0.1), 3), dtype=self.image.dtype)

        self.image[rows, cols, channel_indices] = noise

        return self


class ImageProcessingPipeline:
    def __init__(self, data_dir, results_dir, process_functions, batch_size, kernel_size=2, sigma=0.2, max_workers=6):
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.batch_size = batch_size
        self.process_functions = process_functions
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.max_workers = max_workers
        self.image_paths = [os.path.join(data_dir, image) for image in os.listdir(data_dir)]

        self.start_time = None
        self.end_time = None
        self.batch_times = []

    def load_and_process_batch(self, batch_paths):
        """
        Loads and processes a batch of images as specified by the batch_paths.

        Args:
            batch_paths (list of str): The paths of the images to process in the current batch.

        Returns:
            processed_images (list): A list of processed images (currently not utilized fully).
        """
        start_time = time.time()
        processed_images = []
        for path in batch_paths:
            img = io.imread(path)

            processor = ImageProcessor(img, self.kernel_size, self.sigma)
            processed_img = reduce(lambda x, func: getattr(processor, func)(), self.process_functions, processor)
            self.save_image(cp.asnumpy(processed_img.image), os.path.basename(path))

        self.batch_times.append(time.time() - start_time)
        return processed_images

    def save_image(self, img, img_name):
        """
        Saves an image to the results directory, converting to the correct format as needed.

        Args:
            img (cp.ndarray or np.ndarray): The image to save.
            img_name (str): The name of the image file.
        """
        if isinstance(img, cp.ndarray):
            img = cp.asnumpy(img)

        if img.dtype == np.float32 or img.dtype == np.float64:
            img = (img - img.min()) / (img.max() - img.min())

        result_path = os.path.join(self.results_dir, img_name)
        if img.dtype != np.uint8:
            img_to_save = img_as_ubyte(img)
        else:
            img_to_save = img
        io.imsave(result_path, img_to_save)

    def execute(self):
        """
        Executes the image processing pipeline, processing all images in the data directory in batches.
        """
        self.start_time = time.time()
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.load_and_process_batch, self.image_paths[i:i+self.batch_size]) for i in range(0, len(self.image_paths), self.batch_size)]
            for future in as_completed(futures):
                future.result()
        self.end_time = time.time()

    def display_metrics(self):
        total_time = self.end_time - self.start_time
        logging.info(f"Total execution time: {total_time:.2f} seconds")

        # Can't seem to keep it to work. It always breaks and I don't know why. Possible issues with memory management?
        # logging.info(f'Images per second: {len(self.image_paths) / total_time:.2f}')
        # logging.info(f"Image processing techniques per second: {len(self.process_functions) * len(self.image_paths) / total_time:.2f}")
        # logging.info(f"Average time per batch: {np.mean(self.batch_times):.2f} seconds")

if __name__ == '__main__':

    cp.get_default_memory_pool().free_all_blocks()

    pipeline = ImageProcessingPipeline(cfg.DATA_DIR,
                                       cfg.RESULTS_DIR,
                                       # ['apply_gaussian_blur', 'apply_gaussian_noise', 'apply_grayscale'],
                                       ['apply_gaussian_blur', 'apply_gaussian_noise', 'apply_grayscale'],
                                       batch_size=8000,
                                       kernel_size=5,
                                       sigma=2)
    pipeline.execute()
    pipeline.display_metrics()
    # pipeline.plot_processing_times()
