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
    """
    A class that processes image batches by dividing images into strips and applying transformations in parallel.
    """
    def __init__(self, workers):
        """
        Initializes the ImageProcessor with a specific number of worker processes.
        
        Parameters:
        workers (int): Number of worker processes to use for image processing tasks.
        """
        self.cpu_workers = workers
        self.pool = Pool(processes=workers)  # Pool is now instantiated once
        self.black_pixels_count = None

    @staticmethod  
    def divide_image_into_strips(image, cpu_workers):
        """
        Divides an image into horizontal strips for parallel processing.
        
        :param image: The image to be divided.
        :param cpu_workers: The number of workers, dictating the number of strips.
        :return: A list of sub-images (strips).
        """
        #Determine height of each strip based on the number of workers
        sub_img_height = image.shape[0] // cpu_workers
        #Create sub-images for each worker
        sub_images = [(image[i * sub_img_height:(i + 1) * sub_img_height, :]) for i in range(cpu_workers)]
        #Handle any remainder of the image to ensure the entire image is covered
        if image.shape[0] % cpu_workers != 0:
            sub_images.append(image[cpu_workers * sub_img_height:, :])
        return sub_images

    def process_image(self, img, processing_function):
        """
        Processes an image by dividing it into strips and applying a given processing function in parallel.
        
        Parameters:
        img (numpy.ndarray): The image to process.
        processing_function (function): The function to apply to each strip.

        Returns: 
        img (numpy.ndarray): The processed image, reassembled from the processed strips.
        """
        sub_images = self.divide_image_into_strips(img, self.cpu_workers)
        func_with_black_pixels_count = partial(processing_function, var=self.black_pixels_count)
        #Process each strip in parallel.
        results = self.pool.map(func_with_black_pixels_count, sub_images)
        #Reassemble the processed strips into a full image.
        return np.concatenate(results, axis=0)

    def apply_gaussian_blur(self, img):
        """
        Applies a Gaussian blur to an image.
        
        Parameters:
        img (numpy.ndarray): The image to be blurred.
        
        Returns:
        numpy.ndarray: The blurred image.
        """
        return self.process_image(img, f.apply_gaussian_blur_to_sub_image)

    def apply_grayscale(self, img):
        """
        Convert the provided image to grayscale and count the number of black pixels.
        
        Parameters:
        img (numpy.ndarray): The image to turn grayscale.
        
        Returns:
        grayscale_img (numpy.ndarray): The grayscale version of the image.
        """
        grayscale_img = self.process_image(img, f.apply_conversion_to_grayscale)
        self.black_pixels_count = np.sum(grayscale_img < 0.1)  # black pixel sum
        return grayscale_img

    def apply_gaussian_noise(self, img):
        """
        Applies Gaussian noise to an image. If the black pixels count hasn't been set, it applies
        grayscale conversion first to determine that count.
        
        Parameters:
        img (numpy.ndarray): The image to have noise added.
        
        Returns:
        numpy.ndarray: The image with Gaussian noise added.
        """
        if self.black_pixels_count is None:
            self.apply_grayscale(img)
        noise_img = self.process_image(img, f.apply_gaussian_noise_to_sub_image)
        self.black_pixels_count = None
        return noise_img

    def close_pool(self):
        """
        Closes the processing pool.
        """
        self.pool.close()


class ImageProcessingPipeline:
    """
    A pipeline for processing images in batches using parallel processing, 
    applying a series of specified image
    processing functions to each image.
    """
    def __init__(self, data_dir, process_functions, batch_size, cpu_workers_count, results_dir=cfg.RESULTS_DIR):
        """
        Initializes the image processing pipeline.
        
        Parameters:
        data_dir (str): The directory containing the set of images to be processed.
        process_functions (list of str): Names of the processing functions to be applied.
        batch_size (int): The number of images to process in each batch.
        cpu_workers_count (int): The number of worker processes to use for image processing.
        results_dir (str): The directory where the processed images will be saved.
        """
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
        """
        Processes all images in the input directory, applying all specified functions.
        It times the processing and prints out the total number of images to process.
        """
        #Save the time at the beginning of the processing
        self.start_time = time.time()
        #Get a list of all the images in the directory
        images = os.listdir(self.data_dir)
        print(f"Total images to process: {len(images)}")
        processed_images = []

        for i in range(0, len(images), self.batch_size):
            #Save the time at the beginning of batch processing
            batch_start_time = time.time()
            # print(f"Processing batch {i // self.batch_size + 1}")
            batch = images[i:i + self.batch_size]
            batch_processed_images = self.process_batch(batch)
            processed_images.extend(batch_processed_images)
            # print(f"Processed {len(batch_processed_images)} images")

            #Save the time at the end of batch processing
            batch_end_time = time.time()
            self.batch_times.append(batch_end_time - batch_start_time)
            self.total_images_processed += len(batch_processed_images)
        #Save the end time of the processing
        self.end_time = time.time()

        self.processor.close_pool()
        return processed_images

    def process_batch(self, batch):
        """
        Processes a batch of images, applying the specified processing functions to each image.
        
        Parameters:
        batch (list of str): A list of image filenames to be processed.

        Returns:
        batch_processed_images (list of numpy.ndarray): A list of processed images.
        """
        batch_processed_images = []
        for img_name in batch:
            img_path = os.path.join(self.data_dir, img_name)
            img = io.imread(img_path)
            processed_img = reduce(lambda x, func: self.processor.__getattribute__(func)(x), self.process_functions, img)
            batch_processed_images.append(processed_img)
            self.write_image(processed_img, img_name)

        return batch_processed_images

    def write_image(self, img, img_name):
        """
        Save the processed image into the specified directory.
        
        Parameters:
        img (numpy.ndarray): The processed image to be saved.
        img_name (str): The name of the file for the saved image.
        """
        #Get a file path for the output image
        result_path = os.path.join(self.results_dir, img_name)
        if img.dtype != np.uint8:
            img_to_save = img_as_ubyte(img)
        else:
            img_to_save = img
        #Save the image into the directory
        io.imsave(result_path, img_to_save)

    def display_metrics(self):
        """
        Display processing time metrics after all images have been processed.
        """
        #Calculate the amount of time it took to process all of the images
        total_time = self.end_time - self.start_time
        print(f"Total execution time: {total_time:.2f} seconds")
        print(f"Images per second: {self.total_images_processed / total_time:.2f}")
        print(
            f"Image processing techniques per second: {len(self.process_functions) * self.total_images_processed / total_time:.2f}")
        print(f"Average time per batch: {np.mean(self.batch_times):.2f} seconds")


if __name__ == '__main__':
    #Create an instance of the class ImageProcessor:
    pipeline = ImageProcessingPipeline(cfg.DATA_DIR,
                                       ['apply_gaussian_blur', 'apply_grayscale', 'apply_gaussian_noise'],
                                       batch_size=25,
                                       cpu_workers_count=2
                                       )
    #Calls upon the processor functions
    processed_images = pipeline.process_all_images()
    pipeline.display_metrics()

