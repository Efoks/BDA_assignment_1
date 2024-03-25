import numpy as np 
from skimage import io
import os
from src import config as cfg
import time
from src import image_processing_utils as f
from skimage.util import img_as_ubyte


class SequentialImageProcessor:
    """
    A class that processes images sequentially using a set of provided image processing functions.
    This processor reads images from a data folder, applies each function, and saves the images into the results folder on disc.
    """
    def __init__(self, data_dir, process_functions, results_dir=cfg.RESULTS_DIR):
        """
        Initialization of the class SequentialImageProcessor.
        
        Parameters:
        data_dir (str): The directory containing the set of images to be processed.
        process_functions (list of str): Names of the processing functions to be applied.
        results_dir (str): The directory where the processed images will be saved.
        """

        #Path of the input image directory
        self.data_dir = data_dir
        #List of processing functions to apply
        self.process_functions = process_functions
        #Path to the output image directory
        self.results_dir = results_dir

        #Variables to keep track of the number of processed images and times
        self.total_images_processed = 0
        self.start_time = None
        self.end_time = None

        #Variable to keep track of black pixels in an image
        self.black_pixels_count = None

    def process_image(self, img, processing_function):
        """
        Processes an image using a specified function.
        
        Parameters:
        img (numpy.ndarray): The image to be processed.
        processing_function (function): The function to apply to the image.
        
        Returns:
        numpy.ndarray: The processed image.
        """
        # Call the processing function with the image and the black pixel count variable
        return processing_function(img, self.black_pixels_count)

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
        grayscale_img = self.process_image(img, f.apply_conversion_to_grayscale) #Logic of this roundabout function calling should be improved
        #Counts the number of pixels that reach the "black" benchmark
        self.black_pixels_count = np.sum(grayscale_img < 0.1)  #Black pixel sum
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
        self.black_pixels_count = None #Reset the black pixel count for the next image to be processed
        return noise_img

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
        for img_name in images:
            img_path = os.path.join(self.data_dir, img_name) #Get the path to the image being processed
            img = io.imread(img_path) #COnvert the image into numpy ndarray
            #Apply all the given functions to the image
            for func in self.process_functions:
                img = self.__getattribute__(func)(img)
            #Save the processed image
            self.write_image(img, img_name)
            self.total_images_processed += 1
        #Save the end time of the processing
        self.end_time = time.time()

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

if __name__ == '__main__':
    #Create an instance of the SequentialImageProcessor
    processor = SequentialImageProcessor(cfg.DATA_DIR,
                                         ['apply_gaussian_blur',
                                          'apply_grayscale',
                                          'apply_gaussian_noise'])
    #Calls upon the processor functions
    processor.process_all_images()
    processor.display_metrics()