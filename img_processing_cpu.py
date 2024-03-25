import numpy as np
from multiprocessing import Pool, cpu_count
from skimage import io
from src import config as cfg
import matplotlib.pyplot as plt
import os
from src import image_processing_utils as f
import time


class ImageProcessor:
    def __init__(self, img, thread_count = 12):
        self.image = img
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
        with Pool(processes=thread_count) as pool:
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


class BatchLoader:
    def __init__(self, data_dir, batch_size, thread_count=12):

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_paths = self._load_image_paths()
        self.current_index = 0
        self.thread_count = thread_count

    def _load_image_paths(self):
        return [os.path.join(self.data_dir, image) for image in os.listdir(self.data_dir)]

    def __len__(self):
        return len(self.image_paths)

    def __iter__(self):
        self.current_index = 0
        return self

    def __next__(self):
        if self.current_index >= len(self.image_paths):
            raise StopIteration

        start_idx = self.current_index
        self.current_index += self.batch_size
        end_idx = min(self.current_index, len(self.image_paths))
        bathc_image_paths = self.image_paths[start_idx:end_idx]

        with Pool(processes=self.thread_count) as pool:
            batch_images = pool.map(io.imread, bathc_image_paths)

        return batch_images

    def __getitem__(self, idx):
        return io.imread(self.image_paths[idx])


class ImageProcessingPipeline:
    def __init__(self, data_dir, process_functions, batch_size = 4, thread_count=12):

        data_loader_threads = int(thread_count * 0.25)
        self.img_processing_threads = 12 - data_loader_threads

        self.batch_loader = BatchLoader(data_dir, batch_size, data_loader_threads)
        self.process_functions = process_functions

        self.batch_times = []
        self.total_images_processed = 0
        self.start_time = None
        self.end_time = None

    def process_all_images(self):
        self.start_time = time.time()
        processed_images = []

        for batch in self.batch_loader:
            batch_start_time = time.time()
            preprocessed_images = [ImageProcessor(img, self.img_processing_threads) for img in batch]
            processed_images = [img.apply_gaussian_blur().image for img in preprocessed_images]
            batch_end_time = time.time()
            self.batch_times.append(batch_end_time - batch_start_time)
            self.total_images_processed += len(batch)

        self.end_time = time.time()
        return processed_images

    def display_metrics(self):
        total_time = self.end_time - self.start_time
        print(f"Total execution time: {total_time} seconds")
        if total_time > 0:
            print(f"Images per second: {self.total_images_processed / total_time}")
            print(f"Image processing techniques per second: {len(self.process_functions) * self.total_images_processed / total_time}")
        for i, batch_time in enumerate(self.batch_times):
            print(f"Time taken for batch {i+1}: {batch_time} seconds")

def plot_images(images, rows = 2, cols = 2):
    fig, axes = plt.subplots(rows, cols)
    axes = axes.flatten()

    for images, ax in zip(images, axes):
        ax.imshow(images, cmap='gray')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    images = os.listdir(cfg.DATA_DIR)
    pipeline = ImageProcessingPipeline(cfg.DATA_DIR,
                                       [f.apply_gaussian_blur_to_sub_image,
                                                       f.apply_conversion_to_grayscale,
                                                       f.apply_gaussian_noise_to_sub_image]
                                       )
    processed_images = pipeline.process_all_images()
    pipeline.display_metrics()
    # plot_images(processed_images)
