#! /usr/bin/python3

from concurrent.futures import ProcessPoolExecutor
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from skimage.filters import gaussian
from sklearn.cluster import KMeans

COLOR_NORMALISATION_CONSTANT = 255


class ColorTransfer:

    def __init__(self, nr_clusters: int):
        """Class that handles the IO, processing and plotting

        Args:
            nr_clusters: number of colors that will be grouped
        """

        self.nr_clusters = nr_clusters


    def quantise_image(self, image: np.ndarray, labels: np.ndarray, colors: np.ndarray) -> np.ndarray:
        """Substitutes the colors of an image with the color of the cluster it belongs to
        
        Args:
            image: image to be quantised
            labels: color-cluster id to which each pixel belongs
            colors: RGB code of the cluster

        Returns:
            quantised picture (np.ndarray): image that has only the colors of the clusters
        """

        width,height,col = image.shape

        # Initialising new image as matrix
        new_image = np.zeros((width,height,col))
        idx = 0

        # Iterating over the pixels. For each pixel we substitute the value of the cluster centre
        for w in range(width):
            for h in range(height):
                
                new_image[w,h] = colors[labels[idx]]
                idx += 1

        return new_image/COLOR_NORMALISATION_CONSTANT


    def clusterise_image(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Clusters the colours of an image

        Args:
            image: image whose colours have to be grouped

        Returns:
            labels: list of cluster labels for each pixel
            colors: RGB colour code for each cluster
            color_frequency: list with the relative frequencies of each colour
        """

        width, height, color_size = image.shape
        image_features = np.reshape(image, (width*height, color_size))

        # Clustering the colours of the two pictures
        kmeans = KMeans(n_clusters=self.nr_clusters, random_state=0).fit(image_features)
        labels = kmeans.labels_
        colors = kmeans.cluster_centers_*COLOR_NORMALISATION_CONSTANT


        # Calculating the relative frequency of the cluster colours in each picture
        frequency = [(sum(labels == i)/len(labels)) for i in range(self.nr_clusters)]


        # Sorting the frequencies and
        frequency_sort = frequency.copy()
        color_frequency = np.zeros(self.nr_clusters)
        idx = int(0)
        for _ in range(self.nr_clusters):
            f = np.argmax(frequency_sort)
            frequency_sort[f] = 0
            color_frequency[f] = idx
            idx += 1

        return labels, colors, color_frequency


    def read_images(self, path_image_1: str, path_image_2: str):
        """Read images from paths and add to arguments

        Args:
            path_image_1: path of the first image
            path_image_2: path of the second image
        """

        self.image_1, self.image_1_norm = self.read_image(path_image_1)
        self.image_2, self.image_2_norm = self.read_image(path_image_2)


    def read_image(self, path_image: str) -> Tuple[np.ndarray, np.ndarray]:
        """Read single image from path

        Args:
            path_image: path to image

        Returns:
            image: picture with colors encoded in range [0,255]
            image_norm: picture with colors encoded in range [0,1]
        """

        image = io.imread(path_image)
        image_norm = np.array(image, dtype=np.float64) / COLOR_NORMALISATION_CONSTANT
        return image, image_norm


    def process_images(self) -> Tuple[np.ndarray, np.ndarray]:
        """Groups and swaps the colors of two images simulatneously

        Returns:
            image_1_like_2: image 1 with the colours of image 2
            image_2_like_1: image 2 with the colours of image 1
        """

        # The two images are processed in paralled
        with ProcessPoolExecutor() as executor:
            e1 = executor.submit(self.clusterise_image, self.image_1_norm)
            e2 = executor.submit(self.clusterise_image, self.image_2_norm)
            self.labels_1, self.colors_1, color_frequency_1 = e1.result()
            self.labels_2, self.colors_2, color_frequency_2 = e2.result()


        # Matching the colours of one picture with the corresponding colour of the other picture by comparing the frequencies
        self.colors_2_in_1 = []
        for id_1 in color_frequency_1:
            self.colors_2_in_1.append(self.colors_2[color_frequency_2 == id_1])

        self.colors_1_in_2 = []
        for id_2 in color_frequency_2:
            self.colors_1_in_2.append(self.colors_1[color_frequency_1 == id_2])

        image_1_like_2 = gaussian(self.quantise_image(self.image_1_norm, self.labels_1, self.colors_2_in_1), sigma = 2, multichannel=True)
        image_2_like_1 = gaussian(self.quantise_image(self.image_2_norm, self.labels_2, self.colors_1_in_2), sigma = 2, multichannel=True)

        return image_1_like_2, image_2_like_1


    def plot_image(self, image_1_like_2: np.ndarray, image_2_like_1: np.ndarray):
        """Plots the processed images and the originals
        """

        fig, ax = plt.subplots(3,2)

        # Smooth clustered image
        image_1_clustered = gaussian(self.quantise_image(self.image_1_norm, self.labels_1, self.colors_1), sigma = 2, multichannel=True)
        image_2_clustered = gaussian(self.quantise_image(self.image_2_norm, self.labels_2, self.colors_2), sigma = 2, multichannel=True)

        # Original picture
        ax[0,0].imshow(self.image_1)
        ax[0,1].imshow(self.image_2)
        ax[0,0].axis("off")
        ax[0,1].axis("off")
        ax[0,0].set_title("Original")
        ax[0,1].set_title("Original")

        # Original picture with quantised colours
        ax[1,0].imshow(image_1_clustered)
        ax[1,1].imshow(image_2_clustered)
        ax[1,0].axis("off")
        ax[1,1].axis("off")
        ax[1,0].set_title("Colour quantisation")
        ax[1,1].set_title("Colour quantisation")

        # Picture with swapped colours
        ax[2,0].imshow(image_1_like_2)
        ax[2,1].imshow(image_2_like_1)
        ax[2,0].axis("off")
        ax[2,1].axis("off")
        ax[2,0].set_title("Colour transfer")
        ax[2,1].set_title("Colour transfer")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    color_transfer = ColorTransfer(nr_clusters = 10)
    color_transfer.read_images(path_image_1 = "src/kiss_klimt.jpg", path_image_2 = "src/lake.jpg")
    image_1_like_2, image_2_like_1 = color_transfer.process_images()
    color_transfer.plot_image(image_1_like_2, image_2_like_1)
