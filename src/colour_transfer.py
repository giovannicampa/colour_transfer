from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from skimage.filters import gaussian
from sklearn.cluster import KMeans

# How many colours I want to cluster
nr_clusters = 10


# -----------------------------------------------------------------------
# - Functions

def quantise_image(image, labels, colors):
    """Substitutes the colors of an image with the color of the cluster it belongs to
    
    Return value:
    - quantised picture
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

    return new_image/255


def clusterise_image(image, nr_clusters):
    """ Clusters the colours of an image

    Return values:
    - labels: list of cluster labels for each pixel
    - colors: list of average colours for each cluster
    - color_frequency: list with the relative frequencies of each colour
    """

    width, height, color_size = image.shape
    image_features = np.reshape(image, (width*height, color_size))

    # Clustering the colours of the two pictures
    kmeans = KMeans(n_clusters=nr_clusters, random_state=0).fit(image_features)
    labels = kmeans.labels_
    colors = kmeans.cluster_centers_*255


    # Calculating the relative frequency of the cluster colours in each picture
    frequency = [(sum(labels == i)/len(labels)) for i in range(nr_clusters)]


    # Sorting the frequencies and
    frequency_sort = frequency.copy()
    color_frequency = np.zeros(nr_clusters)
    idx = int(0)
    for _ in range(nr_clusters):
        f = np.argmax(frequency_sort)
        frequency_sort[f] = 0
        color_frequency[f] = idx
        idx += 1

    return labels, colors, color_frequency


# -----------------------------------------------------------------------
# - Load images

image_1 = io.imread("./src/kiss_klimt.jpg")
image_1_norm = np.array(image_1, dtype=np.float64) / 255
image_2 = io.imread("./src/lake.jpg")
image_2_norm = np.array(image_2, dtype=np.float64) / 255


# -----------------------------------------------------------------------
# - Processing

# The two images are processed in paralled
with ProcessPoolExecutor() as executor:
    e1 = executor.submit(clusterise_image, image_1_norm, nr_clusters)
    e2 = executor.submit(clusterise_image, image_2_norm, nr_clusters)
    labels_1, colors_1, color_frequency_1 = e1.result()
    labels_2, colors_2, color_frequency_2 = e2.result()


# Matching the colours of one picture with the corresponding colour of the other picture by comparing the frequencies
colors_2_in_1 = []
for id_1 in color_frequency_1:
    colors_2_in_1.append(colors_2[color_frequency_2 == id_1])

colors_1_in_2 = []
for id_2 in color_frequency_2:
    colors_1_in_2.append(colors_1[color_frequency_1 == id_2])


# -----------------------------------------------------------------------
# - Plotting
fig, ax = plt.subplots(3,2)

image_1_cstr = gaussian(quantise_image(image_1_norm, labels_1, colors_1), sigma = 2, multichannel=True)
image_2_cstr = gaussian(quantise_image(image_2_norm, labels_2, colors_2), sigma = 2, multichannel=True)

image_1_like_2 = gaussian(quantise_image(image_1_norm, labels_1, colors_2_in_1), sigma = 2, multichannel=True)
image_2_like_1 = gaussian(quantise_image(image_2_norm, labels_2, colors_1_in_2), sigma = 2, multichannel=True)

# Original picture
ax[0,0].imshow(image_1)
ax[0,1].imshow(image_2)
ax[0,0].axis("off")
ax[0,1].axis("off")
ax[0,0].set_title("Original")
ax[0,1].set_title("Original")

# Original picture with quantised colours
ax[1,0].imshow(image_1_cstr)
ax[1,1].imshow(image_2_cstr)
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
