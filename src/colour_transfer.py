import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

from skimage import io, color
from skimage.filters import threshold_mean
from sklearn.cluster import KMeans

# Load images
# image_1 = io.imread("./src/nude-woman-naked-face-and-nude-woman-profile-1906.jpg")
image_1 = io.imread("./src/AYE.jpg")
image_1_norm = np.array(image_1, dtype=np.float64) / 255
# image_2 = io.imread("./src/the-girls-of-avignon-1907.jpg")
image_2 = io.imread("./src/ELL.jpeg")
image_2_norm = np.array(image_2, dtype=np.float64) / 255


# How many colours I want to cluster
nr_clusters = 10

def cluster_colours(image, nr_clusters = 5):
    """ Clusters the colours of an image

    Return values:
    - labels: the list of cluster labels for each pixel
    - colors: list of average colours for each cluster
    """

    width, height, color_size = image.shape
    image_features = np.reshape(image, (width*height, color_size))

    kmeans = KMeans(n_clusters=nr_clusters, random_state=0).fit(image_features)
    labels = kmeans.labels_
    colors = kmeans.cluster_centers_*255

    return labels, colors


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


# Clustering the colours of the two pictures
labels_1, colors_1 = cluster_colours(image_1_norm, nr_clusters = nr_clusters)
labels_2, colors_2 = cluster_colours(image_2_norm, nr_clusters = nr_clusters)

# Calculating the relative frequency of the cluster colours in each picture
frequency_1 = [(sum(labels_1 == i)/len(labels_1)) for i in range(nr_clusters)]
frequency_2 = [(sum(labels_2 == i)/len(labels_2)) for i in range(nr_clusters)]

# Sorting the frequencies and
frequency_1_sort = frequency_1.copy()
id_sort_1 = np.zeros(nr_clusters)
idx = int(0)
for _ in range(nr_clusters):
    f = np.argmax(frequency_1_sort)
    frequency_1_sort[f] = 0
    id_sort_1[f] = idx
    idx += 1

frequency_2_sort = frequency_2.copy()
id_sort_2 = np.zeros(nr_clusters)
idx = int(0)
for _ in range(nr_clusters):
    f = np.argmax(frequency_2_sort)
    frequency_2_sort[f] = 0
    id_sort_2[f] = idx
    idx += 1


# Matching the colours of one picture with the corresponding colour of the other picture by comparing the frequencies
colors_2_in_1 = []
for id_1 in id_sort_1:
    colors_2_in_1.append(colors_2[id_sort_2 == id_1])

colors_1_in_2 = []
for id_2 in id_sort_2:
    colors_1_in_2.append(colors_1[id_sort_1 == id_2])


# -----------------------------------------------------------------------
# Plotting
fig, ax = plt.subplots(3,2)

image_1_cstr = quantise_image(image_1_norm, labels_1, colors_1)
image_2_cstr = quantise_image(image_2_norm, labels_2, colors_2)

image_1_like_2 = quantise_image(image_1_norm, labels_1, colors_2_in_1)
image_2_like_1 = quantise_image(image_2_norm, labels_2, colors_1_in_2)

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