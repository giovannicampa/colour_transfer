import numpy as np
import matplotlib.pyplot as plt

from skimage import io, color
from skimage.filters import threshold_mean
from sklearn.cluster import KMeans

image_1 = io.imread("./src/nude-woman-naked-face-and-nude-woman-profile-1906.jpg")
image_1_norm = np.array(image_1, dtype=np.float64) / 255
image_2 = io.imread("./src/the-girls-of-avignon-1907.jpg")
image_2_norm = np.array(image_2, dtype=np.float64) / 255

# Creating a list of the pixels. Each pixel has [r,g,b] as features


nr_clusters = 4

def cluster_colours(image, nr_clusters = 7):
    """ Clusters the colours of an image
    """

    width, height, color_size = image.shape
    image_features = np.reshape(image, (width*height, color_size))

    kmeans = KMeans(n_clusters=nr_clusters, random_state=0).fit(image_features)
    labels = kmeans.labels_
    colors = kmeans.cluster_centers_*255

    return labels, colors


def clusterise_image(image, labels, colors):
    """Substitutes the colors of an image with the average color of the cluster it belongs to
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



labels_1, colors_1 = cluster_colours(image_1_norm)
labels_2, colors_2 = cluster_colours(image_2_norm)


fig, ax = plt.subplots(1,2)

image_1_cstr = clusterise_image(image_1_norm, labels_1, colors_1)
image_2_cstr = clusterise_image(image_2_norm, labels_2, colors_2)


ax[0].imshow(image_1_cstr)
ax[1].imshow(image_2_cstr)
ax[0].axis("off")
ax[1].axis("off")


plt.show()