from colour_transfer import ColorTransfer
import numpy as np
import json
import matplotlib.pyplot as plt

DISTANCE_COLORS_MAX = 1E9
NR_COMBINATIONS = 348

def sort_colors_local_min(colors_original: np.ndarray, colors_new, distance_function) -> list:
    """
    Sort colors by method defined in distance_function.
    
    Finds a local minimum of color distances.

    Both the original and the new colors are defined as np.ndarrays of size [3 x n].
    These contain, for n colors, their encoding as three rgb values.

    Args:
        colors_original: colors of the original image
        colors_new: new colors that have to be applied
        distance_function: function that calculates the distance between two colors values
    
    Returns:
        colors_new_sorted: the new color values
    """

    colors_new_sorted = []
    for color_original in colors_original:
        distance_between_colors_min =  DISTANCE_COLORS_MAX
        
        for i, color_new in enumerate(colors_new):

            distance_between_colors = distance_function(color_new - color_original)


            if distance_between_colors < distance_between_colors_min:
                distance_between_colors_min = distance_between_colors
                color_match = color_new
                color_index_to_delete = i

        colors_new_sorted.append(color_match)
        colors.pop(color_index_to_delete)

    return colors_new_sorted



# Function that calculates the distance between two colors
distance_function = np.linalg.norm

coloriser = ColorTransfer(nr_clusters=4)

image, image_norm = coloriser.read_image(path_image = "/home/giovanni/Downloads/giovanno.jpeg")

with open("/home/giovanni/PyProjects/dictionary-of-colour-combinations/colors.json") as jsonFile:
    jsonObject = json.load(jsonFile)
    jsonFile.close()


labels, colors_original, color_frequency_original = coloriser.clusterise_image(image/255)

for combination in range(NR_COMBINATIONS):
    colors = []
    for color in jsonObject:
        if combination in color["combinations"]:
            colors.append(np.array(color["rgb"]))
    
    if len(colors) < 4: continue

    # Sort colors by method
    colors_new = sort_colors_local_min(colors_original, colors_new, distance_function)

    image_quantized = coloriser.quantise_image(image_norm, labels, colors_new)

    plt.imshow(image_quantized)
    plt.show()
