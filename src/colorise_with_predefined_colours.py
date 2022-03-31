from colour_transfer import ColorTransfer
import numpy as np
import json
import matplotlib.pyplot as plt
import argparse
import colorsys
from scipy.optimize import linear_sum_assignment

DISTANCE_COLORS_MAX = 1E9
NR_COMBINATIONS = 348
INDEX_HUE = 0
INDEX_SATURATION = 1
INDICES_HS = [INDEX_HUE, INDEX_SATURATION]

def sort_colors_local_min(colors_original: list, colors_new: list, distance_function) -> list:
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
        colors_new.pop(color_index_to_delete)

    return colors_new_sorted


def sort_colors_global_min(colors_original: list, colors_new: list, distance_function) -> list:
    """
    Sort colors by method defined in distance_function.
    
    Finds matching colors to the new ones by using the hungarian algorithm.
    First a distance matrix between color pairs is built up. The algorithm then finds the matches that
    generate the minimum summed distance cost.

    Both the original and the new colors are defined as np.ndarrays of size [3 x n].
    These contain, for n colors, their encoding as three rgb values.

    Args:
        colors_original: colors of the original image
        colors_new: new colors that have to be applied
        distance_function: function that calculates the distance between two colors values

    Returns:
        colors_new_sorted: the new color values
    """

    shape = len(colors_original)
    distance_between_colors = np.zeros(shape = (shape, shape))

    colors_new_sorted = []
    for i, color_original in enumerate(colors_original):
        
        for j, color_new in enumerate(colors_new):

            distance_between_colors[i,j] = distance_function(color_new[INDICES_HS] - color_original[INDICES_HS])

    _, rows = linear_sum_assignment(distance_between_colors)

    colors_new = np.array(colors_new)
    colors_new_sorted = colors_new[rows]

    return colors_new_sorted


def colorize_image_all_combinations():

    nr_color_combinations = args.nr_colors
    input_path = args.input_path

    # Function that calculates the distance between two colors
    distance_function = np.linalg.norm

    # Function that sorts the colors
    if args.sort_method == "local":
        sort_colors_function = sort_colors_local_min
    elif args.sort_method == "global":
        sort_colors_function = sort_colors_global_min

    coloriser = ColorTransfer(nr_clusters=nr_color_combinations)

    image, image_norm = coloriser.read_image(path_image = input_path)

    with open("/home/giovanni/PyProjects/dictionary-of-colour-combinations/colors.json") as jsonFile:
        jsonObject = json.load(jsonFile)
        jsonFile.close()


    labels, colors_original, color_frequency_original = coloriser.clusterise_image(image/255)

    for combination in range(NR_COMBINATIONS):
        colors = []
        for color in jsonObject:
            if combination in color["combinations"]:
                colors.append(np.array(color["rgb"]))
        
        if len(colors) < nr_color_combinations: continue

        if args.color_method == "hsv":
            colors = [np.array(colorsys.rgb_to_hsv(*color)) for color in colors]
            colors_original = [np.array(colorsys.rgb_to_hsv(*color)) for color in colors_original]

        # Sort colors by method
        colors_new = sort_colors_function(colors_original, colors, distance_function)

        if args.color_method == "hsv":
            colors_new = [np.array(colorsys.hsv_to_rgb(*color)) for color in colors_new]

        image_quantized = coloriser.quantise_image(image_norm, labels, colors_new)

        plt.imshow(image_quantized)
        plt.show()


def parse_arguments():
    parser = argparse.ArgumentParser("Parser for ")
    parser.add_argument("-n", "--nr_colors", help="How many colors have to be clustered.", type=int, default=4)
    parser.add_argument("-i", "--input_path", help="Path to input image", type=str, default="/home/giovanni/Downloads/giovanno.jpeg")
    parser.add_argument("-o", "--output_path", help="Path to save path", type=str)
    parser.add_argument("-c", "--color_method", help="Whether to use HSV or RGB", type=str, default="hsv")
    parser.add_argument("-s", "--sort_method", help="What method to use for sorting colors [local, global]", type=str, default="global")
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = parse_arguments()
    colorize_image_all_combinations()