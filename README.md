# Colour transfer
This script takes two pictures as input and swaps their **n most prominent** characteristic colours.

The operations done to achieve this are 

1. Definition of the input pictures, and the parameter for the number of colours.
2. Clustering of the colors on each picture with K-Means. It takes as input the color information of all pixels and finds the ones that can be grouped together. 
3. Rank the clusters by size and for each one calculate a representative RGB value.
4. Swap the RGB values of the clusters in the two images. 
5. Smoothen the resulting image with a gaussian filter

![Color swap](https://github.com/giovannicampa/colour_transfer/blob/main/src/color_swap.png)
