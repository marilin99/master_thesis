# CODE for bacterium peak detection #

# libraries
import numpy as np
from scipy import ndimage
from scipy.ndimage import *
from skimage.morphology import disk


# peaker function adopted from https://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array/3689710#3689710
def peaker(data,ty):

    """
    Considering if the array is from the red channel or green channel (ty), peaks in the channel thresholded image are detected
    """
    # channel type condition
    if ty == "green":
        neighborhood = disk(7) 

    elif ty == "red":
        neighborhood = disk(8)

    # max mask contains dilated areas of peaks with background
    local_max = maximum_filter(data, footprint=neighborhood) == data

    # we create the mask of the background
    background = (data==0)

    # erosion of background to remove artifact line
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

    # XOR operation for peak mask 
    detected_peaks = local_max ^ eroded_background

    # peak mask overlaid with the thresholded image
    new_im = np.multiply(data, detected_peaks)

    #############
    # binarizing the output image
    new_im[new_im>0] = 255

    return detected_peaks, new_im