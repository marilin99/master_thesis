import numpy as np
from scipy import ndimage
from scipy.ndimage import *
from skimage.morphology import disk


# https://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array/3689710#3689710
def peaker(data,ty):

    if ty == "green":
        neighborhood = disk(7) # 3 w 512x512 - w 1024 needs +4 size disk?

    elif ty == "red":
        neighborhood = disk(8)

    #apply the local maximum filter; all pixel of maximal value 
    #in their neighborhood are set to 1

    local_max = maximum_filter(data, footprint=neighborhood) == data

    #local_max is a mask that contains the peaks we are 
    #looking for, but also the background.
    #In order to isolate the peaks we must remove the background from the mask.

    #we create the mask of the background
    background = (data==0)

    #a little technicality: we must erode the background in order to 
    #successfully subtract it form local_max, otherwise a line will 
    #appear along the background border (artifact of the local maximum filter)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

    #we obtain the final mask, containing only peaks, 
    #by removing the background from the local_max mask (xor operation)
    detected_peaks = local_max ^ eroded_background

    new_im = np.multiply(data, detected_peaks)

    #### 2nd iteration ####

    #we create the mask of the background
    background = (new_im==0)

    #a little technicality: we must erode the background in order to 
    #successfully subtract it form local_max, otherwise a line will 
    #appear along the background border (artifact of the local maximum filter)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)
    local_max = maximum_filter(new_im, footprint=neighborhood) == new_im
    #we obtain the final mask, containing only peaks, 
    #by removing the background from the local_max mask (xor operation)
    detected_peaks = local_max ^ eroded_background

    new_im = np.multiply(new_im, detected_peaks)

    #############

    new_im[new_im>0] = 255
    
    return detected_peaks, new_im