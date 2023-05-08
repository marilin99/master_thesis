import numpy as np
import cv2
from scipy import ndimage
from scipy.ndimage import *
import time 
import sys

### helper functions ###
from bac_threshold import works_for_both
from peak_detection import peaker 
from czi_pipe_convert import converter

# dict for gathering data
bac_counting = {}

## check what is in the folder
start_time = time.time()

f = str(sys.argv[1])

if f.endswith(".czi") and "control" not in f:
    unique_file = f.split(".czi")[0]

    # empty file for every czi file 
    base_im_r, base_im_g = np.zeros((512,512), dtype=np.uint8), np.zeros((512,512), dtype=np.uint8)

    file = f
    # czi converter returns dict.items() 

    for color, stacks in converter(file):

        if "red" in color:

            # iterating through the stacks
            for im in stacks:
                
                shape_red = im.shape
                red_chan = cv2.resize(im, (512,512))
                
                # centrosymmetric structure
                labeled_array, num_features_r = label(peaker(*works_for_both(red_chan))[0])

                # adding peaks to the empty image
                base_im_r += peaker(*works_for_both(red_chan))[1]

        elif "green" in color:
            # iterating through the stacks

            for im in stacks:
                shape_green = im.shape
                green_chan = cv2.resize(im, (512,512))
                
                # centrosymmetric structure
                labeled_array, num_features_g = label(peaker(*works_for_both(green_chan))[0])

                base_im_g += peaker(*works_for_both(green_chan))[1]
    
    # full amount of bacteria over all the stacks 
    bac_counting[f"{unique_file}_red_bacteria"] = label(base_im_r)[1]
    bac_counting[f"{unique_file}_green_bacteria"] = label(base_im_g)[1]

print("Amount of bacteria in the sample: ", bac_counting)
