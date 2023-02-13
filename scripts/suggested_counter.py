# suggested method to count the bacteria 

import os.path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage import *
import skimage
from skimage.morphology import skeletonize, medial_axis, disk
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from skimage.feature import peak_local_max
import scipy.ndimage.filters as filters 

from varname import argname

# uncomment this if showing image using other src than opencv 
# for k, v in os.environ.items():
# 	if k.startswith("QT_") and "cv2" in v:
# 	    del os.environ[k]

img = cv2.imread('/home/marilin/Documents/ESP/data/SYTO_PI/fibers_24h_growth_syto_PI_1-Image Export-07_c1-3.jpg')
#red_chan = cv2.imread("red_intensities.png",0)
#red_chan = cv2.imread("red_intensities_03.png",0)
# red_chan = cv2.imread("red_intensities_incub_4h_2.png",0)
# #green_chan = cv2.imread("green_intensities.png",0)
# #green_chan = cv2.imread("green_intensities_03.png",0)
# green_chan = cv2.imread("green_intensities_incub_4h_2.png",0)

# # resizing to 512,512 - works_for_both() func dependency 
# red_chan = cv2.resize(red_chan, (512,512))
# green_chan = cv2.resize(green_chan, (512,512))


(B,green_chan,red_chan) = cv2.split(img)
img[:,:,1] = 0 
img[:,:,1] = green_chan
img[:,:,2] = 0 
img[:,:,2] = red_chan
cv2.imshow("green", img[:,:,1])
cv2.imshow("red", img[:,:,2])
## mean filter 

## red chan 
#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
#red_chan_mean = cv2.blur(red_chan, kernel)

def works_for_red(data):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4,4))
    # mean filter
    red_chan_mean = cv2.blur(data, (3,3))

    ## dilate 
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4,4))
    red_dilated = cv2.dilate(red_chan_mean, kernel)
    _, thresh_r = cv2.threshold(red_dilated, 25, 255,cv2.THRESH_TOZERO)

    cv2.imshow("data", data)
    cv2.imshow("normalized", red_chan_mean)
    cv2.imshow("dilated", red_dilated)
    cv2.imshow("thresholded", thresh_r)

    return thresh_r


###
def works_for_green(data):

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4,4))
    ## green chan
    # clahe normalization before?
    clahe2 = cv2.createCLAHE(clipLimit=1, tileGridSize=(20,20))
    cl2 = clahe2.apply(data)

    #green_chan_mean = cv2.blur(cl2, (3,3))
    green_dilated = cv2.dilate(cl2, kernel)
    # green
    #_, thresh_g = cv2.threshold(green_dilated, 40, 255,cv2.THRESH_TOZERO)
    _, thresh_g = cv2.threshold(green_dilated, 0, 255, cv2.THRESH_TOZERO+cv2.THRESH_OTSU)

    cv2.imshow("data", data)
    cv2.imshow("normalized", cl2)
    cv2.imshow("dilated", green_dilated)
    cv2.imshow("thresholded", thresh_g)

    return thresh_g

### control bac counter 
def works_for_both(data):

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4,4))
    ## green chan
    # clahe normalization before?
    # clahe req grayscale img
    clahe2 = cv2.createCLAHE(clipLimit=1, tileGridSize=(20,20))
    cl2 = clahe2.apply(data)

    #green_chan_mean = cv2.blur(cl2, (3,3))
    dilated = cv2.dilate(cl2, kernel)
   
    #_, thresh_g = cv2.threshold(green_dilated, 40, 255,cv2.THRESH_TOZERO)
    _, thresh = cv2.threshold(dilated, 0, 255, cv2.THRESH_TOZERO+cv2.THRESH_OTSU)
    
    # https://github.com/pwwang/python-varname
    
    # change condition if needed - type based on variable name 
    if 'g' in argname("data"):
        ty = "green"

    elif 'r' in argname("data"):
        ty = "red"

    # cv2.imshow("data", data)
    # cv2.imshow("normalized", cl2)
    # cv2.imshow("dilated", dilated)
    # cv2.imshow("thresholded", thresh)

    return thresh, ty

## scipy.ndimage
## control bac counter 
# https://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array/3689710#3689710
def peaker(data,ty):
    # define a disk shape - 4 for red channel, 3 for green channel
    if ty == "green":
        neighborhood = disk(3)
    elif ty == "red":
        neighborhood = disk(4)

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

    cv2.imshow("detected_peaks", np.multiply(data, detected_peaks))

    return detected_peaks

# output 859 for red ("real" - 849) w disk r 4 (w works_for_red)
# w both - 892
labeled_array, num_features_r = label(peaker(*works_for_both(red_chan)))


# # output 796 for green ("real" - 795) w disk r 3 (green_intensities.png)
# w both - 752
labeled_array, num_features_g = label(peaker(*works_for_both(green_chan)))

print("greens: ", num_features_g, "reds: ", num_features_r)

##

## mahotas testing - regional max functionality
# import pylab
# import mahotas as mh

# # http://pythonvision.org/basic-tutorial/

# #red_gaus = mh.mean_filter(red_chan, 3)
# rmax = mh.regmax(thresh_r)
# # pylab.imshow(mh.overlay(red_dilated, rmax))
# # pylab.show()
# # pylab.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# # pylab.show()
# seeds,nr_nuclei = mh.label(rmax)
# #print(nr_nuclei)

# # needs some watershedding 
# #_, thresh_red = cv2.threshold(thresh_r, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# T = mh.thresholding.otsu(thresh_r)
# dist = mh.distance(thresh_r > T)
# dist = dist.max() - dist
# dist -= dist.min()
# dist = dist/float(dist.ptp()) * 255
# dist = dist.astype(np.uint8)
# #pylab.imshow(cv2.cvtColor(dist, cv2.COLOR_BGR2RGB))
# #pylab.show()
# nuclei = mh.cwatershed(dist, seeds)
# # pylab.imshow(nuclei)
# # pylab.show()

# whole = mh.segmentation.gvoronoi(nuclei)


##


## local intensity maxima 
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
# neighborhood_size = 3 
# threshold = 30 


# data_max = filters.maximum_filter(red_chan_mean, neighborhood_size)
# maxima = (red_chan_mean == data_max)
# data_min = filters.minimum_filter(red_chan_mean, neighborhood_size)
# diff = ((data_max - data_min) > threshold)
# maxima[diff == 0] = 0
# labeled, num_objects = ndimage.label(maxima)
# slices = ndimage.find_objects(labeled)
##

cv2.imshow("orig", img)
# cv2.imshow("red_chan", red_chan)
# # cv2.imshow("red_chan_mean", cl2)
# # cv2.imshow("red_dilated", green_dilated)
# # cv2.imshow("red_thresh", thresh_g)
cv2.waitKey(0)
cv2.destroyAllWindows()