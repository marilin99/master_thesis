### here I will be testing out the color-based thresholded objects to see if some particle counting could be done based on singular 
# object - so far thresholder.py handles this 

# took color thresholded images - maybe should consider r,g channel intensities instead? - color threhsolding a bit too discrete
import os.path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage import *
import skimage
from skimage.morphology import skeletonize, medial_axis
from skimage.feature import peak_local_max

img = cv2.imread('/home/marilin/Documents/ESP/data/SYTO_PI/control_50killed_syto_PI_2-Image Export-02_c1-3.jpg')
PATH_1 = cv2.imread("thresholded_g.png", 0)
PATH_2 = cv2.imread("thresholded_r.png", 0)


###################################################################################3
# took color thresholded images from thresholder.py - maybe should consider r,g channel intensities instead? - color threhsolding a bit too discrete


thinned = skimage.morphology.medial_axis(PATH_1).astype(np.uint8)
thinned[thinned == 1] = 255
# remove singular white dots
fil_speck = cv2.filterSpeckles(thinned, 0, 1, 1)[0]


# skimage label - areas on white, amount of px-s 

labeled_array, num_features = label(fil_speck)
white_region_counter = np.bincount(labeled_array.flatten())

# could find the smallest area through contouring and find median over 10 smallest or sth
lengths_of_dots = sorted(np.bincount(labeled_array.flatten()))

# max lengths for one bac - changeable based on the meta - needs some proof why a range or sth 
max_vals = np.array([2,3,4,5,6])
mask = np.in1d(lengths_of_dots, max_vals)
lst = []
#for val in mask: lst.append(lengths_of_dots[val])

#print(np.mean(lst))
#print(np.mean(lengths_of_dots[mask]))

# avg len is actually around 2.8 which is not good enough 
# singular bac on average - length 
avg_length = 4.5 
print("amount of greens", int(np.count_nonzero(fil_speck)/avg_length))

### red bac 

thinned_r = skimage.morphology.medial_axis(PATH_2).astype(np.uint8)
thinned_r[thinned_r == 1] = 255
# remove singular white dots
fil_speck = cv2.filterSpeckles(thinned_r, 0, 1, 1)[0]

# skimage label - areas on white, amount of px-s 

labeled_array, num_features = label(fil_speck)
white_region_counter = np.bincount(labeled_array.flatten())

# singular bac on average - length 
# could find the smallest area through contouring and find median over 10 smallest or sth

print("amount of reds", int(np.count_nonzero(fil_speck)/avg_length))


############################################################################3
# taking original image and splitting img into color channels 
# can omit blue - wavelengths/colors and such in xml 

# (B, G, R) = cv2.split(img)
# cv2.imshow("green", G)
# cv2.imshow("red", R)

### retrieving r,g channel images from czitotiff converter 
red_chan = np.uint32(cv2.imread("red_intensities.png",0))
green_chan = cv2.imread("green_intensities.png",0)

# thresholding so that the lower values become black, and lighter values will remain the same 

# red has brighter intensities - this also needs to be automated 
# _, thresh_red = cv2.threshold(red_chan, 70, 255,cv2.THRESH_TOZERO_INV)
# _, thresh_green = cv2.threshold(green_chan, 50, 255,cv2.THRESH_TOZERO_INV)
lengths_of_dots = sorted(np.bincount(labeled_array.flatten()))

# normalizing w local mean kernel 
kerny = np.ones((3,3))
#red_chan[kerny] -  cv2.blur(red_chan,(3,3))


# cubic 
red_chan = np.uint8((red_chan**3 / np.amax(red_chan**3) ) * 255)

#result = ndimage.gaussian_gradient_magnitude((red_chan).astype(float), sigma=1).astype(np.uint8)


# maximum filter - need to establish kernel size - based on bac size 
#image_max = maximum_filter(red_chan, size=5, mode='constant')

# skimage feature for original and local max comparison 
#coordinates = peak_local_max(red_chan, min_distance=9)


# testing w blob detector - not useful 

##### watershed algo testing - not  useful 

# _, thresh_r = cv2.threshold(red_chan, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# labeled_array, num_features = label(thresh_r)

# amount_of_bacs = sorted(np.bincount(labeled_array.flatten()))


# maximum filter - need to establish kernel size - based on bac size 
#image_max = maximum_filter(red_chan, size=5, mode='constant')# red_chan[markers == -1] = (255, 0, 0)

# cv2.imshow('unknown', unknown)

#cv2.imshow("green_chan", green_chan)
cv2.imshow("red_chan", np.uint8(red_chan))
#cv2.imwrite("red_chan_cubic.png", red_chan)
#cv2.imshow("thresh_r", thresh_r)
#cv2.imshow("res", result)
#cv2.imshow("image_max_red", image_max)
# cv2.imshow("thresh_green", thresh_green)
# cv2.imshow("thresh_red", thresh_red)
# cv2.imshow("edged", edged)
#dist = cv2.distanceTransform(PATH_1, cv2.DIST_L2, 3)
#print(np.max(dist))
# max distance from edge to medial axis before normalization
#n2 = int(round(np.max(dist)))
# Normalize the distance image for range = {0.0, 1.0}
# so we can visualize and threshold it
#dist = cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)

#print(np.unique(dist))
#cv2.imshow('Distance Transform Image', dist)


cv2.imshow("orig", img)
#cv2.imshow("ske", thinned)
#cv2.imshow("fill_speck", fil_speck)
cv2.waitKey(0)
cv2.destroyAllWindows()


