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

img = cv2.imread('/home/marilin/Documents/ESP/data/SYTO_PI/control_50killed_syto_PI_3-Image Export-02_c1-3.jpg')
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
print("amount of color thresholded greens", int(np.count_nonzero(fil_speck)/avg_length))

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

print("amount of color color thresholded reds", int(np.count_nonzero(fil_speck)/avg_length))


############################################################################3
# taking original image and splitting img into color channels 
# can omit blue - wavelengths/colors and such in xml 

# (B, G, R) = cv2.split(img)
# cv2.imshow("green", G)
# cv2.imshow("red", R)

### retrieving r,g channel images from czitotiff converter 
red_chan = cv2.imread("red_intensities.png",0)
green_chan = cv2.imread("green_intensities.png",0)

# contrast stretching == histog eq
eq_img = cv2.equalizeHist(red_chan)
eq__med_img = cv2.medianBlur(eq_img, 3)


# adaptive hg eq - https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html
clahe1 = cv2.createCLAHE(clipLimit=1, tileGridSize=(30,30))
cl1 = clahe1.apply(red_chan)

# chaning the tile grid size - generally needs to change the threshold value (50,50) - 60, (60,60)-70
clahe2 = cv2.createCLAHE(clipLimit=1, tileGridSize=(60,60))
cl2 = clahe2.apply(green_chan)

# cv2.imshow("clahe", cl1)
# cv2.imshow("clahe2", cl2)

#fil_speck = cv2.filterSpeckles(thinned, 0, 1, 1)[0]
# thresholding so that the lower values become black, and lighter values will remain the same 

# red has brighter intensities - this also needs to be automated 
# _, thresh_red = cv2.threshold(red_chan, 70, 255,cv2.THRESH_TOZERO_INV)
# _, thresh_green = cv2.threshold(green_chan, 50, 255,cv2.THRESH_TOZERO_INV)
#lengths_of_dots = sorted(np.bincount(labeled_array.flatten()))

# normalizing w local mean kernel 
kerny = np.ones((3,3))
#red_chan[kerny] -  cv2.blur(red_chan,(3,3))


# cubic 
app_img = cl1
pwr = 2
red_chan_eq = np.uint8((np.float128(app_img)**pwr / np.amax(np.float128(app_img)**pwr) ) * 255)


app_img_2 = cl2
pwr2 = 1
# green maxima won't help 
# if pwr = 1, th = 80
green_chan_eq = np.uint8((np.float128(app_img_2)**pwr2 / np.amax(np.float128(app_img_2)**pwr2) ) * 255)
_, thresh_gr = cv2.threshold(green_chan_eq, 80, 255,cv2.THRESH_TOZERO)

#thresh_gr = cv2.equalizeHist(thresh_gr)
## edge detection ##
# edge = cv2.Canny(thresh_gr, 60, 255)
# cv2.imshow("edge", edge)
##


### top/black hat transform 
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1,1))
# tophat = cv2.morphologyEx(green_chan_eq, cv2.MORPH_BLACKHAT, kernel)
# cv2.imshow("tophat_g", tophat)
###

## contouring w green chan - 431 ## 
# contours,hierarchy = cv2.findContours(thresh_gr,2,1)
# print("cont", len(contours))
##

## peak_local_max skimage 
#from scipy.ndimage.measurements import center_of_mass, label 

# merging results in 425 greens the same as binarized thresholded image label 
#peaks = peak_local_max(thresh_gr, min_distance = 0, exclude_border = False)
# is_peak = peak_local_max(thresh_gr, indices=False, min_distance=0, exclude_border=False) # outputs bool image
# labels = label(is_peak)[0]
# merged_peaks = center_of_mass(is_peak, labels, range(1, np.max(labels)+1))
# merged_peaks = np.array(merged_peaks)


# print(len(merged_peaks))
##

## hough ellipse from skimage - takes ages ##
# from skimage.transform import hough_ellipse

# result = hough_ellipse(thresh_gr, accuracy=20, threshold=60,
#                        min_size=1, max_size=25)
# result.sort(order='accumulator')
# print("hoguh_ell", len(result))
##

#red_chan_eq = cv2.erode(red_chan_eq, None, iterations=1)


#result = ndimage.gaussian_gradient_magnitude((red_chan).astype(float), sigma=1).astype(np.uint8)


# maximum filter - need to establish kernel size - based on bac size 
#image_max = maximum_filter(red_chan, size=5, mode='constant')

# skimage feature for original and local max comparison 
#coordinates = peak_local_max(red_chan, min_distance=9)


# initial testing w blob detector - not useful 

##### merged channel testing #####

# new_chan = np.zeros_like(img)
# new_chan[:,:,2] += red_chan 
# new_chan[:,:,1] += green_chan
# #cv2.normalize(new_chan, new_chan, 0, 255, cv2.NORM_MINMAX))

# gray_newchan = cv2.cvtColor(new_chan, cv2.COLOR_BGR2GRAY)
# cv2.imshow("new_chan", np.uint8( gray_newchan))

# eq_img = cv2.equalizeHist(red_chan)
# cv2.imshow("eq_img", eq_img)
# clahe1 = cv2.createCLAHE(clipLimit=1, tileGridSize=(30,30))
# cl_newchan = clahe1.apply(gray_newchan)

# cv2.imshow("new_chan_eq", np.uint8( cl_newchan))

#######

## binary thresholding

_, thresh_r = cv2.threshold(red_chan_eq, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

_, thresh_g = cv2.threshold(thresh_gr, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

##


#### erosion w circular structuring el 
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
# dilate_g = cv2.dilate(thresh_g, kernel, iterations =1)
# cv2.imshow("dilate_g", dilate_g)
erode_g = cv2.erode(thresh_g, kernel, iterations = 2)
#erode_r = cv2.erode(thresh_r, kernel, iterations = 4)
#cv2.imshow("erode_g", erode_g)

# print("amount of reds", int(np.count_nonzero(erode_r)/20))

# at n = 20, green is 773 - very case-dependent - red: 510
#print("amount of greens", int(np.count_nonzero(erode_g)/20))
####

### hough gradient method in opencv - 1 circle :))

# circles = cv2.HoughCircles(thresh_gr, cv2.HOUGH_GRADIENT, dp = 1, minDist=1, param1 = 30, param2 = 5, minRadius = 0, maxRadius = -1)
# print("circles", len(circles))

# np.around(circles)
###

####

dist= cv2.distanceTransform(thresh_g, cv2.DIST_L2, 3)
dist = cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)


# dist[dist <= np.unique(dist)[1]] = 0
# print(np.unique(dist))
#cv2.imshow('Distance Transform Image', dist)

##### watershedding test 2

# opening = cv2.morphologyEx(thresh_g, cv2.MORPH_OPEN, kernel, iterations=1)
# #cv2.imshow("opening", opening)
# dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 3)
# sure_bg = cv2.dilate(opening,kernel,iterations=3)

# #cv2.imshow("sure_bg", sure_bg)
# ret, sure_fg = cv2.threshold(dist_transform,0.3*dist_transform.max(),255,0)

# # Finding unknown region
# sure_fg = np.uint8(sure_fg)
# #cv2.imshow("sure_fg", sure_fg)
# unknown = cv2.subtract(sure_bg,sure_fg)
# cv2.imshow("unkn", unknown)

# Marker labelling
# ret, markers = cv2.connectedComponents(sure_fg)
# # Add one to all labels so that sure background is not 0, but 1
# markers = markers+1
# # Now, mark the region of unknown with zero
# markers[unknown==255] = 0

# color_g = cv2.cvtColor(thresh_gr, cv2.COLOR_GRAY2BGR)
# markers = cv2.watershed(color_g,markers)
# color_g[markers == -1] = [255,0,0]

#print("amount of markers",len(markers))



#####

#### diff between distance transform of between dilated and and not dilated image 

# dilated_g = cv2.dilate(thresh_g, (3,3), iterations = 3)
# #cv2.imshow("dilated_g", dilated_g)

# # distance transforms 
# dist_thresh = cv2.distanceTransform(thresh_g, cv2.DIST_L2, 3)
# dist_dilated = cv2.distanceTransform(dilated_g, cv2.DIST_L2, 3)

# dist_diff = dist_dilated - dist_thresh
# #print(np.unique(dist_diff))

# #print(dist_diff)
# dist_diff = cv2.normalize(dist_diff, dist_diff, 0, 1, cv2.NORM_MINMAX) * 255
#cv2.imshow('Distance Transform Image', dist_diff)

#print("amount of vals", len(np.where(dist_diff > 0.46)[0]))
#print(np.unique(dist_diff))
####

### distance transform ###

# dist = cv2.distanceTransform(thresh_g, cv2.DIST_L2, 3)
# #print(np.max(dist))
# # max distance from edge to medial axis before normalization

# # Normalize the distance image for range = {0.0, 1.0}
# # so we can visualize and threshold it
# dist = cv2.normalize(dist, dist, 0, 1, cv2.NORM_MINMAX) * 255

# print(np.unique(dist))
# cv2.imshow('Distance Transform Image', dist)

###

## thinning ## - 844 g, 910 r (for export 02)
# thinned = skimage.morphology.medial_axis(thresh_g).astype(np.uint8)
# thinned[thinned == 1] = 255

# thinnedr = skimage.morphology.medial_axis(thresh_r).astype(np.uint8)
# thinnedr[thinnedr == 1] = 255


# print("thinend", int(np.count_nonzero(thinned)/6))
# print("thinendr", int(np.count_nonzero(thinnedr)/6))
##

### trying watershed with green c ### - 512 markers detected

# noise removal
# kernel = np.ones((3,3),np.uint8)
# opening = cv2.morphologyEx(thresh_g,cv2.MORPH_OPEN,kernel, iterations = 2)
# # sure background area
# sure_bg = cv2.dilate(opening,kernel,iterations=1)
# # Finding sure foreground area
# dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
# ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
# # Finding unknown region
# sure_fg = np.uint8(sure_fg)


# unknown = cv2.subtract(sure_bg,sure_fg)
# # Marker labelling
# ret, markers = cv2.connectedComponents(sure_fg)
# # Add one to all labels so that sure background is not 0, but 1
# markers = markers+1
# # Now, mark the region of unknown with zero
# markers[unknown==255] = 0

# color_g = cv2.cvtColor(green_chan, cv2.COLOR_GRAY2BGR)
# markers = cv2.watershed(color_g,markers)
# color_g[markers == -1] = [255,0,0]

# print("amount of markers",len(markers))

# cv2.imshow("markers", color_g)

######


# white px-s counting not viable - bac size depends if its alone or in a colony (length from medial axis is also fishy)
# print("amount of reds", int(np.count_nonzero(thresh_r)/14))

labeled_array, num_features = label(thresh_r)

# px per region
amount_of_bacs = sorted(np.bincount(labeled_array.flatten()))[:-1]

print("r", num_features)


# white px-s counting not viable - bac size depends if its alone or in a colony
# print("amount of reds", int(np.count_nonzero(thresh_r)/14))

### green ### 
labeled_array, num_features = label(thresh_g)

# px per region
amount_of_bacs = sorted(np.bincount(labeled_array.flatten()))[:-1]

print("g", num_features)
# actual: 849 (red), green - 795

## maximum filter - need to establish kernel size - based on bac size ##

image_max = maximum_filter(thresh_gr, size=5, mode='constant')
msk = (thresh_gr == image_max)

print(msk.shape)
##

# cv2.imshow('unknown', unknown)

#cv2.imshow("green_chan", green_chan)
# cv2.imshow("red_chan", np.uint8(red_chan_eq))
#cv2.imshow("green_chan_eq", np.uint8(green_chan_eq))
# #cv2.imwrite("red_chan_cubic.png", red_chan)
# cv2.imshow("thresh_bin_g", thresh_g)
# cv2.imshow("thresh_gr", thresh_gr)
# #cv2.imshow("res", result)
# cv2.imshow("image_max_green", image_max)
# # cv2.imshow("thresh_green", thresh_green)
# # cv2.imshow("thresh_red", thresh_red)
# # cv2.imshow("edged", edged)



cv2.imshow("orig", img)
# cv2.imshow("ske", thinned)
#cv2.imshow("fill_speck", fil_speck)
cv2.waitKey(0)
cv2.destroyAllWindows()


