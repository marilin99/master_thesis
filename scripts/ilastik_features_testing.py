from scipy import ndimage
from scipy.ndimage import *
import cv2
import numpy as np 
import skimage.exposure
import matplotlib.pyplot as plt 
from skimage import color
from skimage import data
from skimage import morphology
from skimage import segmentation

PATH_1 = cv2.imread("/home/marilin/Documents/ESP/data/SEM/EcN_II_PEO_131120_GML_15k_03.tif")
prob_1 = cv2.imread("/home/marilin/Documents/ESP/data/SEM_jpg/EcN_II_PEO_131120_GML_15k_03_Probabilities.tif")
#PATH_2 = cv2.imread("/home/marilin/Documents/ESP/data/SEM_jpg/Gaussian Gradient Magnitude (σ=1).tif")
#PATH_3 = cv2.imread("test.jpg")
#img = cv2.GaussianBlur(PATH_1, (0,0), sigmaX=1, sigmaY=1)

# sobel 
# Kx = np.array([[-1, 0, 1], 
#                [-2, 0, 2], 
#                [-1, 0, 1]])
# Ky = np.array([[1,   2,  1], 
#                [0,   0,  0], 
#               [-1,  -2, -1]])

# Ix = cv2.filter2D(img, -1, Kx)
# Iy = cv2.filter2D(img, -1, Ky)

# G = np.hypot(Ix, Iy)
#G = skimage.exposure.rescale_intensity(float(PATH_1), in_range='image', out_range=(0,255)).astype(np.uint8)

# min_max norm to 0,1 
#norm_img = (PATH_1 - np.min(PATH_1))/np.ptp(PATH_1)


result = ndimage.gaussian_gradient_magnitude((PATH_1).astype(float), sigma=1).astype(np.uint8)
# sharpening 
kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])

norm_img = (result - np.min(result))/np.ptp(result)

image_sharp = cv2.filter2D(src=result, ddepth=-1, kernel=kernel)
#cv2.imwrite("sharpened_img.png", image_sharp)
# sharpening 
#edged = cv2.Canny(result, 30, 30)
#print(np.unique(result))
#_, thresh = cv2.threshold(cv2.cvtColor(image_sharp,cv2.COLOR_BGR2GRAY),13,255,cv2.THRESH_BINARY)
#thresh = cv2.adaptiveThreshold(cv2.cvtColor(image_sharp,cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,9,2)
#thresh = cv2.adaptiveThreshold(cv2.cvtColor(image_sharp,cv2.COLOR_BGR2GRAY),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,9,2)
_, thresh = cv2.threshold(cv2.cvtColor(image_sharp,cv2.COLOR_BGR2GRAY),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


labeled_array, num_features = label(thresh)
white_counter = np.bincount(labeled_array.flatten())

### region with < 70 white px will become black 
#print(sorted(np.bincount(labeled_array.flatten()))[-10:])
#print(np.argsort(np.bincount(labeled_array.flatten()))[-10:])

# white_px = 70
# filtered_counter_w = np.where(white_counter >= white_px, white_counter, 0)
# test_w = np.argwhere(np.bincount(labeled_array.flatten())<=white_px)

# kernel = np.ones((3,3), np.uint8)
# opening = cv2.medianBlur(cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel), 3)


#print(labeled_array.ravel())

#a_tst = np.where(labeled_array.ravel() != test_w.ravel(), labeled_array.ravel(), 0)
#a_tst[a_tst > 0] = 255

# flat_lab = labeled_array.flatten() 
# test_w = test_w.ravel()

# for i in range(len(flat_lab)):
#     for val in test_w:
#         if flat_lab[i] == val:
#             flat_lab[i] = 0
#         else:
#             flat_lab[i] = 255

# flat_lab[flat_lab == test_w] = 0
# flat_lab[flat_lab>0] = 255

#new_thresh = flat_lab.reshape(thresh.shape).astype(np.uint8)


#cv2.imshow("test", new_thresh)

#print(np.unique(labeled_array), num_features)
#print(white_counter[12443])
#print(labeled_array.flatten()[:10])
#print(white_counter[0])

#print(np.unique(filtered_counter_w))

        
#cv2.imwrite("thresh.png", thresh)
#thresh = cv2.adaptiveThreshold(cv2.cvtColor(image_sharp,cv2.COLOR_BGR2GRAY),255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,9,2)
#_, thresh = cv2.threshold(cv2.cvtColor(image_sharp,cv2.COLOR_BGR2GRAY), 0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#med_thresh = cv2.medianBlur(thresh, 3)
#edged = cv2.dilate(med_thresh, None, iterations=1)
#edged = cv2.erode(edged, None, iterations=1)
#bin_fil = binary_fill_holes((med_thresh).astype(float)).astype(np.uint8)
#_, thresh2 = cv2.threshold(cv2.cvtColor(prob_1,cv2.COLOR_BGR2GRAY), 0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
#pic_2 = cv2.bitwise_and(PATH_1, PATH_1, mask = thresh2)

# floodfill 
# Copy the thresholded image.
# im_floodfill = thresh.copy()

# # Mask used to flood filling.
# # Notice the size needs to be 2 pixels than the image.
# h, w = thresh.shape[:2]
# mask = np.zeros((h+2, w+2), np.uint8)

# # Floodfill from point (0, 0)
# cv2.floodFill(im_floodfill, mask, (0,0), 255)

# # Invert floodfilled image
# im_floodfill_inv = cv2.bitwise_not(im_floodfill)

# # Combine the two images to get the foreground.
# im_out = thresh | im_floodfill_inv
#cv2.imshow("il_out", PATH_2)
#cv2.imshow("test", bin_fil)

#theta = np.arctan2(Iy, Ix)
#theta = skimage.exposure.rescale_intensity(result, in_range='image', out_rage=(0,255)).astype(np.uint8)
   
#cv2.imshow("magnitude", G)
#cv2.imshow("direction", theta)
#cv2.imshow(":", norm_img)
cv2.imshow("res", result)
#cv2.imshow("med_thres", med_thresh)
#cv2.imshow("edged", edged)
cv2.imshow("th", thresh)
#cv2.imshow("Foreground", im_out)
#cv2.imshow("med", pic_2)
cv2.waitKey(0)