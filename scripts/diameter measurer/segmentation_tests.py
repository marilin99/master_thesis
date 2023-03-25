### testing segmentation methods for bacteria images and imitating fiber segmentation in imagej

import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage import segmentation, color, io
from skimage.future import graph
from skimage.segmentation import felzenszwalb, mark_boundaries
from skimage.util import img_as_float
from scipy import ndimage
from scipy.ndimage import * 
from scipy.ndimage.morphology import generate_binary_structure
import skimage.morphology 
import os 

############### fiber segmentation and actual edge overlay #########################

#img = '/home/marilin/Documents/ESP/data/SYTO_PI/control_50killed_syto_PI_2-Image Export-02_c1-3.jpg'
img = "/home/marilin/Documents/ESP/data/SEM/EcN_II_PEO_131120_GML_15k_03.tif"
#unet_img = "/home/marilin/Documents/ESP/data/unet_test/unet3_image_0.6.png"
edged_img = "/home/marilin/Documents/ESP/data/unet_test/edged_15_03.png"

data = cv2.imread(img, 0) #[:650, :]
#net_data = cv2.imread(unet_img, 0) 
edged_data = cv2.imread(edged_img) 

UNET_PATH = "/home/marilin/Documents/ESP/data/fiber_tests/segmented_img_class/unet_colab/"
TARGET_PATH = "/home/marilin/Documents/ESP/data/fiber_tests/segmented_img_class/unet_colab/masked_w_edge/"

for f in os.listdir(UNET_PATH):
    if f.endswith(".png"):
        unet_seg = cv2.imread(UNET_PATH+f)

        

        #unet_seg[(np.where(unet_seg==[255, 255, 255]).all(axis=2))] = [0, 0, 255]
    #rgb_edges[rgb_edges == (255,255,255)] = (255,0,0)

    #print(rgb_edges.shape)
        edged_data[np.where((edged_data==[255, 255, 255]).all(axis=2))] = [0, 0, 255]
        #print(np.where(edged_data==[255, 255, 255]))
        #edged_data = cv2.cvtColor(edged_data, cv2.COLOR_GRAY2BGR)
    # adding the edges to the black and red image and converting to grayscale
        merged = cv2.bitwise_or(edged_data, unet_seg) #), cv2.COLOR_BGR2GRAY)

        #cv2.imshow("median", median)
        #cv2.imshow("thresh", thresh1)
        #cv2.imshow(f"unet im_{f}", unet_seg)
        #cv2.imshow("merged2", merged2)
        #cv2.imshow(f"merged_{f}", merged)
        # saving merged images 
        core_name = f.split(".png")[0]
        cv2.imwrite(f"{TARGET_PATH}{core_name}.png", np.uint8(merged))
        #cv2.imshow('thinned', thinned)

# cv2.imshow('original', data)
# cv2.imshow("edge data", edged_data)
    #cv2.imshow('end im', result_pwr)
cv2.waitKey(0)
cv2.destroyAllWindows()

###################################################################3




# ### contour detection 
# #img=cv2.resize(img,(256,256))
# gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
# _,thresh = cv2.threshold(gray, np.mean(gray), 255, cv2.THRESH_BINARY_INV)
# edges = cv2.dilate(cv2.Canny(thresh,0,255),None)

# cnt = sorted(cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2], key=cv2.contourArea)[-1]
# mask = np.zeros((512,512), np.uint8)
# masked=cv2.drawContours(mask,[cnt],-1,255,-1)
# dst = cv2.bitwise_and(img, img, mask=mask)
# segmented = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)


### fiber segmentation tests ###
import cv2 
import scipy.ndimage as ndimage

# following methodology from https://pubs.acs.org/doi/pdf/10.1021/ie901179m

# local normalization 

#gray = cv2.GaussianBlur(data, (7,7), 0)
#gray = cv2.blur(data, (5,5))

gray = cv2.bilateralFilter(data, 5, 6, 16.0)

clahe2 = cv2.createCLAHE(tileGridSize=(20,20), clipLimit=1) #clipLimit=1, 
cl2 = clahe2.apply(gray)

result = ndimage.gaussian_gradient_magnitude((cl2).astype(float), sigma=1).astype(np.uint8)
#result[result>1] = 255
# pwr2 = 5
result2 = ndimage.gaussian_gradient_magnitude((gray).astype(float), sigma=1).astype(np.uint8)


# result_pwr = np.uint8((np.float128(cl2)**pwr2 / np.amax(np.float128(cl2)**pwr2) ) * 255)

#_, thresh_gr = cv2.threshold(result_pwr, 80, 255,cv2.THRESH_TOZERO)

# image denoising - wavelet filter daubechies (used for texture/surface/denoising) - 8 scaling coeff, M = 2,
# import mahotas as mh
# t = mh.daubechies(cl2, 'D2', inline=False)



#blur = cv2.GaussianBlur(cl2, (0,0), sigmaX=10, sigmaY=10)

#dilated = cv2.dilate(cl2, (3,3))
# divide
#divide = cv2.divide(data, blur, scale=150)

#edged = cv2.Canny(cl2, 140, 250)
# consider bilateral filter

# gray = cv2.bilateralFilter(data, 5, 12.0, 16.0)
# #gray = cv2.GaussianBlur(data, (5,5), 0)
# # laplacian 
# dst = cv2.Laplacian(gray,cv2.CV_16S, ksize=3)

# # converting back to uint8
# abs_dst = cv2.convertScaleAbs(dst)

# import os 
# for k, v in os.environ.items():
# 	if k.startswith("QT_") and "cv2" in v:
# 	    del os.environ[k]

#color = ('b','g','r')
#hist = cv2.calcHist([clahe2],[0],None,[256],[0,256])

# hist,bin = np.histogram(cl2.ravel(),256,[0,255])
# plt.xlim([0,255])
# plt.plot(hist)
# plt.title('histogram')
# plt.show()


# step thresholder
_, thresh1 = cv2.threshold(cl2, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
_, thresh2 = cv2.threshold(result2, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)


#merged = cv2.bitwise_and(thresh1, unet_data)
merged = cv2.erode(thresh1, None, iterations=4)
merged = cv2.dilate(merged, None, iterations=4)
merged = cv2.medianBlur(merged, 5)
#merged = cv2.erode(merged, None, iterations=1)
unet_data = cv2.medianBlur(merged, 15)

thinned = skimage.morphology.medial_axis(unet_data).astype(np.uint8)
thinned[thinned == 1] = 255

#kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
#merged = cv2.morphologyEx(thresh1, cv2.MORPH_GRADIENT, kernel, iterations=3)

# merged = cv2.cvtColor(cv2.bitwise_not(merged), cv2.COLOR_GRAY2BGR)

# # white values to red 
# merged[np.where((merged==[255, 255, 255]).all(axis=2))] = [0, 0, 255]
# #rgb_edges[rgb_edges == (255,255,255)] = (255,0,0)

# #print(rgb_edges.shape)
# edged_data = cv2.cvtColor(edged_data, cv2.COLOR_GRAY2BGR)
# # adding the edges to the black and red image and converting to grayscale
# merged2 = cv2.cvtColor(cv2.bitwise_or(edged_data, merged), cv2.COLOR_BGR2GRAY)

# #print(np.unique(merged2))

# #labeled_array, num_features = label(merged2, structure = generate_binary_structure(2,2))

# print(np.where(merged2==255))

# #print(np.bincount(labeled_array.ravel()))

# #print(np.unique(labeled_array))
# ## removing redundant whites/blacks in gray 



### visuals 
#cv2.imshow('result', np.uint8(morph))
# cv2.imshow('original', data)
# #cv2.imshow("median", median)
# #cv2.imshow("thresh", thresh1)
# cv2.imshow("unet im", unet_data)
# #cv2.imshow("merged2", merged2)
# cv2.imshow("merged", merged)
# #cv2.imshow("edge data", edged_data)
# cv2.imshow('thinned', thinned)

# #cv2.imshow('end im', result_pwr)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
