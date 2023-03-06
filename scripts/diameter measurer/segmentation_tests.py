### testing segmentation methods for bacteria images and imitating fiber segmentation in imagej

import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage import segmentation, color, io
from skimage.future import graph
from skimage.segmentation import felzenszwalb, mark_boundaries
from skimage.util import img_as_float

#img = '/home/marilin/Documents/ESP/data/SYTO_PI/control_50killed_syto_PI_2-Image Export-02_c1-3.jpg'
img = "/home/marilin/Documents/ESP/data/SEM/EcN_II_PEO_131120_GML_15k_01.tif"

data = cv2.imread(img, 0)[:650, :]
#img = cv2.cvtColor(sample_image,cv2.COLOR_BGR2HSV)

# ### k means segmentation 
# # 3d to 2d
# twoDimage = img.reshape((-1,3))
# twoDimage = np.float32(twoDimage)

# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# K = 4
# attempts=10

# ret,label,center=cv2.kmeans(twoDimage,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
# center = np.uint8(center)
# res = center[label.flatten()]
# result_image = res.reshape((img.shape))


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

# ####color masking 
# low_green = np.array([36,25,26])
# high_green = np.array([70,255,249])

# low_red =  np.array([0,25,26])
# high_red = np.array([31,255,249])

# thresholded = cv2.inRange(img, low_green, high_green)

# result = cv2.bitwise_and(sample_image, sample_image, mask = mask)

# ### felzenszwalb segmentation
# img = img_as_float(img[::2, ::2])

# segments_fz = felzenszwalb(img, scale=100, sigma=0.5, min_size=10)

#print(f'Felzenszwalb number of segments: {len(np.unique(segments_fz))}')

### rag skimage
# https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_rag_merge.html#sphx-glr-auto-examples-segmentation-plot-rag-merge-py

# def _weight_mean_color(graph, src, dst, n):
#     """Callback to handle merging nodes by recomputing mean color.

#     The method expects that the mean color of `dst` is already computed.

#     Parameters
#     ----------
#     graph : RAG
#         The graph under consideration.
#     src, dst : int
#         The vertices in `graph` to be merged.
#     n : int
#         A neighbor of `src` or `dst` or both.

#     Returns
#     -------
#     data : dict
#         A dictionary with the `"weight"` attribute set as the absolute
#         difference of the mean color between node `dst` and `n`.
#     """

#     diff = graph.nodes[dst]['mean color'] - graph.nodes[n]['mean color']
#     diff = np.linalg.norm(diff)
#     return {'weight': diff}


# def merge_mean_color(graph, src, dst):
#     """Callback called before merging two nodes of a mean color distance graph.

#     This method computes the mean color of `dst`.

#     Parameters
#     ----------
#     graph : RAG
#         The graph under consideration.
#     src, dst : int
#         The vertices in `graph` to be merged.
#     """
#     graph.nodes[dst]['total color'] += graph.nodes[src]['total color']
#     graph.nodes[dst]['pixel count'] += graph.nodes[src]['pixel count']
#     graph.nodes[dst]['mean color'] = (graph.nodes[dst]['total color'] /
#                                       graph.nodes[dst]['pixel count'])

# # nr of area based
# labels = segmentation.slic(sample_image, compactness=30,n_segments=100,start_label=1)
# g = graph.rag_mean_color(sample_image, labels)
# labels2 = graph.merge_hierarchical(labels, g, thresh=40, rag_copy=False,
#                                    in_place_merge=True,
#                                    merge_func=merge_mean_color,
#                                    weight_func=_weight_mean_color)

# out = color.label2rgb(labels2, sample_image, kind='avg', bg_label=0)
# result = segmentation.mark_boundaries(out, labels2, (0, 0, 0)) 

# gray scale value based?

### fiber segmentation tests ###
import cv2 
import scipy.ndimage as ndimage

# following methodology from https://pubs.acs.org/doi/pdf/10.1021/ie901179m

# local normalization 

#gray = cv2.GaussianBlur(data, (7,7), 0)
#gray = cv2.blur(data, (5,5))
# clahe2 = cv2.createCLAHE(tileGridSize=(20,20), clipLimit=1) #clipLimit=1, 
# cl2 = clahe2.apply(gray)

#result = ndimage.gaussian_gradient_magnitude((cl2).astype(float), sigma=1).astype(np.uint8)

#pwr2 = 5
# green maxima won't help 
# if pwr = 1, th = 80
#result_pwr = np.uint8((np.float128(cl2)**pwr2 / np.amax(np.float128(cl2)**pwr2) ) * 255)
#_, thresh_gr = cv2.threshold(result_pwr, 80, 255,cv2.THRESH_TOZERO)

# image denoising - wavelet filter daubechies (used for texture/surface/denoising) - 8 scaling coeff, M = 2,
# import mahotas as mh
# t = mh.daubechies(cl2, 'D2', inline=False)

# end_im = cl2-t
# multiway pca 

#blur = cv2.GaussianBlur(cl2, (0,0), sigmaX=10, sigmaY=10)

#dilated = cv2.dilate(cl2, (3,3))
# divide
#divide = cv2.divide(data, blur, scale=150)

#edged = cv2.Canny(cl2, 140, 250)

gray = cv2.GaussianBlur(data, (5,5), 0)
# laplacian 
dst = cv2.Laplacian(gray,cv2.CV_16S, ksize=3)

# converting back to uint8
abs_dst = cv2.convertScaleAbs(dst)

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
#_, thresh = cv2.threshold(cl2, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
_, thresh = cv2.threshold(abs_dst, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# merged = cv2.bitwise_and(result_pwr, thresh)

# median = cv2.medianBlur(merged, 3)
# #new_m = cv2.bitwise_or(data, median)
# #gradient thresholder - canny / sobel 


# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
# morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)


### visuals 
#cv2.imshow('result', np.uint8(morph))
cv2.imshow('original', data)
# cv2.imshow("median", median)
cv2.imshow("new_m", thresh)
# cv2.imshow('clahe', cl2)
cv2.imshow("lap", abs_dst)
#cv2.imshow('end im', result_pwr)
cv2.waitKey(0)
cv2.destroyAllWindows()
