### testing segmentation methods for bacteria images

import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.segmentation import felzenszwalb, mark_boundaries
from skimage.util import img_as_float

img = '/home/marilin/Documents/ESP/data/SYTO_PI/control_50killed_syto_PI_2-Image Export-02_c1-3.jpg'

sample_image = cv2.imread(img)
img = cv2.cvtColor(sample_image,cv2.COLOR_RGB2HSV)

### k means segmentation 
# 3d to 2d
twoDimage = img.reshape((-1,3))
twoDimage = np.float32(twoDimage)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 4
attempts=10

ret,label,center=cv2.kmeans(twoDimage,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
center = np.uint8(center)
res = center[label.flatten()]
result_image = res.reshape((img.shape))


### contour detection 
#img=cv2.resize(img,(256,256))
gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
_,thresh = cv2.threshold(gray, np.mean(gray), 255, cv2.THRESH_BINARY_INV)
edges = cv2.dilate(cv2.Canny(thresh,0,255),None)

cnt = sorted(cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2], key=cv2.contourArea)[-1]
mask = np.zeros((512,512), np.uint8)
masked=cv2.drawContours(mask,[cnt],-1,255,-1)
dst = cv2.bitwise_and(img, img, mask=mask)
segmented = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)

####color masking 
low_green = np.array([36,25,26])
high_green = np.array([70,255,249])

low_red =  np.array([0,25,26])
high_red = np.array([31,255,249])

thresholded = cv2.inRange(img, low_green, high_green)

result = cv2.bitwise_and(sample_image, sample_image, mask = mask)

### felzenszwalb segmentation
img = img_as_float(img[::2, ::2])

segments_fz = felzenszwalb(img, scale=100, sigma=0.5, min_size=10)

print(f'Felzenszwalb number of segments: {len(np.unique(segments_fz))}')

### misc conversion


# cv2.imshow('result', result)
cv2.imshow('original', sample_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
