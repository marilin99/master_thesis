from scipy import ndimage
from scipy.ndimage import *
import cv2
import numpy as np 
import skimage.exposure
import matplotlib.pyplot as plt 
import skimage.morphology 
from scipy.spatial import distance

# this was a thresholded image which has been filled in with pinta for cont. purposes - getting a clean img is still an issue (12.01)
orig = cv2.imread("/home/marilin/Documents/ESP/data/SEM/EcN_II_PEO_131120_GML_15k_01.tif",0)[:650, :]

PATH_1 = cv2.imread("/home/marilin/Documents/ESP/diameterJ_test/sem_test/Segmented Images/EcN_II_PEO_131120_GML_15k_01_S1_reverse.tif",0)[:650, :]

# selecting a pixel that is bordering a black pixel
# make this px as the center of a 13x13 kernel - sanity check and for choosing the direction of the regression line


# choosing 1 white px - edges should be fixed 
# while 1:
#     rnd_idx = np.random.randint(1, (len(np.where(PATH_1 > 0)[0]))-1, 1)
#     x, y = np.where(PATH_1 > 0)[0][rnd_idx][0], np.where(PATH_1 > 0)[1][rnd_idx][0]
#     # neighboring px-s from white should be black (U+L, U+R / B+L, B+R) but majority of kernel should be white

#     # relative to px pos
 
#     U = PATH_1[x-1][y]
#     R = PATH_1[x][y+1]
#     L = PATH_1[x][y-1]
#     B = PATH_1[x+1][y]

#     # safe check 
#     if (U == 0 and L == 0) or (U == 0 and R == 0) or (B == 0 and L == 0) or (B == 0 and R == 0):
#         # kernel edge length
#         n = 13
#         #Creating a 13x13 kernel where x,y is the midpoint 
#         kernel_1 = PATH_1[x-(n//2):x+(n//2+1), y-(n//2):y+(n//2+1)]

#         if np.count_nonzero(kernel_1) > ((n**2) // 2):
#             break

# print(kernel_1)
# print(x,y)

# on the img reversed 
x,y = 332, 695

# distance transform
# skimage outputs the distance in ceiled to int 
thinned = skimage.morphology.medial_axis(PATH_1).astype(np.uint8)


thinned[thinned == 1] = 255

#print(np.nonzero(thinned))

dist = cv2.distanceTransform(PATH_1, cv2.DIST_L2, 3)
#print(np.max(dist))
# max distance from edge to medial axis before normalization
n2 = int(round(np.max(dist)))
# Normalize the distance image for range = {0.0, 1.0}
# so we can visualize and threshold it
dist = cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)

#print(np.unique(dist))
cv2.imshow('Distance Transform Image', dist)

#cv2.imshow("edges", edges)
# how can i find perpendicularity from one px? (along on diag? - towards the direction of more whites? - if same, then look 
# at the general direction of whites?)
# 4 diagonals possible - the quarter with the most whites wins?


n = 13
# #Creating a 13x13 kernel where x,y is the midpoint 
kernel_1 = PATH_1[x-(n//2):x+(n//2+1), y-(n//2):y+(n//2+1)]

# # find biggest sum
UL = np.sum(kernel_1[:n//2, :n//2])
UR = np.sum(kernel_1[:n//2:, n//2+1:])
LL = np.sum(kernel_1[n//2+1:, :n//2])
LR = np.sum(kernel_1[n//2+1:, n//2+1:])

# returning idx-s of quarter 
quarters = np.array(["UL", "UR", "LL", "LR"])
# https://stackoverflow.com/questions/17568612/how-to-make-numpy-argmax-return-all-occurrences-of-the-maximum
winners = quarters[np.flatnonzero(np.array([UL, UR, LL, LR]) == np.max(np.array([UL, UR, LL, LR])))]

# if 2 quarters have high sums - px will become in the middle of the outer edge of a new kernel 
# depending on the side of higher values, the px will be on one of the edges
# if one quarter will have higher values - turning the kernel according to the location of the quarter 

# potential combos - UL+LL, UR+LR, UL+UR, LL+LR, and all quarters separately 

if 'UR' and 'LR' in winners:
     kernel_2 = thinned[x-(n2//2): x+(n2//2 +1), y:y+n2+1]

# https://stackoverflow.com/questions/45225474/find-nearest-white-pixel-to-a-given-pixel-location-opencv


def find_nearest_white(img, origin):
     """
     more naive version 
     """
     nonzero =  np.transpose(np.nonzero(img))
     # euc dist
     distances = np.sqrt((nonzero[:,0] - origin[0]) ** 2 + (nonzero[:,1] - origin[1]) ** 2)
     nearest_index = np.argmin(distances)
     return nonzero[nearest_index]


def find_nearest_whites(img, origin):
     """
     for polynomial fitting 
     """
     nonzero =  np.transpose(np.nonzero(img))
     # euc dist
     distances = np.sqrt((nonzero[:,0] - origin[0]) ** 2 + (nonzero[:,1] - origin[1]) ** 2)
     # selecting 15 closest white points
     nearest_index = np.argsort(distances)[:15]
     return nonzero[nearest_index]





############################################################
# polynomial fitting - draw lines against the polynomial to find the most perpendicular one (once a pt is established - needs a tangent and angle between the drawn line and tangent )

# xs,ys,points = [],[],[]
# for i in range(len(find_nearest_whites(kernel_2, [n2//2,0]))):
#      #points.append([695,332])
#      xs.append(332+ find_nearest_whites(kernel_2, [n2//2,0])[i][0])
#      ys.append(695+find_nearest_whites(kernel_2, [n2//2,0])[i][1])
#      points.append([332+ find_nearest_whites(kernel_2, [n2//2,0])[i][0], 695+find_nearest_whites(kernel_2, [n2//2,0])[i][1]])

# xs,ys,points =  np.array(xs),  np.array(ys), np.array(points)
# # could also use x = points[:,0], y = points[:,1]
# #points = points.reshape((-1, 1, 2))

# # poly fitting
# z = np.polyfit(xs,ys,4)
# f = np.poly1d(z)

# x_new = np.linspace(xs[0], xs[-1], 15)
# y_new = f(x_new)
# new_pts = list(zip(x_new, y_new))


# from scipy import interpolate
#tck  = interpolate.splrep(x_new,y_new)

# needs a point on the polyline
#x0 = 7.3
#y0 = interpolate.splev(x0,tck)
#dydx = interpolate.splev(x0,tck,der=1)

#tngnt = lambda x: dydx*x + (y0-dydx*x0)

# print(tngnt(x))

# print(new_pts)

# Python 3 implementation of above approach
 

# Function to check if two straight
# lines are orthogonal or not
# https://www.geeksforgeeks.org/check-whether-two-straight-lines-are-orthogonal-or-not/

# def checkOrtho(x1, y1, x2, y2, x3, y3, x4, y4):
     
#     # Both lines have infinite slope
#     if (x2 - x1 == 0 and x4 - x3 == 0):
#         return False
 
#     # Only line 1 has infinite slope
#     elif (x2 - x1 == 0):
#         m2 = (y4 - y3) / (x4 - x3)
 
#         if (m2 == 0):
#             return True
#         else:
#             return False
 
#     # Only line 2 has infinite slope
#     elif (x4 - x3 == 0):
#         m1 = (y2 - y1) / (x2 - x1)

#         if (m1 == 0):
#             return True
#         else:
#             return False
 
#     else:
         
#         # Find slopes of the lines
#         m1 = (y2 - y1) / (x2 - x1)
#         m2 = (y4 - y3) / (x4 - x3)
 
#         # Check if their product is -1
#         if (m1 * m2 == -1):
#             return True
#         else:
#           return False



############################################################################
# drawing kernel for sanity check
cv2.rectangle(thinned, (y,x-(n2//2)), (y+n2+1,x+(n2//2 +1)), (255,255,255),1)

# other graphical el-s

#cv2.line(thinned, (695,332), (695+find_nearest_white(kernel_2, (17,0))[0], 332+find_nearest_white(kernel_2, (17,0))[-1]), (255,255,255), 1)
#cv2.polylines(thinned, [points], isClosed=False, color = (255,255,255), thickness = 1)
#cv2.line(thinned, (695,332), (712,332), (255,255,255), 1)

cv2.imshow("thinned", thinned.astype(np.uint8))
orig = cv2.circle(orig, (y,x), radius=4, color=(0, 0, 255), thickness=-1)
#print(np.count_nonzero(PATH_1))
#cv2.imshow("direction", theta)
cv2.imshow("thresh", PATH_1)
cv2.imshow("orig", orig)
cv2.waitKey(0)
cv2.destroyAllWindows()