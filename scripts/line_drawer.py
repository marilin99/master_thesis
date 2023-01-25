from scipy import ndimage
from scipy.ndimage import *
import cv2
import numpy as np 
import skimage.exposure
import matplotlib.pyplot as plt 

# this was a thresholded image which has been filled in with pinta for cont. purposes - getting a clean img is still an issue (12.01)
PATH_1 = cv2.imread("/home/marilin/Documents/ESP/thresh.png",0)[:650, :]

# selecting a pixel that is bordering a black pixel
# make this px as the center of a 13x13 kernel - sanity check and for choosing the direction of the regression line


# choosing 1 white px 
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

# x,y = 421, 473

# how can i find perpendicularity from one px? (along on diag? - towards the direction of more whites? - if same, then look 
# at the general direction of whites?)
# 4 diagonals possible - the quarter with the most whites wins?

x,y = 421, 473
n = 13
#Creating a 13x13 kernel where x,y is the midpoint 
kernel_1 = PATH_1[x-(n//2):x+(n//2+1), y-(n//2):y+(n//2+1)]

# find biggest sum
UL = np.sum(kernel_1[:n//2, :n//2])
UR = np.sum(kernel_1[:n//2:, n//2+1:])
LL = np.sum(kernel_1[n//2+1:, :n//2])
LR = np.sum(kernel_1[n//2+1:, n//2+1:])

#print(np.argmax(np.array([UL, UR, LL, LR])))

# 4th quarter - np.diag


#https://stackoverflow.com/questions/28417604/plotting-a-line-from-a-coordinate-with-and-angle
def get_point(point, angle, length):
     '''
     point - Tuple (x, y)
     angle - Angle you want your end point at in degrees.
     length - Length of the line you want to plot.

     '''
     # unpack the first point
     x, y = point

     # find the end point
     endy = y + length * np.sin(np.radians(angle))
     endx = x+ length * np.cos(np.radians(angle))

     return (int(endx), int(endy))

# 200px rand length 
#print(get_point((x,y), -45, 200))

mat = np.arange(9).reshape((3,3))

print(get_point((2,0), -45, 3))

# px distance in diagonals 
# dp = np.sqrt(width**2 + height **2)


#PATH_1 = cv2.circle(PATH_1, (x,y), radius=10, color=(0, 0, 0), thickness=-1)
#print(np.count_nonzero(PATH_1))
#cv2.imshow("direction", theta)
cv2.imshow(":", PATH_1)
cv2.waitKey(0)