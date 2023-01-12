from scipy import ndimage
from scipy.ndimage import *
import cv2
import numpy as np 
import skimage.exposure
import matplotlib.pyplot as plt 

# this was a thresholded image which has been filled in with pinta for cont. purposes - getting a clean img is still an issue (12.01)
PATH_1 = cv2.imread("/home/marilin/Documents/ESP/thresh.png",0)[:650, :]

# selecting a pixel that is bordering a black pixel
# make this px as the center of a 9x9 kernel - sanity check and for choosing the direction of the regression line

#print(np.where(np.all(PATH_1[1:-1, 1:-1] == 0, axis=2)))

rnd_idx = np.random.randint(0, (len(np.where(PATH_1 > 0)[0])), 1)
x, y = np.where(PATH_1 > 0)[0][rnd_idx][0], np.where(PATH_1 > 0)[1][rnd_idx][0]
# neighboring px-s from white should be black (U+L, U+R / B+L, B+R) but majority of kernel should be white

# sanity check 


print(x,y)
# kernel len
n = 13
#Creating a 13x13 kernel where x,y is the midpoint 
kernel_1 = PATH_1[x-(n//2):x+(n//2+1), y-(n//2):y+(n//2+1)]


#PATH_1 = cv2.circle(PATH_1, (x,y), radius=10, color=(0, 0, 0), thickness=-1)
#print(np.count_nonzero(PATH_1))
#cv2.imshow("direction", theta)
cv2.imshow(":", PATH_1)
cv2.waitKey(0)