from scipy import ndimage
from scipy.ndimage import *
import cv2
import numpy as np 
import skimage.exposure
import matplotlib.pyplot as plt 
import skimage.morphology 
from scipy.spatial import distance
from scipy import stats
import time 

start_time = time.time()
# this was a thresholded image which has been filled in with pinta for cont. purposes - getting a clean img is still an issue (12.01)
orig = cv2.imread("/home/marilin/Documents/ESP/data/SEM/EcN_II_PEO_131120_GML_15k_01.tif",0)[:650, :]

# this image is segmented using statistical region merging from diameterj imagej - from 650 is the scale box 
PATH_1 = cv2.imread("/home/marilin/Documents/ESP/diameterJ_test/sem_test/Segmented Images/EcN_II_PEO_131120_GML_15k_01_S1_reverse.tif",0)[:650, :]

#### helper functions ####


# https://stackoverflow.com/questions/45225474/find-nearest-white-pixel-to-a-given-pixel-location-opencv
def find_nearest_white(img, origin):
     """
     more naive version using euc distance
     """

     nonzero =  np.transpose(np.nonzero(img))
     # euc dist
     distances = np.sqrt((nonzero[:,0] - origin[0]) ** 2 + (nonzero[:,1] - origin[1]) ** 2)
     nearest_index = np.argmin(distances)

     return nonzero[nearest_index]


def find_nearest_whites(img, origin):
     """
     for polynomial fitting 
     outputs idx-s of 15 nearest whites
     """
     nonzero =  np.transpose(np.nonzero(img))
     # euc dist
     distances = np.sqrt((nonzero[:,0] - origin[0]) ** 2 + (nonzero[:,1] - origin[1]) ** 2)
     # selecting 15 closest white points
     nearest_index = np.argsort(distances)[:15]
     return nonzero[nearest_index]

#####


######### selecting a pixel that is bordering a black pixel from 2 sides #######

# make this px as the center of a 13x13 kernel - sanity check and for choosing the direction of the regression line

np.random.seed(2)
dist = cv2.distanceTransform(PATH_1, cv2.DIST_L2, 3)


thinned = skimage.morphology.medial_axis(PATH_1).astype(np.uint8)
thinned[thinned == 1] = 255
h,w = PATH_1.shape[0],PATH_1.shape[1]
n2 = int(np.ceil(np.max(dist)))
n = 13

coords = []
# collecting exceptions w x,y and "winners"
exc_cases = []

def point_picker():
     while True:
          #choosing idx of white px - randint should be excluding high values
          
          rnd_idx = np.random.randint(0, (len(np.where(PATH_1 > 0)[0])), 1)

          x, y = np.where(PATH_1 > 0)[0][rnd_idx][0], np.where(PATH_1 > 0)[1][rnd_idx][0]

          # refacto this   (temp safety net: 34<x<h-y, 34 < y < w-y)
          # while not (n2 < x < (h-n2)) and (n2 < y < (w-n2) ):
          #      rnd_idx = np.random.randint(0, (len(np.where(PATH_1 > 0)[0])), 1)
          #      x, y = np.where(PATH_1 > 0)[0][rnd_idx][0], np.where(PATH_1 > 0)[1][rnd_idx][0]


          #neighboring px-s from white should be black (U+L, U+R / B+L, B+R) but majority of kernel should be white

          #relative to px pos
          try:
               U = PATH_1[x-1][y]
               R = PATH_1[x][y+1]
               L = PATH_1[x][y-1]
               B = PATH_1[x+1][y]
          #if too much in the border - find a better location 
          except:
               continue

          # safe check - exclusive or to avoid obscure situations or stick to or?
          if (U == 0 and L == 0) ^ (U == 0 and R == 0) ^ (B == 0 and L == 0) ^ (B == 0 and R == 0):
          #if (U == 0 and L == 0) or (U == 0 and R == 0) or (B == 0 and L == 0) or (B == 0 and R == 0):
               coords.append((x,y))

          # amount of points chosen
          if len(coords) == 100:
               break

     return coords

          # kernel edge length
          #n = 13
          # #Creating a 13x13 kernel where x,y is the midpoint 
          # #kernel_1 = PATH_1[x-(n//2):x+(n//2+1), y-(n//2):y+(n//2+1)]
          # xs.append(x)
          # ys.append(y)
#           # no clue why i have this condition
#           # if np.count_nonzero(kernel_1) > ((n**2) // 2):
#           #      break
#           # else:

            
     

# on the img x,y are reversed 

# # saving dm-s in a list to show histogram in the end
#dm_s = []
# # for understanding exceptions

#coords = [(109,139), (514,828), (138, 341), (470,352), (382,315), (516,827)]

def dm_finder(pt_s, n,n2,thinned):
     dm_s = []
     exc_cases = []
# #print(coords)
# dm_s = []
# pt_s = [(61, 344), (323, 531), (178, 732), (284, 155)]
#pt_s = [(491, 8), (514,1010)]
     for x,y in pt_s: 
     #x,y = 332, 695
          #print(x,y)
     # distance transform
     # skimage outputs the distance in ceiled to int 

     # # #print(np.max(dist))

     # # # max distance from edge to medial axis before normalization


     # # #print(stats.describe(dist))
     # # # if max is 34 - the kernel should be 35x35 - or maybe omit it as an outlier?

     # # # Normalize the distance image for range = {0.0, 1.0}
     # # # so we can visualize and threshold it
     # dist = cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)

     # #print(np.unique(dist))
     # cv2.imshow('Distance Transform Image', dist)


     # # # how can i find perpendicularity from one px? (along on diag? - towards the direction of more whites? - if same, then look 
     # # # at the general direction of whites?)
     # # # 4 diagonals possible - the quarter with the most whites wins?

     # ### direction choosing ###
          if (y+n) > w or (x+n) > h or (y-n) < 0 or (x-n) < 0:
               n = np.min((abs(0-x), (h-x), (w-y), abs(0-y)))
          
          # #Creating a 13x13/nxn kernel where x,y is the midpoint 
          kernel_1 = PATH_1[x-(n//2):x+(n//2+1), y-(n//2):y+(n//2+1)]

          # # find biggest sum
          UL = np.sum(kernel_1[:n//2, :n//2])
          UR = np.sum(kernel_1[:n//2, n//2+1:])
          LL = np.sum(kernel_1[n//2+1:, :n//2])
          LR = np.sum(kernel_1[n//2+1:, n//2+1:])

          # returning strings of quarter 
          quarters = np.array(["UL", "UR", "LL", "LR"])
     #      # https://stackoverflow.com/questions/17568612/how-to-make-numpy-argmax-return-all-occurrences-of-the-maximum
          winners = quarters[np.flatnonzero(np.array([UL, UR, LL, LR]) == np.max(np.array([UL, UR, LL, LR])))]

     # ####

          #print(winners)
     # # # if 2 quarters have high sums - px will become in the middle of the outer edge of a new kernel 
     # # # depending on the side of higher values, the px will be on one of the edges
     # # # if one quarter will have higher values - turning the kernel according to the location of the quarter 

     # #### conditioning for kernel edge size ####

          #h,w = PATH_1.shape[0],PATH_1.shape[1]
          
          #n2 = int(np.ceil(np.max(dist)))

          # edge cases (if x or y are too close to the edge 
          if (y+n2) > w or (x+n2) > h or (y-n2) < 0 or (x-n2) < 0:
               n2 = np.min((abs(0-x), (h-x), (w-y), abs(0-y)))
               # point in close upper right corner - very unlikely case 
               




     # ####

     # # potential combos - UL+LL, UR+LR, UL+UR, LL+LR, and all quarters separately 
     # flooring w even kernel can become uneven - heads-up!
     # # refacto try-excepts please :) 
          # edge cases
          
          #print(n2)
  
          try:
               if len(winners) == 2:
                    if 'UR' and 'LR' in winners:

                         if n2 == int(np.ceil(np.max(dist))):
                              kernel_2 = thinned[ x-(n2//2): x+(n2//2 ), y:(y+n2)]

                              # function input origin is quarter specific 
                              x_new = x + (find_nearest_white(kernel_2, [n2//2, 0])[0] - n2//2)
                              y_new = y + find_nearest_white(kernel_2, [n2//2, 0])[1]

                         else:
                              n3 = int(np.ceil(np.max(dist)))
                              if ((y-n3) < 0 and (x-n3) < 0) or ((x-n3) < 0):

                                   kernel_2 = thinned[ : (n2+n2//2), y : y+n3]

                                   x_new = x + (find_nearest_white(kernel_2, [n2, 0])[0] - n2)

                                   y_new = y + find_nearest_white(kernel_2, [n2, 0])[1] 
                              
                              elif ((x+n2) > h and (y-n3) < 0) or ( (x+n2) > h):

                                   kernel_2 = thinned[ : (n2+n2//2), y : y+n3]
                              



                    elif 'UL' and 'LL' in winners:
                         kernel_2 = thinned[ x-(n2//2) : x+(n2//2), (y-n2):y]

                         x_new = x + (find_nearest_white(kernel_2, [n2//2, n2])[0] - n2//2)
                         y_new = y + (find_nearest_white(kernel_2, [n2//2, n2])[1] - n2)
                         
                    elif 'LR' and 'LL' in winners:
                         kernel_2 = thinned[ x : (x+n2), y-(n2//2) :y + (n2//2)]

               

                         x_new = x + find_nearest_white(kernel_2, [0, n2//2])[0] 
                         y_new = y + (find_nearest_white(kernel_2, [0, n2//2])[1] - n2//2)

                    elif 'UR' and 'UL' in winners:
                         kernel_2 = thinned[ (x-n2) : x, y-(n2//2) :y + (n2//2)]

                         x_new = x + (find_nearest_white(kernel_2, [n2, n2//2])[0] - n2)
                         y_new = y + (find_nearest_white(kernel_2, [n2, n2//2])[1] - n2//2)


          # # could also use rhombus for individual quarters
          # # from skimage.draw import polygon 
          # # points: rhombus vertices 
          # # np.transpose(points)
          # # rr,cc = polgon(*points) - these are the coordinates for rhombi
          # # img[rr,cc] = 1

               elif len(winners) == 1: # 1 winner or 3 which is unlikely but still sth to look out for 
                    if 'UR' in winners:    
                         kernel_2 = thinned[ x-n2: x, y:y+n2]

                         x_new = x + (find_nearest_white(kernel_2, [n2, 0])[0] - n2)
                         y_new = y + find_nearest_white(kernel_2, [n2, 0])[1]


                    elif 'UL' in winners:
                         kernel_2 = thinned[ x-n2: x, y-n2:y]

                         x_new = x + (find_nearest_white(kernel_2, [n2, n2])[0] - n2)
                         y_new = y + (find_nearest_white(kernel_2, [n2, n2])[1] - n2)

                    elif 'LL' in winners:
                         kernel_2 = thinned[ x: x + n2, y-n2:y]
                    
                         x_new = x + find_nearest_white(kernel_2, [0, n2])[0] 
                         y_new = y + (find_nearest_white(kernel_2, [0, n2])[1] - n2)

                    elif 'LR' in winners:
                         kernel_2 = thinned[ x:x+n2, y:y+n2]
          
                         x_new = x + find_nearest_white(kernel_2, [0,0])[0] 
                         y_new = y + find_nearest_white(kernel_2, [0,0])[1] 

                         

               elif len(winners) == 3: # 3 winners has two cases - just creating bigger rectangles in that case 

                    if 'UL' and 'UR' and 'LL' in winners: 
                         kernel_2 = thinned[x-n2 : x+(n2//2), y-n2:y+(n2//2)]

                         x_new = x + (find_nearest_white(kernel_2, [n2,n2])[0] - n2)
                         y_new = y + (find_nearest_white(kernel_2, [n2,n2])[1] - n2)


                    elif 'UL' and 'UR' and 'LR' in winners: 
                         kernel_2 = thinned[x-n2: x+(n2//2), y-(n2//2) : y+n2]

                         x_new = x + (find_nearest_white(kernel_2, [n2,n2//2])[0] - n2)
                         y_new = y + (find_nearest_white(kernel_2, [n2,n2//2])[1] - n2//2)

                    elif 'UL' and 'LR' and 'LL' in winners: 
                         kernel_2 = thinned[x-(n2//2): x+n2, y-n2:y+(n2//2)]

                         x_new = x + (find_nearest_white(kernel_2, [n2//2,n2])[0] - n2//2)
                         y_new = y + (find_nearest_white(kernel_2, [n2//2,n2])[1] - n2)

                    elif 'LR' and 'UR' and 'LL' in winners: 
                         kernel_2 = thinned[x-(n2//2) : x+n2, y-(n2//2):y+n2]

                         x_new = x + (find_nearest_white(kernel_2, [n2//2, n2//2])[0] - n2//2)
                         y_new = y + (find_nearest_white(kernel_2, [n2//2, n2//2])[1] - n2//2)
               
               px_dist = dist[x_new][y_new]
               # values of this image from scale_obtain.py 
               nano_per_px = 400 / 22
               dm = int(2 * px_dist * nano_per_px)
               #dm_s.append((x, y, x_new, y_new, dm))
               dm_s.append(dm)
                    
               
          except: 
               # cases where kernel_2 only has 0-s need to probably create long slits along a specific edge :))
               # re-running the exceptions - get rids of most of them - for some reason unknown
               #exc_cases.append((x,y, winners))
               exc_cases.append((x,y))

     return (dm_s, exc_cases)

          #print(kernel_2)

#pt_s = point_picker()
pt_s = [(642, 611), (12, 951), (265, 3), (6, 825), (648, 960)]
#pt_s = [(392, 82), (601, 781), (467, 543), (606, 203), (124, 920), (639, 995), (480, 552), (23, 824), (522, 630), (550, 442), (410, 188), (388, 282), (626, 420), (344, 837), (126, 919), (369, 248), (59, 920), (392, 153), (311, 416), (620, 216), (42, 887), (435, 431), (245, 1003), (205, 183), (52, 831), (47, 841), (578, 785), (190, 47), (397, 47), (115, 476), (249, 670), (125, 1006), (517, 428), (45, 846), (550, 76), (109, 139), (453, 912), (178, 418), (146, 879), (492, 677), (127, 322), (479, 830), (179, 246), (537, 528), (514, 828), (503, 753), (46, 103), (161, 953), (549, 155), (297, 271), (253, 635), (325, 743), (138, 341), (171, 140), (135, 347), (369, 459), (492, 885), (406, 306), (259, 7), (470, 352), (382, 315), (516, 827), (150, 111), (407, 564), (42, 1014), (349, 775), (285, 413), (31, 228), (178, 953), (516, 377), (584, 276), (501, 812), (105, 678), (491, 8), (122, 467), (61, 344), (204, 416), (503, 985), (10, 365), (454, 311), (52, 831), (143, 285), (611, 684), (161, 754), (640, 941), (581, 164), (51, 811), (490, 211), (323, 531), (270, 1020), (151, 576), (465, 860), (434, 690), (178, 732), (284, 155), (514, 1010), (188, 178), (58, 206), (454, 810), (561, 996)]
#print(pt_s)
first_dm_s, first_excs = dm_finder(pt_s, n,n2,thinned)[0], dm_finder(pt_s, n,n2,thinned)[1]

print(first_excs)
#print("length_of_first_dm-s: ", len(first_dm_s))
leftovers = []
for val in first_excs:
     second_dm_s, second_excs = dm_finder([val], n,n2,thinned)[0], dm_finder([val], n,n2,thinned)[1]
     if len(second_dm_s) > 0: first_dm_s.append(*second_dm_s)
     elif len(second_excs) > 0: leftovers.append(*second_excs)
     # print("second round of dm-s ", second_dm_s)
     # print("second round of excs ", second_excs)

# print("second round of exceptions: ", len(second_excs))

# third_dm_s, third_excs = dm_finder(second_excs, n,n2,thinned)[0], dm_finder(second_excs, n,n2,thinned)[1]

# print("third round of exceptions", third_excs)


     # while len(excs) != 0:
     #      dm_s, excs = dm_finder(excs,n,n2,thinned)[0], dm_finder(excs,n,n2,thinned)[1]

     # print(dm_s)
     # print(len(dm_s))

     # print("exc", exc_cases)
     # print(len(exc_cases))
     # could include it in the function  - this computation is done to ensure that the end selection stays constant no matter the kernel size
     #continue

print(len(first_dm_s))
print(leftovers)
print("time taken:", time.time() - start_time)

# histogram creating
# import os 
# for k, v in os.environ.items():
# 	if k.startswith("QT_") and "cv2" in v:
# 	    del os.environ[k]

# plt.hist(first_dm_s)
# plt.title("Fiber diameter measurements (n=100)")
# plt.ylabel("Frequency")
# plt.xlabel("Fiber diameter (nm)")
# plt.show()

## for analysis

# with open("dm_info_2.txt", "w+") as file:
#      for val in first_dm_s:
#           file.write(f"{val}")
#           file.write("\n")
     
# with open("exc_cases_2.txt", "w+") as file:
#      for val in leftovers:
#           file.write(f"{val}")
#           file.write("\n")

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
#cv2.rectangle(thinned.astype(np.uint8), (y-(n//2), y+(n//2+1)), (x-(n2//2), x+(n2//2 +1)), (255,255,255),1)

# other graphical el-s

#cv2.line(thinned, (695,332), (695+find_nearest_white(kernel_2, (17,0))[0], 332+find_nearest_white(kernel_2, (17,0))[-1]), (255,255,255), 1)
#cv2.polylines(thinned, [points], isClosed=False, color = (255,255,255), thickness = 1)
#cv2.line(thinned, (695,332), (712,332), (255,255,255), 1)

# thinned = skimage.morphology.medial_axis(PATH_1).astype(np.uint8)
# thinned[thinned == 1] = 255

# orig = cv2.circle(orig, (y,x), radius=4, color=(0, 0, 255), thickness=-1)
# # orig = cv2.circle(orig, (y_new,x_new), radius=4, color=(255, 255, 255), thickness=-1)

# thinned = cv2.circle(thinned, (y,x), radius=4, color=(255, 255, 255), thickness=-1)
# # thinned = cv2.circle(thinned, (y_new,x_new), radius=4, color=(255, 255, 255), thickness=-1)
# # dist = cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
# # #print(np.unique(dist))
# # dist = cv2.circle(dist, (y,x), radius=4, color=(1, 1, 1), thickness=-1)
# # cv2.imshow('Distance Transform Image', dist)
# # # # # # #print(np.count_nonzero(PATH_1))
# # # # # # #cv2.imshow("direction", theta)
# # # # # # cv2.imshow("thresh", PATH_1)
# cv2.imshow("thinned", thinned.astype(np.uint8))
# cv2.imshow("orig", orig)
# cv2.waitKey(0)
# cv2.destroyAllWindows()