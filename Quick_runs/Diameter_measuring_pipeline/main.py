### CODE FOR TESTING EXAMPLE SEM IMAGES when provided as arqument input ###

# libraries
import cv2
import numpy as np 
import time 
import os
import re
import sys 

## helper functions ##
from classical_segmenter import classical_segment
from scale_obtain import scale_obtain 
from thinner import thinner, thinner_2k_5k
from point_picker import point_picker
from dm_finder import dm_finder
from unet_pred import net_prediction

PATH_1 = str(sys.argv[1])
method = str(sys.argv[2])
points = int(sys.argv[3])

## CLASSICAL segmentation block ##
# otsu thresholding for 2k and 5k if specified in path

if method == "Classical":
    if re.search("(_2k_|_5k_|2k)", PATH_1) != None:
        
        segmented_im = cv2.threshold(cv2.imread(PATH_1, 0), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        dist, thinned = thinner_2k_5k(segmented_im)

    else:
        # classical segmentation for other magnifications
        segmented_im = classical_segment(PATH_1)

        dist, thinned = thinner(segmented_im)
    
### END OF CLASSICAL segmentation block ###

## U-NET segmentation ## 

elif method == "U-Net":
    if re.search("(_2k_|_5k_|2k)", PATH_1) != None:
        print("U-Net cannot handle 2k or 5k images right now, opting for Otsu's method instead")
        segmented_im = cv2.threshold(cv2.imread(PATH_1, 0), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        dist, thinned = thinner_2k_5k(segmented_im)
        
    else:
        segmented_im = net_prediction(PATH_1)
        dist, thinned = thinner(segmented_im)

else:
    print("OWNER NOTE: Make sure you have written the method correctly (either 'U-Net' or 'Classical') and that the arguments provided are in the correct order: abosolute file path, name of method, amount of points.")


### END OF U-NET segmentation block ##

# 100 measurements per image 
pt_s = point_picker(segmented_im, points)

# val, unit, px amount - from original image
scales = scale_obtain(PATH_1)

if scales[1] == "um":
    nano_per_px = int(scales[0]) * 1000 / int(scales[-1])
# scale = nm 
elif scales[1] == "nm": 
    nano_per_px = int(scales[0]) / int(scales[-1])


# leaving the lower part (scale included) in for now
h,w = segmented_im.shape[:2]

first_dm_s, first_excs, coords  = dm_finder(thinned, dist, segmented_im, pt_s, h,w,nano_per_px)

# prints out diameters 
print("Measured diameters", first_dm_s)

# visualizing diameters 

rgb_im = cv2.imread(PATH_1)

start_color = (0, 0, 0)
end_color = (255,0,0)  

for combo in coords:
    #combo = eval(combo)

    start_point = (combo[1], combo[0])
    mid_point = (combo[3], combo[2])

    # testing the actual end point
    end_point_x = (2 * mid_point[0]) - start_point[0]
    end_point_y = (2 * mid_point[1]) - start_point[1]

    end_point = (end_point_x, end_point_y)

    # px dist
    distance1 = np.sqrt((end_point[1] - start_point[1]) ** 2 + (end_point[0] - start_point[0]) ** 2)

    cv2.line(rgb_im, start_point, mid_point, start_color, 2)
    cv2.line(rgb_im, mid_point, end_point, (0,255,0), 2)

    cv2.circle(rgb_im, mid_point, 3, end_color, -1)


image = cv2.cvtColor(rgb_im, cv2.COLOR_RGB2BGR)

cv2.imshow(f'Original image with diameter lines', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#####
    

