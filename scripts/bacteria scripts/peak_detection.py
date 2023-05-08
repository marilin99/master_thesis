import numpy as np
from scipy import ndimage
from scipy.ndimage import *
from skimage.morphology import disk
import cv2


# https://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array/3689710#3689710
def peaker(data,ty):

    if ty == "green":
        neighborhood = disk(7) # 3 w 512x512 - w 1024 needs +4 size disk?

    elif ty == "red":
        neighborhood = disk(8)

    local_max = maximum_filter(data, footprint=neighborhood)  == data

    # cv2.imshow("eroded_bg", np.uint8(local_max))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    #local_max is a mask that contains the peaks we are 
    #looking for, but also the background.

    #In order to isolate the peaks we must remove the background from the mask.

    #we create the mask of the background
    background = (data==0)
    

    #a little technicality: we must erode the background in order to 
    #successfully subtract it form local_max, otherwise a line will 
    #appear along the background border (artifact of the local maximum filter)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)


    #we obtain the final mask, containing only peaks, 
    #by removing the background from the local_max mask (xor operation)
    detected_peaks = local_max ^ eroded_background

    new_im = np.multiply(data, detected_peaks)

    # cv2.imshow("eroded_bg", np.uint8(new_im))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    

    tmp_im = new_im.copy()

    #############

    new_im[new_im>0] = 255
    
    return detected_peaks, new_im, tmp_im


if __name__ == "__main__":
     im = cv2.imread("/home/marilin/Documents/ESP/thesis_visuals/thresh_pi_red_26.png",0)
     peaker(im, "red")
#      cv2.imshow("thresh", works_for_both(im)[0])
#      cv2.waitKey(0)
#      cv2.destroyAllWindows()