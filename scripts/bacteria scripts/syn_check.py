import os.path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage import *
from skimage.morphology import skeletonize, medial_axis, disk
from skimage.feature import peak_local_max
import time 
from varname import argname
from natsort import natsorted

# uncomment this if showing image using other src than opencv 
# for k, v in os.environ.items():
# 	if k.startswith("QT_") and "cv2" in v:
# 	    del os.environ[k]

### control bac counter 
counter_2 = 0
def works_for_both(data):
    global counter_2
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4,4))
    ## green chan

    # clahe req grayscale img
    clahe2 = cv2.createCLAHE(clipLimit=1, tileGridSize=(20,20))
    cl2 = clahe2.apply(data)

    #green_chan_mean = cv2.blur(cl2, (3,3))
    # over 200 for level 
    # dilating in case the bac are small
    #eroded = cv2.erode(cl2, kernel)
    #dilated = cv2.dilate(eroded, kernel)
    #print(np.unique(dilated))
    #thresh = cv2.adaptiveThreshold(dilated, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 30)

    #_, thresh = cv2.threshold(cl2, 200, 255,cv2.THRESH_TOZERO)

    # for one stack 
    _, thresh = cv2.threshold(data, 0, 255, cv2.THRESH_TOZERO+cv2.THRESH_OTSU)

    #_, thresh = cv2.threshold(data, 200, 255,cv2.THRESH_TOZERO)

    

    # works for 1024x1024
    # PI_1 - 200, PI_2 - 100, PI_3 - 140, PI_4 - 200, PI_5 - 220, PI_6 - 110, PI_7 - 200
    #_, thresh = cv2.threshold(data, 200, 255,cv2.THRESH_TOZERO)

    # https://github.com/pwwang/python-varname
    # change condition if needed - type based on variable name 
    if 'green' in argname("data"):
        ty = "green"

    elif 'red' in argname("data"):
        ty = "red"

    #cv2.imshow(f"data_{counter_2}", data)
    #cv2.imshow("normalized", cl2)
    #cv2.imshow(f"dilated_counter{counter_2}", dilated)
    #cv2.imshow(f"thresholded_{counter_2}", thresh)
    counter_2+=1

    return thresh, ty

counter = 0
base_im_r, base_im_g = np.zeros((512,512), dtype=np.uint8), np.zeros((512,512), dtype=np.uint8)
# https://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array/3689710#3689710

def peaker(data,ty):
    global counter, base_im_r, base_im_g
    # define a disk shape - 4 for red channel, 3 for green channel
    # no disk shape edit needed here

    if ty == "green":
        neighborhood = disk(7) # 3 w 512x512 - w 1024 needs +4 size disk?

    elif ty == "red":
        neighborhood = disk(8)

    #apply the local maximum filter; all pixel of maximal value 
    #in their neighborhood are set to 1

    local_max = maximum_filter(data, footprint=neighborhood) == data

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

    #### 2nd iteration ####

    #we create the mask of the background
    background = (new_im==0)

    #a little technicality: we must erode the background in order to 
    #successfully subtract it form local_max, otherwise a line will 
    #appear along the background border (artifact of the local maximum filter)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)
    local_max = maximum_filter(new_im, footprint=neighborhood) == new_im
    #we obtain the final mask, containing only peaks, 
    #by removing the background from the local_max mask (xor operation)
    detected_peaks = local_max ^ eroded_background

    new_im = np.multiply(new_im, detected_peaks)

    #############

    new_im[new_im>0] = 255
    # contours, hierarchy = cv2.findContours(new_im, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # print(contours)
    if ty == "red":
        base_im_r += new_im
    elif ty == "green":
        base_im_g += new_im

    #cv2.imshow(f"detected_peaks_{counter}", new_im)

    counter+=1
    return detected_peaks

## SYTO-PI case study ##
## Q1: What is % of bacterial cells that stain with PI - red (dead cells) - green/red ratio?
## Q2: Is there a difference in bacterial viability depending on fiber material? - this is only about PCL-PEO I think

## check what is in the folder - unique names - no growth fibers, 24h fibers, control 
start_time = time.time()

# one stack of 24h growth 
PATH = "/home/marilin/Documents/ESP/data/SYTO_PI_conversion/"
FILES = os.listdir(PATH) 

region_red = {}
region_green = {}
unique_files = []
counter_red = 0
counter_green = 0
FIL_FILES = []
bac_counting = {}

#print(natsorted(FILES))

# # filtering out png and not control values 
# for f in natsorted(FILES):
#     # # created xml from core files, can fetch the unique files from there 
#     # if f.endswith(".xml"):
#     #     unique_file = f.split(".xml")[0]
#     #     unique_files.append(unique_file)
    
#     #for val in unique_files:
#         # control filtered out for now 
#     if f.endswith("png"):
#         FIL_FILES.append(f)

# global_core_name = None

# # tmp sol - ideally should have a mark that ends the iteration or use the piece of code
# # adding a pseudo file to mark the end of the list 
# FIL_FILES.append("_".join((natsorted(list(set(FIL_FILES)))[-1]).split("_")[:-3])+"_zzz.png")

# # using set to remove duplicates 
# for f in natsorted(list(set(FIL_FILES))):

#     #if f.startswith("stack_fibers_24h_growth_syto_PI"):
       
#     file = PATH+f
#     core_name = "_".join(f.split("_")[:-2])


#         # gathering data and resetting after every new set of stacks, considering first round
 
#     if core_name != global_core_name:
#         if global_core_name != None:
      
#             bac_counting[f"{global_core_name}_red"] = label(base_im_r)[1]
#             bac_counting[f"{global_core_name}_green"] = label(base_im_g)[1]

#             base_im_r, base_im_g = np.zeros((512,512), dtype=np.uint8), np.zeros((512,512), dtype=np.uint8)

#             #### saving detected peaks data to text file #####
#             # with open(f"{PATH}{global_core_name}_regions_r.txt", "w+") as file_m:
#             #     for key, val in region_red.items():
#             #         file_m.write(f"{key}: {val}")
#             #         file_m.write("\n")
            
#             # with open(f"{PATH}{global_core_name}_regions_g.txt", "w+") as file_l:
#             #     for key, val in region_green.items():
#             #         file_l.write(f"{key}: {val}")
#             #         file_l.write("\n")

#         # resetting var name
#         global_core_name = core_name

  
#         # final case
  
#         # global_core_name!= None and ("_".join(global_core_name.split("_")[:-1])+"_"+str(int(global_core_name.split("_")[-1])+1)) not in unique_files:
 
PATH = "/home/marilin/Documents/ESP/data/bacteria_tests/synthesised_images/"
#PATH = "/home/marilin/Documents/ESP/data/SYTO_PI_conversion/"
FILES = os.listdir(PATH)

for f in FILES:
    file = PATH+f
    #print(file)

    if file.endswith(".png") and "red" in file:
        orig_red = cv2.imread(file,0)
        shape_red = orig_red.shape
        red_chan = cv2.resize(orig_red, (512,512))
        
        # centrosymmetric structure
        labeled_array, num_features_r = label(peaker(*works_for_both(red_chan)))
        uniq_vals = np.unique(labeled_array.flatten())

        #print(np.unique(labeled_array.flatten())!=0)
        ## for the cases where the bacteria is layered on top of each other 
        # for val in uniq_vals[uniq_vals!=0]:
        #     idxs = np.argwhere(labeled_array==val)
        #     start_point = idxs[0]
        #     end_point = idxs[-1]
        #     anchor_point = start_point + (end_point-start_point)//2
        #     area = len(np.argwhere(labeled_array==val))
        #     region_red[f"region_{counter_red}"] = []
        #     # areas taken up, starting point of area, last coordinate of area, anchor_point, area in amount of px-s
        #     region_red[f"region_{counter_red}"].extend([idxs, start_point, end_point, anchor_point, area])
        #     counter_red+=1

        #print(np.nonzero(labeled_array))
        print(file, num_features_r)

    elif file.endswith(".png") and "green" in file: 

        orig_green = cv2.imread(file,0)
        shape_green = orig_green.shape
        green_chan = cv2.resize(orig_green, (512,512))
        labeled_array, num_features_g = label(peaker(*works_for_both(green_chan)))
        print(file, num_features_g)

        uniq_vals = np.unique(labeled_array.flatten())

        #print(np.unique(labeled_array.flatten())!=0)
        ## for the cases where the bacteria is layered on top of each other 
        # for val in uniq_vals[uniq_vals!=0]:
        #     idxs = np.argwhere(labeled_array==val)
        #     start_point = idxs[0]
        #     end_point = idxs[-1]
        #     anchor_point = start_point + (end_point-start_point)//2
        #     area = len(np.argwhere(labeled_array==val))
        #     region_green[f"region_{counter_green}"] = []
        #     region_green[f"region_{counter_green}"].extend([idxs, start_point, end_point, anchor_point, area])
        #     counter_green+=1

        # breaking the loop when reached the end of the file list 
        # else: 
        #     break


# ## red chan 
# #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
# #red_chan_mean = cv2.blur(red_chan, kernel)


#print(regions)
print("time it took: ", time.time()-start_time)
print(bac_counting)



###########################################################################
## mahotas testing - regional max functionality
# import pylab
# import mahotas as mh

# # http://pythonvision.org/basic-tutorial/

# #red_gaus = mh.mean_filter(red_chan, 3)
# rmax = mh.regmax(thresh_r)
# # pylab.imshow(mh.overlay(red_dilated, rmax))
# # pylab.show()
# # pylab.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# # pylab.show()
# seeds,nr_nuclei = mh.label(rmax)
# #print(nr_nuclei)

# # needs some watershedding 
# #_, thresh_red = cv2.threshold(thresh_r, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# T = mh.thresholding.otsu(thresh_r)
# dist = mh.distance(thresh_r > T)
# dist = dist.max() - dist
# dist -= dist.min()
# dist = dist/float(dist.ptp()) * 255
# dist = dist.astype(np.uint8)
# #pylab.imshow(cv2.cvtColor(dist, cv2.COLOR_BGR2RGB))
# #pylab.show()
# nuclei = mh.cwatershed(dist, seeds)
# # pylab.imshow(nuclei)
# # pylab.show()

# whole = mh.segmentation.gvoronoi(nuclei)


##


## local intensity maxima 
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
# neighborhood_size = 3 
# threshold = 30 


# data_max = filters.maximum_filter(red_chan_mean, neighborhood_size)
# maxima = (red_chan_mean == data_max)
# data_min = filters.minimum_filter(red_chan_mean, neighborhood_size)
# diff = ((data_max - data_min) > threshold)
# maxima[diff == 0] = 0
# labeled, num_objects = ndimage.label(maxima)
# slices = ndimage.find_objects(labeled)
##

### visualisation ###
#cv2.imshow("orig", img)
# cv2.imshow("red_chan", red_chan)
# cv2.imshow("green_chan", green_chan)
#cv2.imshow("merged_im", base_im)
# # # cv2.imshow("red_chan_mean", cl2)
# # # cv2.imshow("red_dilated", green_dilated)
# # # cv2.imshow("red_thresh", thresh_g)
cv2.waitKey(0)
cv2.destroyAllWindows()

