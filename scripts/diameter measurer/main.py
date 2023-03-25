import cv2
import numpy as np 
import matplotlib.pyplot as plt 
import time 
import os

## helper functions ##
from classical_segmenter import classical_segment
from scale_obtain import scale_obtain 
from thinner import thinner 
from point_picker import point_picker
from dm_finder import dm_finder



# this image is segmented using statistical region merging from diameterj imagej - from 650 is the scale box 
#PATH_1 = cv2.imread("/home/marilin/Documents/ESP/diameterJ_test/sem_test/Segmented Images/EcN_II_PEO_131120_GML_15k_01_S1_reverse.tif",0)[:650, :]

TARGET_PATH = "/home/marilin/Documents/ESP/data/fiber_tests/results/"

ORIG_PATH = "/home/marilin/Documents/ESP/data/fiber_tests/original_img/"

FILES = os.listdir(ORIG_PATH) 

for f in FILES:
     
    start_time = time.time()
    PATH_1 = ORIG_PATH+f
    print("**********************")
    print(PATH_1)

    # segmentation 
    segmented_im = classical_segment(PATH_1)
    print("time taken for segmentation classically", time.time() - start_time)

    # for u-net do not forget to median blur w 15 the segmented im first (already done in the classical approach)

    # leaving the lower part (scale included) in for now
    h,w = segmented_im.shape[:2]

    dist, thinned = thinner(segmented_im)
    # cumulative time 
    print("time taken for thinning", time.time() - start_time)

    pt_s = point_picker(segmented_im, 100)
    # cumulative time
    print("time taken for point picking", time.time() - start_time)

    # val, unit, px amount - from original image
    scales = scale_obtain(PATH_1)
    print("time taken for scale obtaining", time.time() - start_time)

    print(scales)
    try:
        if scales[1] == "um":
            nano_per_px = int(scales[0]) * 1000 / int(scales[-1])
        # scale = nm 
        elif scales[1] == "nm": 
            nano_per_px = int(scales[0]) / int(scales[-1])
    except Exception: 
        core_name = f.split(".tif")[0]
        with open(f"{TARGET_PATH}{core_name}.txt", "w+") as file_ex:
            file_ex.write("Scaling off, check the image")
        continue



    first_dm_s, first_excs = dm_finder(thinned, dist, segmented_im, pt_s, h,w,nano_per_px)[:2]
    print("time taken for dm finding", time.time()-start_time)

    ## saving values + time to a txt file ##
    # splitting from tif assuming that tif is still in the file name

    core_name = f.split(".tif")[0]
    with open(f"{TARGET_PATH}{core_name}.txt", "w+") as file:

        file.write("***diameter values***")
        file.write("\n")
        for val in first_dm_s:
            file.write(f"{val}")
            file.write("\n")

        file.write(f"***time taken***: {time.time() - start_time}")
        file.write("\n")
        file.write("***exceptions***")
        file.write("\n")
        for val in first_excs:
            file.write(f"{val}")
            file.write("\n")

    print("time taken", (time.time() - start_time))
    start_time = time.time()

    ########

    for k, v in os.environ.items():
        if k.startswith("QT_") and "cv2" in v:
            del os.environ[k]
        
    ## saving 5 bin histogram ##
    plt.hist(first_dm_s, bins = 5)
    plt.title("Fiber diameter measurements (n=100)")
    plt.ylabel("Frequency")
    plt.xlabel("Fiber diameter (nm)")
    plt.savefig(f"{TARGET_PATH}{core_name}.png")
    plt.clf()

    #########
    print(first_dm_s)
    print(first_excs)

