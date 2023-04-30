import cv2
import numpy as np 
import matplotlib.pyplot as plt 
import time 
import os
import re

## helper functions ##
from classical_segmenter import classical_segment
#from scale_obtain import scale_obtain 
from thinner import thinner, thinner_2k_5k
from point_picker import point_picker
from dm_finder import dm_finder
from unet_pred import net_prediction


TARGET_PATH = "/home/marilin/Documents/ESP/data/fiber_tests/synthesised_fibers/three_dm_fibers_autom_results_500/" 

ORIG_PATH = "/home/marilin/Documents/ESP/data/fiber_tests/synthesised_fibers/three_dm_fibers_sub/"

#ORIG_PATH = "/run/user/1000/gvfs/smb-share:server=kivi.ut.ee,share=tensonilabor/Georg L/Fibre diameter_Marilin/Co-Axial  SEM PVA-Pleu/"

FILES = os.listdir(ORIG_PATH) 

# testing with no conditions for the pt picker 
#FILES = ["/home/marilin/Documents/ESP/data/fiber_tests/fiber_test_1/original_img/"]

for f in FILES:
     if f.endswith(".png") and "unordered" in f:
          start_time = time.time()
          PATH_1 = ORIG_PATH+f
          core_name = f.split(".png")[0]

          print("**********************")
          print(PATH_1)

          ## CLASSICAL SEGMENTATION ##
          # otsu thresholding for 2k and 5k if specified in path
          # if re.search("(_2k_|_5k_|2k)", PATH_1) != None:
          #      continue
          #      # segmented_im = cv2.threshold(cv2.imread(PATH_1, 0), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
          #      # print("time taken for otsu segementation", time.time() - start_time)
          
          #      # dist, thinned = thinner_2k_5k(segmented_im)
          
          # else:
          #      # classical segmentation for other magnifications
          #      segmented_im = classical_segment(PATH_1)
          #      cv2.imwrite(f"{TARGET_PATH}{core_name}_segmented.png", segmented_im)
          #      print("time taken for segmentation classically", time.time() - start_time)

          #      dist, thinned = thinner(segmented_im)
          #      cv2.imwrite(f"{TARGET_PATH}{core_name}_thinned.png", np.uint8(thinned))

          # print("time taken for thinning", time.time() - start_time)
          
          ## U-NET segmentation ##

          # this might skip 15k - needs a better way to assess this
          if re.search("(_2k_|_5k_|2k)", PATH_1) != None:
              continue
          else:
              #segmented_im = net_prediction(PATH_1)
              segmented_im = cv2.imread(PATH_1,0)
              print("time taken for unet segmentation", time.time() - start_time)
              dist, thinned = thinner(segmented_im)
              print("time taken for thinning", time.time() - start_time)

          pt_s = point_picker(segmented_im, 750)
          # cumulative time
          print("time taken for point picking", time.time() - start_time)

          # val, unit, px amount - from original image
          #     scales = scale_obtain(PATH_1)
          #     print("time taken for scale obtaining", time.time() - start_time)

          #     print(scales)
          #     try:
          #         if scales[1] == "um":
          #             nano_per_px = int(scales[0]) * 1000 / int(scales[-1])
          #         # scale = nm 
          #         elif scales[1] == "nm": 
          #             nano_per_px = int(scales[0]) / int(scales[-1])
          #     except Exception: 
          
          #         with open(f"{TARGET_PATH}{core_name}.txt", "w+") as file_ex:
          #             file_ex.write("Scaling off, check the image")
          #         continue


          # leaving the lower part (scale included) in for now
          h,w = segmented_im.shape[:2]

          first_dm_s, first_excs, coords  = dm_finder(thinned, dist, segmented_im, pt_s, h,w, 1.0) #[:3]
          print("time taken for dm finding", time.time()-start_time)

          ## saving values + time to a txt file ##
          # splitting from tif assuming that tif is still in the file name

          
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
               file.write("***coordinates***")
               file.write("\n")
               for val in coords:
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
          plt.title("Fiber diameter measurements (n=750)")
          plt.ylabel("Frequency")
          plt.xlabel("Fiber diameter (pixels)")
          plt.savefig(f"{TARGET_PATH}{core_name}.png")
          plt.clf()

          #########
          print(first_dm_s)
          print(first_excs)

