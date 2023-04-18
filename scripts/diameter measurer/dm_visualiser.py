# hg, mode, median, std, mean, coefficient of variation - std/mean
# pandad df.to_latex() functionality

import numpy as np 
import pandas as pd 
import os 
import re
import matplotlib.pyplot as plt 
import cv2

# fetch values between ***diameter values*** and ***time taken*** from the txt file 
ORIG_PATH = "/home/marilin/Documents/ESP/data/fiber_tests/synthesised_fibers/2_tone_fibers_autom_results/"

IMG_PATH = "/home/marilin/Documents/ESP/data/fiber_tests/synthesised_fibers/2_tone_fibers/"
VIS_PATH = "/home/marilin/Documents/ESP/data/fiber_tests/synthesised_fibers/visuals/"
FILES = os.listdir(ORIG_PATH) 

        
for file_path in FILES:
     
    #start_time = time.time()
    file = ORIG_PATH+file_path
    #print(file_path)

    if file.endswith(".txt") and re.search("(_2k_|_5k_|2k)", file) == None:

        lst = []
        with open(file) as f:
            for line in f:
                lst.append(line.strip("\n").split(":"))

        lst = sum(lst, [])
        
        coords = np.array((lst[len(lst)-lst[::-1].index("***coordinates***"):]))
      


 
        core_f  = file_path.split(".txt")[0][:-3] #-3 syn fib specific

        #print(core_f)
        #print(f"{IMG_PATH+core_f}.png")
        # original images in .tif

        rgb_im = cv2.imread(f"{IMG_PATH+core_f}.png")

        start_color = (0, 0, 0)
        #start_color = (255, 255, 255)  # white
        end_color = (255,0,0)  # Red for end point

        for combo in coords:
            combo = eval(combo)

            start_point = (combo[1], combo[0])
            mid_point = (combo[3], combo[2])

            # testing the actual end point
            end_point_x = (2 * mid_point[0]) - start_point[0]
            end_point_y = (2 * mid_point[1]) - start_point[1]

            end_point = (end_point_x, end_point_y)

            # px dist
            distance1 = np.sqrt((end_point[1] - start_point[1]) ** 2 + (end_point[0] - start_point[0]) ** 2)
            
            #print((start_point, end_point, end_point2), distances)

            cv2.line(rgb_im, start_point, mid_point, start_color, 2)
            cv2.line(rgb_im, mid_point, end_point, (0,255,0), 2)

            cv2.circle(rgb_im, mid_point, 3, end_color, -1)

    
        image = cv2.cvtColor(rgb_im, cv2.COLOR_RGB2BGR)

        #cv2.imshow('Image with Gradient Lines', image)
        # cv2.imwrite(f"{VIS_PATH+core_f}_start_end.png", image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
                