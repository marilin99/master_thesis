# hg, mode, median, std, mean, coefficient of variation - std/mean
# pandad df.to_latex() functionality

import numpy as np 
import pandas as pd 
import os 
import re
import matplotlib.pyplot as plt 
import cv2

# fetch values between ***diameter values*** and ***time taken*** from the txt file 
ORIG_PATH = "/home/marilin/Documents/ESP/data/fiber_tests/fiber_test_2/classical_results_2/"

IMG_PATH = "/home/marilin/Documents/ESP/data/fiber_tests/fiber_test_2/original_data_2/"
FILES = os.listdir(ORIG_PATH) 
# d = {}
# #"Values"
# keys = ["File path", "Mean", "Median", "Mode", "Standard deviation", "Coefficient of variation"]
# for k in keys:  d[k] = []

        
for file_path in FILES:
     
    #start_time = time.time()
    file = ORIG_PATH+file_path
    #print(file_path)

    if file.endswith(".txt"): #and re.search("(_2k_|_5k_|2k)", file) == None:

        lst = []
        with open(file) as f:
            for line in f:
                lst.append(line.strip("\n").split(":"))

        lst = sum(lst, [])
        
        coords = np.array((lst[len(lst)-lst[::-1].index("***coordinates***"):]))
      


 
        core_f  = file_path.split(".txt")[0]
        print(f"{IMG_PATH+core_f}.tif")
        rgb_im = cv2.imread(f"{ORIG_PATH+core_f}_thinned.png")

        #start_color = (0, 0, 0)
        start_color = (255, 255, 255)  # white
        end_color = (255,0,0)  # Red for end point

        for combo in coords:
            combo = eval(combo)

            start_point = (combo[1], combo[0])
            end_point = (combo[3], combo[2])

            cv2.line(rgb_im, start_point, end_point, start_color, 2)
            cv2.circle(rgb_im, end_point, 3, end_color, -1)

    
        image = cv2.cvtColor(rgb_im, cv2.COLOR_RGB2BGR)

        # Display the image with lines
        cv2.imshow('Image with Gradient Lines', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
                


# pts = [((596, 57), (604, 48)), ((124, 664), (130, 658))]
# line_color = (0, 255, 0) 
# line_thickness = 2
# image = cv2.imread("/home/marilin/Documents/ESP/data/fiber_tests/fiber_test_2/original_data_2/PCL_MSN_CAM_10k_12031901.tif")
# # for pt1, pt2 in pts:
# #     cv2.line(image, pt1, pt2, line_color, line_thickness)

# # Create an array of points for the lines
# #lines = [((10, 10), (100, 100)), ((200, 200), (300, 300)), ((400, 400), (500, 500))]

# # Draw each line with gradient color on the image
# for line in lines:
#     start_point = tuple(line[0])
#     end_point = tuple(line[1])
#     # Define the gradient colors for start and end points
#     start_color = (255, 255, 255)  # Green for starting point
#     end_color = (255,0,0)  # Red for end point
#     # Draw the line on the image with gradient color
#     cv2.line(image, start_point, end_point, start_color, 2)
#     # Draw a small filled circle at the end point with the end color
#     cv2.circle(image, end_point, 3, end_color, -1)

# # Convert the image back to BGR color format
# image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

# # Display the image with lines
# cv2.imshow('Image with Gradient Lines', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# Display the image with lines
# cv2.imshow('Image with Lines', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#         d["File path"].append(file_path.split(".txt")[0])
#         #d["Values"].append(dm_values)
#         d["Mean"].append(int(np.mean(dm_values)))
#         d["Median"].append(int(np.median(dm_values)))
#         values, counts = np.unique(dm_values, return_counts=True)
#         # gives the first, smallest mode value in case of two values of the same frequency, or the first smallest value if all values are unique
#         d["Mode"].append(values[np.argmax(counts)])
#         d["Standard deviation"].append(round(np.std(dm_values), 3))
#         d["Coefficient of variation"].append(round(np.std(dm_values) / np.mean(dm_values), 3))

# dataframe = pd.DataFrame(data = d)
# dataframe.to_csv(f"{ORIG_PATH}classical_analysis.csv", index=False, sep=",")
# print(dataframe)

# # visualizing the df
# fig, ax = plt.subplots()
# dataframe.plot(ax=ax)
# plt.show() # plt in place of ax