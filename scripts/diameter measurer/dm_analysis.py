# hg, mode, median, std, mean, coefficient of variation - std/mean
# pandad df.to_latex() functionality

import numpy as np 
import pandas as pd 
import os 
import re
import matplotlib.pyplot as plt 

# fetch values between ***diameter values*** and ***time taken*** from the txt file 
#ORIG_PATH = "/home/marilin/Documents/ESP/data/fiber_tests/fiber_test_3/unet_results/"
ORIG_PATH = "/home/marilin/Documents/ESP/data/fiber_tests/fiber_test_1/unet_results/compound_results/"
#ORIG_PATH = "/home/marilin/Documents/ESP/data/fiber_tests/fiber_test_1/unet_results/compound_results/"


FILES = os.listdir(ORIG_PATH) 
d = {}
#"Values"
keys = ["File path", "Mean", "Median", "Mode", "Standard deviation", "Coefficient of variation"]
for k in keys:  d[k] = []

for file_path in FILES:
     
    #start_time = time.time()
    file = ORIG_PATH+file_path

    if file.endswith(".txt"): #and re.search("(_2k_|_5k_|2k)", file) == None:

        lst = []
        with open(file) as f:
            for line in f:
                lst.append(line.strip("\n").split(":"))

        lst = sum(lst, [])
        try:
            lst = [float(val) if len(val.split(".")[0])>1 else float(val) * 1000 for val in lst[1:]]
            lst.insert(0,"***diameter values***")
            dm_values = np.array((lst[len(lst)-lst[::-1].index("***diameter values***"): ]), dtype = np.uint0) #lst.index("***time taken***")
            #lst = [float(val) if len(val.split(".")[0])>1 else float(val) * 1000 for val in lst[1:]]
        except:
            continue

        d["File path"].append(file_path.split(".txt")[0])
        #d["Values"].append(dm_values)
        d["Mean"].append(int(np.mean(dm_values)))
        d["Median"].append(int(np.median(dm_values)))
        values, counts = np.unique(dm_values, return_counts=True)
        # gives the first, smallest mode value in case of two values of the same frequency, or the first smallest value if all values are unique
        d["Mode"].append(values[np.argmax(counts)])
        d["Standard deviation"].append(round(np.std(dm_values), 3))
        d["Coefficient of variation"].append(round(np.std(dm_values) / np.mean(dm_values), 3))

dataframe = pd.DataFrame(data = d)
#dataframe.to_csv(f"{ORIG_PATH}classical_analysis.csv", index=False, sep=",")
print(dataframe)

# # visualizing the df
# fig, ax = plt.subplots()
# dataframe.plot(ax=ax)
# plt.show() # plt in place of ax