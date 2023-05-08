import os
import matplotlib.pyplot as plt
import numpy as np
import re
import seaborn as sns
import pandas as pd
from natsort import natsorted
from glob import glob 

ORIG_PATH =  "/home/marilin/Documents/ESP/data/fiber_tests/seed_time_test/classical_results/"
#RES_PATH =  "/home/marilin/Documents/ESP/data/fiber_tests/seed_time_test/three_dm_fibers_autom_results/"
VIS_PATH = "/home/marilin/Documents/ESP/data/fiber_tests/seed_time_test/visuals/"
times = []
actions = ["Segmented image", "Thinned image", "Points picked", "Scale obtained", "Dm-s found"]


FILES = natsorted(os.listdir(ORIG_PATH))
#print(FILES)
#print(FILES)
#SUBS = [x[0] for x in os.walk(ORIG_PATH)]

# for path in natsorted(glob(ORIG_PATH+"*/", recursive = True)):
#     files = natsorted(os.listdir(path))
    #print(files)

for file_p in natsorted(FILES):
   
    file = ORIG_PATH+file_p
    #print(file)
    if file.endswith(".txt"): # and re.search("(_2k_|_5k_|2k)", file) == None:
        print(file)

        lst = []
        with open(file) as f:
            for line in f:
                lst.append(line.strip("\n").split(":"))

        lst = sum(lst, [])
        #lst = [float(val) if len(val.split(".")[0])>1 else float(val) * 1000 for val in lst[1:]]
        #lst.insert(0,"***diameter values***")

        #print(file,lst)

        dms_manual = lst[len(lst)-lst[::-1].index("***timestamps***"):lst.index("***coordinates***")]
        val_s = np.array([i.split(',')[0].strip("}") for i in dms_manual][1:], dtype=np.float32)
        times.append(val_s)


    

# for val in SUBS:
#     PATH = os.listdir(val)
#     if "sub" in PATH:
#         FOLDER_PATH = PATH + val
#         print(FOLDER_PATH)

# MAN1_FILES = os.listdir(MAN1_PATH)
# MAN2_FILES = os.listdir(MAN2_PATH)
#RES_FILES = os.listdir(RES_PATH)
# U_FILES =os.listdir(U_PATH)

#     for file_path in natsorted(val):

#         #start_time = time.time()
#         file = FILES+file_path


#         if file.endswith(".txt"): # and re.search("(_2k_|_5k_|2k)", file) == None:
#             print(file)
#             lst = []
#             with open(file) as f:
#                 for line in f:
#                     lst.append(line.strip("\n").split(":"))

#             lst = sum(lst, [])
#             #lst = [float(val) if len(val.split(".")[0])>1 else float(val) * 1000 for val in lst[1:]]
#             #lst.insert(0,"***diameter values***")

#             #print(file,lst)

#             dms_manual = lst[len(lst)-lst[::-1].index("***timestamps***"):lst.index("***coordinates***")]
#             val_s = np.array([i.split(',')[0].strip("}") for i in dms_manual][1:], dtype=np.float32)
#             #val_s = [float((i.split(',')[0]).split('}')[0]) if i == len(dms_manual)-1 else float(i.split(',')[0]) for i in dms_manual]
#             times.append(val_s)
#         #     if i[-1] == "}":
#         #         i[-1] = i[:-1]+""
#         # print(val_s)

        
#         #dms_m.append(dms_manual)

#print(times)
# ["2k", "5k",
#mag = ["2k", "5k", "10k", "15k","20k"]
import matplotlib.pylab as pylab
params = {
         'axes.labelsize': 14,
         'xtick.labelsize':14,
         'ytick.labelsize':14}
pylab.rcParams.update(params)

mag = ["15k", "5k", "2k", "20k", "10k"]
# # n = 0
# # for val in range(n, n+15, 15):
# means = []
j = 0

f,ax = plt.subplots(1) #plt.figure()
for i in range(0, 50, 10):
    mean = np.mean(times[i:i+10],axis=0)
    plt.plot(actions, np.array([0])+np.mean(times[i:i+10],axis=0), '-o', label=mag[j])
    j+=1


# #ax.set_ylim(ymin=0)

plt.ylabel("Cumulative average time (s)")
plt.xlabel("Action")

# #plt.title('Generated vs automatic measured dm-s')
# #plt.xlim((0,2000))
# #ax.set_xticks(bins)
plt.yscale("log")

plt.legend(fontsize=14)
plt.xticks(rotation=45)
# #plt.xlim(left=0)
#plt.show()

# # # figure = plt.gcf()  # get current figure
# # # figure.set_size_inches(18,12)
plt.savefig(f"{VIS_PATH}timeline_classical_2.png",bbox_inches="tight")
plt.clf()
    
# times = []
# actions = ["image segmentated", "image thinned", "points picked", "scale obtained", "dm found"]
# for val in dms_manual:
#     print(val.split(",")[0])

#[i.split(',')[0] for i in dms_manual] )

        #print(file,dms_manual)
        #core = file.split("/")[-1].split(".txt")[0]