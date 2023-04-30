### Generated values from natural distro and checking if automated values are from the same distro ### 
### also for comparing against the human measurements ## 

import numpy as np
import os
from scipy import stats
import pandas as pd


# fetch values between ***diameter values*** and ***time taken*** from the txt file
ORIG_PATH = "/home/marilin/Documents/ESP/data/fiber_tests/synthesised_fibers/one_dm_fibers/"
#RES_PATH = "/home/marilin/Documents/ESP/data/fiber_tests/synthesised_fibers/2_tone_fibers_manual/manual_2/"
RES_PATH = "/home/marilin/Documents/ESP/data/fiber_tests/synthesised_fibers/one_dm_fibers_autom_results/"

FILES = os.listdir(ORIG_PATH)
RES_FILES = os.listdir(RES_PATH)
#print(RES_FILES)

gen_t_p, gen_conf_low, gen_conf_high = [], [], []
measured_t_p, measured_conf_low, measured_conf_high  = [], [], []
f_p, h_p = [], []
gen_files, measured_files = [], []
per_error = {}


for file_path in FILES:

    #start_time = time.time()
    file = ORIG_PATH+file_path
    

    if file.endswith(".txt"): #and re.search("(_2k_|_5k_|2k)", file) == None:

        lst = []
        tmp = []
        with open(file) as f:
            for line in f:
                lst.append(line.strip("\n").split(":"))

        lst = sum(lst, [])
        dms_gen = np.array((lst[len(lst)-lst[::-1].index("***diameter values***"):]), dtype=np.int16)

        ## t-test stats
        # gen_t_p.append(stats.ttest_1samp(dms_gen, popmean=20).pvalue)
        # gen_conf_low.append(stats.ttest_1samp(dms_gen, popmean=20).confidence_interval(confidence_level=0.99).low)
        # gen_conf_high.append(stats.ttest_1samp(dms_gen, popmean=20).confidence_interval(confidence_level=0.99).high)

        #print(file)
        core = file.split("/")[-1].split(".txt")[0]

        print(file)

        gen_files.append(file_path.split(".txt")[0])

        for f_path in RES_FILES:

            f_res = RES_PATH+f_path

            if f_res.endswith(".txt") and file_path in f_res:
                print(f_res)

                lst = []

                with open(f_res) as f:

                    for line in f:
                        lst.append(line.strip("\n").split(":"))
                
                lst = sum(lst, [])
                # lst = [float(val) if len(val.split(".")[0])>1 else float(val) * 1000 for val in lst[1:]]
                # lst.insert(0,"***diameter values***")
                
                dms_measured = np.array((lst[len(lst)-lst[::-1].index("***diameter values***"): lst.index("***time taken***")]), dtype = np.int16)
          

                measured_files.append(f_path.split(".txt")[0])
                
                # percent error
                for i,val in enumerate(dms_measured):
                    tmp.append(np.abs( (val - dms_gen[i]) / dms_gen[i] ) * 100 )
                per_error[f"{core}"] = np.mean(tmp)

#                 # The bounds of the 95% confidence interval are the minimum and maximum values of the parameter popmean for which the p-value of the test would be 0.05
#                 # t-test stats
#                 measured_t_p.append(stats.ttest_1samp(dms_measured, popmean=20).pvalue)
#                 measured_conf_low.append(stats.ttest_1samp(dms_measured, popmean=20).confidence_interval(confidence_level=0.99).low)
#                 measured_conf_high.append(stats.ttest_1samp(dms_measured, popmean=20).confidence_interval(confidence_level=0.99).high)
#                 # f-test (assumes that std.devs of populations are equal, considers pop.mean)
#                 f_p.append(stats.f_oneway(dms_gen, dms_measured, equal_var=False).pvalue)
#                 # h-test (consider pop.median)
#                 h_p.append(stats.kruskal(dms_gen, dms_measured).pvalue)

# #gen_files =  [string.split(".txt")[0] for string in FILES if string.endswith(".txt")]
# #measured_files =  [string.split(".txt")[0] for string in RES_FILES if string.endswith(".txt")]

# print(pd.DataFrame(data = {"T-test p value": list(zip(gen_t_p, measured_t_p)), "Low CI": list(zip(gen_conf_low, measured_conf_low)), \
#                            "High CI":  list(zip(gen_conf_high, measured_conf_high)), "F-test p value":  (f_p), \
#                             "H-test p value": (h_p)}, index=list(zip(gen_files, measured_files)))) #.to_csv("/home/marilin/Documents/ESP/data/fiber_tests/synthesised_fibers/stat_results_man2.csv")


import matplotlib.pyplot as plt

for k, v in os.environ.items():
    if k.startswith("QT_") and "cv2" in v:
        del os.environ[k]

reordered_keys = sorted(per_error.keys())
reordered_dict = {k: per_error[k] for k in reordered_keys}

# for k, v in reordered_dict.items():
#     plt.plot(v, label=k)

print(reordered_dict)
#print(per_error)

# for i in range(len(per_error)):
#     plt.plot(per_error[i], label=list(zip(gen_files, measured_files))[i])

x = np.array([10,20,30], dtype=np.uint8)

plt.plot(x,list(reordered_dict.values())[:3],label = "straight lines unordered")
plt.plot(x,list(reordered_dict.values())[5:8],label = "curves unordered")
#plt.xlim(5,45)¤plt.xticks([10,20,30,40])

plt.xlabel("Line diameter (px)")
plt.ylabel("Percent error (%)")
plt.legend()
plt.show()



    
# saving 5 bin histogram ##
# plt.plot([5.75000000e+01, 0.00000000e+00, 1.00000000e+02, 4.75000000e+01,
#        8.75000000e+01, 2.50000000e+00, 2.00000000e+01, 6.25000000e+01,
#        0.00000000e+00, 1.02500000e+02, 5.50000000e+01, 2.50000000e+00,
#        2.50000000e+00, 1.00000000e+01, 5.00000000e+00, 4.61168602e+19,
#        5.00000000e+00, 4.25000000e+01, 2.75000000e+01, 4.75000000e+01,
#        4.61168602e+19, 3.25000000e+01, 5.50000000e+01, 0.00000000e+00,
#        4.61168602e+19, 4.25000000e+01, 1.10000000e+02, 0.00000000e+00,
#        7.50000000e+01, 0.00000000e+00])
# plt.show()
# plt.hist(first_dm_s, bins = 5)
# plt.title("Fiber diameter measurements (n=100)")
# plt.ylabel("Frequency")
# plt.xlabel("Fiber diameter (nm)")
# plt.savefig(f"{TARGET_PATH}{core_name}.png")
# plt.clf()
