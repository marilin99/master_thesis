### Generated values from natural distro and checking if automated values are from the same distro ### 

import numpy as np
import os
from scipy import stats
import pandas as pd


# fetch values between ***diameter values*** and ***time taken*** from the txt file
ORIG_PATH = "/home/marilin/Documents/ESP/data/fiber_tests/synthesised_fibers/2_tone_fibers/"
RES_PATH = "/home/marilin/Documents/ESP/data/fiber_tests/synthesised_fibers/2_tone_fibers_autom_results/"
VIS_PATH = "/home/marilin/Documents/ESP/data/fiber_tests/synthesised_fibers/visuals/"

FILES = os.listdir(ORIG_PATH)
RES_FILES = os.listdir(RES_PATH)
#print(RES_FILES)

gen_t_p, gen_conf_low, gen_conf_high = [], [], []
measured_t_p, measured_conf_low, measured_conf_high  = [], [], []
f_p, h_p = [], []
gen_files, measured_files = [], []


for file_path in FILES:

    #start_time = time.time()
    file = ORIG_PATH+file_path


    if file.endswith(".txt"): #and re.search("(_2k_|_5k_|2k)", file) == None:

        lst = []
        with open(file) as f:
            for line in f:
                lst.append(line.strip("\n").split(":"))

        lst = sum(lst, [])
        dms_gen = np.array((lst[len(lst)-lst[::-1].index("***diameter values***"):]), dtype=np.uint0)

        ## t-test stats
        gen_t_p.append(stats.ttest_1samp(dms_gen, popmean=20).pvalue)
        gen_conf_low.append(stats.ttest_1samp(dms_gen, popmean=20).confidence_interval(confidence_level=0.99).low)
        gen_conf_high.append(stats.ttest_1samp(dms_gen, popmean=20).confidence_interval(confidence_level=0.99).high)

        #print(file)
        core = file.split("/")[-1].split(".txt")[0]

        gen_files.append(file_path.split(".txt")[0])

        for f_path in RES_FILES:

            f_res = RES_PATH+f_path

            if f_res.endswith(".txt") and core in f_res:

                lst = []

                with open(f_res) as f:

                    for line in f:
                        lst.append(line.strip("\n").split(":"))
                
                lst = sum(lst, [])
                
                dms_measured = np.array((lst[len(lst)-lst[::-1].index("***diameter values***"): lst.index("***time taken***")]), dtype = np.uint0)

                measured_files.append(f_path.split(".txt")[0])

                # The bounds of the 95% confidence interval are the minimum and maximum values of the parameter popmean for which the p-value of the test would be 0.05
                # t-test stats
                measured_t_p.append(stats.ttest_1samp(dms_measured, popmean=20).pvalue)
                measured_conf_low.append(stats.ttest_1samp(dms_measured, popmean=20).confidence_interval(confidence_level=0.99).low)
                measured_conf_high.append(stats.ttest_1samp(dms_measured, popmean=20).confidence_interval(confidence_level=0.99).high)
                # f-test (assumes that std.devs of populations are equal, considers pop.mean)
                f_p.append(stats.f_oneway(dms_gen, dms_measured).pvalue)
                # h-test (consider pop.median)
                h_p.append(stats.kruskal(dms_gen, dms_measured).pvalue)
       

#gen_files =  [string.split(".txt")[0] for string in FILES if string.endswith(".txt")]
#measured_files =  [string.split(".txt")[0] for string in RES_FILES if string.endswith(".txt")]

pd.DataFrame(data = {"T-test p value": list(zip(gen_t_p, measured_t_p)), "Low CI": list(zip(gen_conf_low, measured_conf_low)), \
                           "High CI":  list(zip(gen_conf_high, measured_conf_high)), "F-test p value":  (f_p), \
                            "H-test p value": (h_p)}, index=list(zip(gen_files, measured_files))).to_csv("/home/marilin/Documents/ESP/data/fiber_tests/synthesised_fibers/global_results.csv")

