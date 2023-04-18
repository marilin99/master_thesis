###  statistical ### 

import numpy as np
import os
from scipy import stats
import pandas as pd
import re


# fetch values between ***diameter values*** and ***time taken*** from the txt file
ORIG_PATH = "/home/marilin/Documents/ESP/data/fiber_tests/fiber_test_1/original_data/compound_results/"
RES_PATH = "/home/marilin/Documents/ESP/data/fiber_tests/fiber_test_1/classical_results/compound_results/"
U_PATH =  "/home/marilin/Documents/ESP/data/fiber_tests/fiber_test_1/unet_results/compound_results/"

FILES = os.listdir(ORIG_PATH)
RES_FILES = os.listdir(RES_PATH)
U_FILES =os.listdir(U_PATH)

# t-test for synthesised fibers - distro known 
# manual_t_p, manual_conf_low, manual_conf_high = [], [], []
# measured_t_p, measured_conf_low, measured_conf_high  = [], [], []

f_p, h_p = [], []
manual_files, classical_files, u_files = [], [], []
spearman_corr, pearson_corr = [], []

for file_path in FILES:

    #start_time = time.time()
    file = ORIG_PATH+file_path


    if file.endswith(".txt") and re.search("(_2k_|_5k_|2k)", file) == None:

        lst = []
        with open(file) as f:
            for line in f:
                lst.append(line.strip("\n").split(":"))

        lst = sum(lst, [])
        lst = [float(val) if len(val.split(".")[0])>1 else float(val) * 1000 for val in lst[1:]]
        lst.insert(0,"***diameter values***")

        #print(lst)

        dms_manual = np.array((lst[len(lst)-lst[::-1].index("***diameter values***"):]), dtype=np.uint0)


        core = file.split("/")[-1].split(".txt")[0]

        manual_files.append("manual_"+file.split("/")[-1].split(".txt")[0])


        for f_path in RES_FILES:

            f_res = RES_PATH+f_path

            if f_res.endswith(".txt") and file_path in f_res:
               
                lst = []
                with open(f_res) as f:
                    for line in f:
                        lst.append(line.strip("\n").split(":"))
                lst = sum(lst, [])
                
                dms_measured = np.array((lst[len(lst)-lst[::-1].index("***diameter values***"):]), dtype = np.uint0)
                
                classical_files.append("classical_"+f_res.split("/")[-1].split(".txt")[0])




                for u_path in U_FILES:

                    u_res = U_PATH+u_path

                    if u_res.endswith(".txt") and file_path in u_res:
                        lst = []
                        with open(u_res) as f:
                            for line in f:
                                lst.append(line.strip("\n").split(":"))
                        lst = sum(lst, [])
                        
     
                        dms_unet = np.array((lst[len(lst)-lst[::-1].index("***diameter values***"):]), dtype = np.uint0)
     
                        u_files.append("unet_"+u_res.split("/")[-1].split(".txt")[0])

                



# #                 # The bounds of the 95% confidence interval are the minimum and maximum values of the parameter popmean for which the p-value of the test would be 0.05
# #                 # t-test stats
# #                 measured_t_p.append(stats.ttest_1samp(dms_measured, popmean=20).pvalue)
# #                 measured_conf_low.append(stats.ttest_1samp(dms_measured, popmean=20).confidence_interval(confidence_level=0.99).low)
# #                 measured_conf_high.append(stats.ttest_1samp(dms_measured, popmean=20).confidence_interval(confidence_level=0.99).high)
#                 # f-test (assumes that std.devs of populations are equal, considers pop.mean)
                # print(manual_files)
                # print(classical_files)
                # print(u_files)

                f_p.append(stats.f_oneway(dms_manual, dms_measured).pvalue)
                f_p.append(stats.f_oneway(dms_manual, dms_unet).pvalue)
                f_p.append(stats.f_oneway(dms_manual, dms_measured, dms_unet).pvalue)

                # h-test (consider pop.median)
                h_p.append(stats.kruskal(dms_manual, dms_measured).pvalue)
                h_p.append(stats.kruskal(dms_manual, dms_unet).pvalue)
                h_p.append(stats.kruskal(dms_manual, dms_measured, dms_unet).pvalue)

                

#                 ## spearman correlation - for smaller samples (< 500 obs) permutation test is needed (checking correlation between unet and classical seg)
#                 # for cor, samples must be same of same length - rm last el from list
   
                # dms_unet_tmp = dms_unet[:len(dms_measured)]
                def statistic(x):  # permute only `x`
                    return stats.spearmanr(sorted(dms_measured), sorted(dms_unet)).statistic
                
                res_exact = stats.permutation_test((sorted(dms_measured),), statistic, permutation_type='pairings')
                res_asymptotic = stats.spearmanr(sorted(dms_measured), sorted(dms_unet))
                
                spearman_corr.append((res_exact.statistic, res_exact.pvalue, res_asymptotic.statistic, res_asymptotic.pvalue))

                ## pearson's correlation
                
                pearson_stat = stats.pearsonr(sorted(dms_measured), sorted(dms_unet))
                pearson_corr.append((pearson_stat.statistic, pearson_stat.pvalue))



print(spearman_corr)
print(pearson_corr)
# #print(res_exact.pvalue, res_asymptotic.pvalue)
# print(f_p)
# print(h_p)

# # #gen_files =  [string.split(".txt")[0] for string in FILES if string.endswith(".txt")]
# # #measured_files =  [string.split(".txt")[0] for string in RES_FILES if string.endswith(".txt")]

# print(f_p)
# print(h_p)
index = list(zip(manual_files, classical_files)) + list(zip(manual_files, u_files)) + list(zip(manual_files, classical_files, u_files))

pd.DataFrame(data = {"F-test p value":  (f_p), "H-test p value": (h_p)},index= index).to_csv("/home/marilin/Documents/ESP/data/fiber_tests/fiber_test_1/stat_results.xlsx")

