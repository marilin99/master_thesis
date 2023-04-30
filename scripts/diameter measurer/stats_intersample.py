import numpy as np
import os
from scipy import stats
import pandas as pd
import re


# fetch values between ***diameter values*** and ***time taken*** from the txt file
ORIG_PATH = "/home/marilin/Documents/ESP/data/fiber_tests/fiber_test_1/original_data/compound_results/"
#ORIG_PATH = "/home/marilin/Documents/ESP/data/fiber_tests/fiber_test_1/unet_results/compound_results/"
#ORIG_PATH = "/home/marilin/Documents/ESP/data/fiber_tests/manual_2/sub_3/"
#U_PATH =  "/home/marilin/Documents/ESP/data/fiber_tests/fiber_test_3/unet_results/"

#RES_PATH = "/home/marilin/Documents/ESP/data/fiber_tests/synthesised_fibers/2_tone_fibers_manual/manual_2/"

FILES = os.listdir(ORIG_PATH)
#RES_FILES = os.listdir(RES_PATH)
#U_FILES =os.listdir(U_PATH)

# t-test for synthesised fibers - distro known 
# manual_t_p, manual_conf_low, manual_conf_high = [], [], []
# measured_t_p, measured_conf_low, measured_conf_high  = [], [], []
manual_dms = []

f_p, h_p = [], []
manual_files, classical_files, u_files = [], [], []
spearman_corr, pearson_corr = [], []

for file_path in FILES:

    #start_time = time.time()
    file = ORIG_PATH+file_path


    if file.endswith(".txt") and re.search("(_2k_|_5k_|2k)", file) == None:
        print(file_path)

        lst = []
        with open(file) as f:
            for line in f:
                lst.append(line.strip("\n").split(":"))

        lst = sum(lst, [])
        try:
            lst = [float(val) if len(val.split(".")[0])>1 else float(val) * 1000 for val in lst[1:lst.index("***time taken***")]]
        except:
            lst = [float(val) if len(val.split(".")[0])>1 else float(val) * 1000 for val in lst[1:]]
        lst.insert(0,"***diameter values***")

        #print(lst)

        dms_manual = np.array((lst[len(lst)-lst[::-1].index("***diameter values***"):]), dtype=np.uint0)
        manual_dms.append(dms_manual)

        core = file.split("/")[-1].split(".txt")[0]

        manual_files.append("manual_"+file.split("/")[-1].split(".txt")[0])


        # for f_path in RES_FILES:

        #     f_res = RES_PATH+f_path

        #     if f_res.endswith(".txt") and file_path in f_res:
               
        #         lst = []
        #         with open(f_res) as f:
        #             for line in f:
        #                 lst.append(line.strip("\n").split(":"))
        #         lst = sum(lst, [])
        #         lst = [float(val) if len(val.split(".")[0])>1 else float(val) * 1000 for val in lst[1:]]
        #         lst.insert(0,"***diameter values***")
                
        #         dms_measured = np.array((lst[len(lst)-lst[::-1].index("***diameter values***"):]), dtype = np.uint0)
                
        #         classical_files.append("classical_"+f_res.split("/")[-1].split(".txt")[0])




                # for u_path in U_FILES:

                #     u_res = U_PATH+u_path

                #     if u_res.endswith(".txt") and file_path in u_res:
                #         lst = []
                #         with open(u_res) as f:
                #             for line in f:
                #                 lst.append(line.strip("\n").split(":"))
                #         lst = sum(lst, [])
                        
     
                #         dms_unet = np.array((lst[len(lst)-lst[::-1].index("***diameter values***"):]), dtype = np.uint0)
     
                #         u_files.append("unet_"+u_res.split("/")[-1].split(".txt")[0])

                



# #                 # The bounds of the 95% confidence interval are the minimum and maximum values of the parameter popmean for which the p-value of the test would be 0.05
# #                 # t-test stats
# #                 measured_t_p.append(stats.ttest_1samp(dms_measured, popmean=20).pvalue)
# #                 measured_conf_low.append(stats.ttest_1samp(dms_measured, popmean=20).confidence_interval(confidence_level=0.99).low)
# #                 measured_conf_high.append(stats.ttest_1samp(dms_measured, popmean=20).confidence_interval(confidence_level=0.99).high)
#                 # f-test (assumes that std.devs of populations are equal, considers pop.mean)
                # print(manual_files)
                # print(classical_files)
                # print(u_files)

                # z_score_measured = stats.zscore(dms_measured, ddof=len(dms_measured)-1)
                # z_score_orig = stats.zscore(dms_manual, ddof=len(dms_manual)-1)
                #z_score_unet = stats.zscore(dms_unet, ddof=len(dms_unet)-1)

                #f_p.append(stats.f_oneway(z_score_orig, z_score_measured).pvalue)
                #f_p.append(stats.f_oneway(z_score_orig, z_score_unet).pvalue)
                #f_p.append(stats.f_oneway(z_score_orig, z_score_measured, dms_unet).pvalue)


import itertools

combos = list(itertools.combinations(range(len(manual_dms)), 2))

for val in combos:
    #Find intersection of two sets
    manual_dms_set_1 = set(manual_dms[val[0]])
    manual_dms_set_2 = set(manual_dms[val[1]])
    #manual_dms_set_3 = set(manual_dms[2])

    nominator = manual_dms_set_1.intersection(manual_dms_set_2)
    #nominator = manual_dms_set_1.intersection(manual_dms_set_2)

    #Find union of two sets
    denominator =  manual_dms_set_1.union( manual_dms_set_2)

    #Take the ratio of sizes
    similarity = len(nominator)/len(denominator)

    print(val, similarity)


f_p.append(stats.f_oneway(manual_dms[0], manual_dms[1],  manual_dms[2]).pvalue)

                 # h-test (consider pop.median)
                #h_p.append(stats.kruskal(z_score_orig, z_score_measured).pvalue)
                #h_p.append(stats.kruskal(z_score_orig, z_score_unet).pvalue)
                #h_p.append(stats.kruskal(z_score_orig, z_score_measured, dms_unet).pvalue)
h_p.append(stats.kruskal(manual_dms[0], manual_dms[1], manual_dms[2]).pvalue)

                

#                 ## spearman correlation - for smaller samples (< 500 obs) permutation test is needed (checking correlation between unet and classical seg)
#                 # for cor, samples must be same of same length - rm last el from list
   
                # dms_unet_tmp = dms_unet[:len(dms_measured)]
                # def statistic(x):  # permute only `x`
                #     return stats.spearmanr(sorted(dms_measured), sorted(dms_unet)).statistic
                
                # res_exact = stats.permutation_test((sorted(dms_measured),), statistic, permutation_type='pairings')
                # res_asymptotic = stats.spearmanr(sorted(dms_measured), sorted(dms_unet))
                
                # spearman_corr.append((res_exact.statistic, res_exact.pvalue, res_asymptotic.statistic, res_asymptotic.pvalue))

                # ## pearson's correlation
                
                # pearson_stat = stats.pearsonr(sorted(dms_measured), sorted(dms_unet))
                # pearson_corr.append((pearson_stat.statistic, pearson_stat.pvalue))



# print(spearman_corr)
# print(pearson_corr)
# #print(res_exact.pvalue, res_asymptotic.pvalue)
# print(f_p)
# print(h_p)

# # #gen_files =  [string.split(".txt")[0] for string in FILES if string.endswith(".txt")]
# # #measured_files =  [string.split(".txt")[0] for string in RES_FILES if string.endswith(".txt")]

print(f_p)
print(h_p)
# index = list(zip(manual_files, classical_files))  #+ list(zip(manual_files, u_files)) + list(zip(manual_files, classical_files, u_files))
# print(pd.DataFrame(data = {"F-test p value":  (f_p), "H-test p value": (h_p)},index= index)) #.to_csv("/home/marilin/Documents/ESP/data/fiber_tests/fiber_test_3/zscore_stat_results.xlsx")

