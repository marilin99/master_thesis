###  statistical ### 

import numpy as np
import os
from scipy import stats
import pandas as pd
import re

# z-score function
def z_score(value, pop_mean, pop_std):
    return (value-pop_mean) / pop_std

# fetch values between ***diameter values*** and ***time taken*** from the txt file
ORIG_PATH = "/home/marilin/Documents/ESP/data/fiber_tests/fiber_test_2/original_data_2/" # reference pt 

# automated - CI + comparing the CI-s 
RES_PATH = "/home/marilin/Documents/ESP/data/fiber_tests/fiber_test_2/classical_results_2/compound_results/"
U_PATH =  "/home/marilin/Documents/ESP/data/fiber_tests/fiber_test_2/unet_results/"

# manuals - fetching CI from here
MAN1_PATH = "/home/marilin/Documents/ESP/data/fiber_tests/manual_1/"
MAN2_PATH = "/home/marilin/Documents/ESP/data/fiber_tests/manual_2/"


FILES = os.listdir(ORIG_PATH)
RES_FILES = os.listdir(RES_PATH)
U_FILES =os.listdir(U_PATH)
MAN1_FILES = os.listdir(MAN1_PATH)
MAN2_FILES = os.listdir(MAN2_PATH)

# t-test for synthesised fibers - distro known 
manual1_t_p, classical_t_p, unet_t_p = [], [], []
manual2_t_p, manual2_conf_low, manual2_conf_high = [], [], []

# measured_t_p, measured_conf_low, measured_conf_high  = [], [], []

f_p, h_p = [], []
manual_files, classical_files, u_files, manual1_files, manual2_files = [], [], [], [], []
# spearman_corr, pearson_corr = [], []
z_score_orig, z_score_measured, z_score_unet = [], [], []
z_score_man1, z_score_man2 = [], []

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
        lst = [float(val) if len(val.split(".")[0])>1 else float(val) * 1000 for val in lst[1:]]
        lst.insert(0,"***diameter values***")

        #print(lst)

        dms_manual = np.array((lst[len(lst)-lst[::-1].index("***diameter values***"):]), dtype=np.uint0)

        pop_mean = np.mean(dms_manual)
        pop_std = np.std(dms_manual)

        z_score_orig.append(z_score(dms_manual, pop_mean, pop_std))
        

        core = file.split("/")[-1].split(".txt")[0]

        manual_files.append("reference_"+file.split("/")[-1].split(".txt")[0])


        for f_path in MAN1_FILES:

            f_res = MAN1_PATH+f_path

            if f_res.endswith(".txt") and file_path in f_res:
               
                lst = []
                with open(f_res) as f:
                    for line in f:
                        lst.append(line.strip("\n").split(":"))

                lst = sum(lst, [])
                lst = [float(val) if len(val.split(".")[0])>1 else float(val) * 1000 for val in lst[1:]]
                lst.insert(0,"***diameter values***")
                
                dms_measured = np.array((lst[len(lst)-lst[::-1].index("***diameter values***"):]), dtype = np.uint0)
     
                manual1_files.append("manual_1_"+f_res.split("/")[-1].split(".txt")[0])

                z_score_man1.append(z_score(dms_measured, pop_mean, pop_std))

                for f2_path in MAN2_FILES:

                    f2_res = MAN2_PATH+f2_path

                    if f2_res.endswith(".txt") and file_path in f2_res:
                        
                    
                        lst = []
                        with open(f2_res) as f:
                            for line in f:
                                lst.append(line.strip("\n").split(":"))

                        lst = sum(lst, [])
                        lst = [float(val) if len(val.split(".")[0])>1 else float(val) * 1000 for val in lst[1:]]
                        lst.insert(0,"***diameter values***")
                        
                        dms_measured2 = np.array((lst[len(lst)-lst[::-1].index("***diameter values***"):]), dtype = np.uint0)
            
                        manual2_files.append("manual_2_"+f2_res.split("/")[-1].split(".txt")[0])

                        z_score_man2.append(z_score(dms_measured2, pop_mean, pop_std))
                        

                        for res_path in RES_FILES:

                            f3_res = RES_PATH+res_path

                            if f3_res.endswith(".txt") and file_path in f3_res:
                            
                                lst = []
                                with open(f3_res) as f:
                                    for line in f:
                                        lst.append(line.strip("\n").split(":"))

                                lst = sum(lst, [])
                                lst = [float(val) if len(val.split(".")[0])>1 else float(val) * 1000 for val in lst[1:]]
                                lst.insert(0,"***diameter values***")
                                
                                dms_measured3 = np.array((lst[len(lst)-lst[::-1].index("***diameter values***"):]), dtype = np.uint0)
                    
                                classical_files.append("classical_"+f3_res.split("/")[-1].split(".txt")[0])

                                z_score_measured.append(z_score(dms_measured3, pop_mean, pop_std))

                                for u_path in U_FILES:

                                    f4_res = U_PATH+u_path

                                    if f4_res.endswith(".txt") and file_path in f4_res:
                                    
                                        lst = []
                                        with open(f4_res) as f:
                                            for line in f:
                                                lst.append(line.strip("\n").split(":"))

                                        lst = sum(lst, [])
                                        lst = [float(val) if len(val.split(".")[0])>1 else float(val) * 1000 for val in lst[1:]]
                                        lst.insert(0,"***diameter values***")
                                        
                                        dms_measured4 = np.array((lst[len(lst)-lst[::-1].index("***diameter values***"):]), dtype = np.uint0)
                            
                                        u_files.append("unet_"+f4_res.split("/")[-1].split(".txt")[0])

                                        z_score_unet.append(z_score(dms_measured4, pop_mean, pop_std))



                # z-test giving too significant values on its own # 
                
                # two sample case - difference between mean of x1 and x2, ddof=1 in case of comparing means - z-test
                # import statsmodels.stats.weightstats as ws
                # cm_obj = ws.CompareMeans(ws.DescrStatsW(dms_manual), ws.DescrStatsW(dms_measured))
                # zstat, z_pval = cm_obj.ztest_ind(usevar='unequal')
                # print(zstat, z_pval)



                #, ddof=len(dms_manual)-1))
                # print(z_score_man1)
                # z_score_measured = stats.zscore(dms_measured, ddof=len(dms_measured)-1)
                # z_score_orig = stats.zscore(dms_manual, ddof=len(dms_manual)-1)
               
                # #print(len(z_score_measured), len(dms_manual))
                #print(z_score_orig)
                # f_p.append(stats.f_oneway(z_score_orig, z_score_man1).pvalue)
                # h_p.append(stats.kruskal(z_score_orig, z_score_man1).pvalue)
                # f_p.append(stats.ttest_ind(z_score_orig, z_score_man1).pvalue)
                # f_p.append(stats.ttest_ind(z_score_orig, z_score_man2).pvalue)
                # f_p.append(stats.ttest_ind(z_score_orig, z_score_unet).pvalue)
                # f_p.append(stats.ttest_ind(z_score_orig, z_score_measured).pvalue)




print(len(z_score_man2))
# for n in range(len(manual1_files)):
#     for stat1, stat2 in zip(z_score_man1[n], z_score_orig[n]):

# how much a given value differs from the ref standard deviation 
# z_scores have equal var and mean
for n in range(len(z_score_orig)):

    manual1_t_p.append(stats.ttest_ind(z_score_man1[n], z_score_orig[n], equal_var=True).pvalue)
    manual2_t_p.append(stats.ttest_ind(z_score_man2[n], z_score_orig[n], equal_var=True).pvalue)
    unet_t_p.append(stats.ttest_ind(z_score_unet[n], z_score_orig[n], equal_var=True).pvalue)
    classical_t_p.append(stats.ttest_ind(z_score_measured[n], z_score_orig[n], equal_var=True).pvalue)

    print(stats.ttest_1samp(z_score_man1[n], np.mean(z_score_orig[n])).pvalue)
    print(stats.ttest_1samp(z_score_man2[n], np.mean(z_score_orig[n])).pvalue)
    print(stats.ttest_1samp(z_score_unet[n], np.mean(z_score_orig[n])).pvalue)
    print(stats.ttest_1samp(z_score_measured[n], np.mean(z_score_orig[n])).pvalue)

    print(stats.ttest_1samp(z_score_man1[n], np.mean(z_score_orig[n])).confidence_interval(0.95))
    print(stats.ttest_1samp(z_score_man2[n], np.mean(z_score_orig[n])).confidence_interval(0.95))
    print(stats.ttest_1samp(z_score_unet[n], np.mean(z_score_orig[n])).confidence_interval(0.95))
    print(stats.ttest_1samp(z_score_measured[n], np.mean(z_score_orig[n])).confidence_interval(0.95))

# 1-sample test too significant # 
#print(stats.ttest_1samp(z_score_man1[0], np.mean(z_score_orig[0])))

# print(stats.f_oneway(z_score_man1[0], z_score_orig[0]).pvalue)
# print(stats.kruskal(z_score_man1[0], z_score_orig[0]).pvalue)    
# print(stats.f_oneway(z_score_unet[0], z_score_orig[0]).pvalue)
# print(stats.kruskal(z_score_unet[0], z_score_orig[0]).pvalue)

print(manual1_t_p, manual2_t_p, unet_t_p, classical_t_p)
# print(len(z_score_man1))
# print(f_p)
# print(h_p)

# index = list(zip(manual_files, classical_files))  + list(zip(manual_files, u_files)) + list(zip(manual_files, manual1_files)) + list(zip(manual_files, manual2_files))

#print(pd.DataFrame(data = {"T-test p value":  list(zip(classical_t_p, unet_t_p, manual1_t_p,manual2_t_p))},index= index)) #.to_csv("/home/marilin/Documents/ESP/data/fiber_tests/fiber_test_2/z_stat_results_man1.xlsx")

# (f_p), "H-test p value": (h_p)},index= index)


