### visuals' script ###

import os
import matplotlib.pyplot as plt
import numpy as np
import re
import seaborn as sns
import pandas as pd

# fiber_test_1 has compound_results in all folders, fiber_test2 in just one, fiber_test_3 has none



# ORIG_PATH = "/home/marilin/Documents/ESP/data/fiber_tests/fiber_test_3/original_data_3/"
# RES_PATH = "/home/marilin/Documents/ESP/data/fiber_tests/fiber_test_3/classical_results/"
# U_PATH =  "/home/marilin/Documents/ESP/data/fiber_tests/fiber_test_3/unet_results/"
# VIS_PATH = "/home/marilin/Documents/ESP/data/fiber_tests/fiber_test_3/visuals/"

# ORIG_PATH =  "/home/marilin/Documents/ESP/data/fiber_tests/synthesised_fibers/2_tone_fibers/"
# MAN1_PATH = "/home/marilin/Documents/ESP/data/fiber_tests/manual_1/"
# MAN2_PATH = "/home/marilin/Documents/ESP/data/fiber_tests/manual_2/"
# RES_PATH = "/home/marilin/Documents/ESP/data/fiber_tests/synthesised_fibers/2_tone_fibers_autom_results/"
# VIS_PATH = "/home/marilin/Documents/ESP/data/fiber_tests/synthesised_fibers/visuals/"

ORIG_PATH =  "/home/marilin/Documents/ESP/data/fiber_tests/synthesised_fibers/three_dm_fibers_sub/"
RES_PATH =  "/home/marilin/Documents/ESP/data/fiber_tests/synthesised_fibers/three_dm_fibers_autom_results_500/"
VIS_PATH = "/home/marilin/Documents/ESP/data/fiber_tests/synthesised_fibers/visuals/"


# ORIG_PATH =  "/home/marilin/Documents/ESP/data/fiber_tests/seed_time_test/classical_results/sub_1/"
# #RES_PATH =  "/home/marilin/Documents/ESP/data/fiber_tests/seed_time_test/three_dm_fibers_autom_results/"
# VIS_PATH = "/home/marilin/Documents/ESP/data/fiber_tests/seed_time_test/visuals/"


FILES = os.listdir(ORIG_PATH)
# MAN1_FILES = os.listdir(MAN1_PATH)
# MAN2_FILES = os.listdir(MAN2_PATH)
RES_FILES = os.listdir(RES_PATH)
# U_FILES =os.listdir(U_PATH)
dms_m = []
for file_path in FILES:

    #start_time = time.time()
    file = ORIG_PATH+file_path


    if file.endswith(".txt"): # and re.search("(_2k_|_5k_|2k)", file) == None:
        
        lst = []
        with open(file) as f:
            for line in f:
                lst.append(line.strip("\n").split(":"))

        lst = sum(lst, [])
        #lst = [float(val) if len(val.split(".")[0])>1 else float(val) * 1000 for val in lst[1:]]
        #lst.insert(0,"***diameter values***")

        #print(lst)
        #lst.index("***time taken***")
        dms_manual = np.array((lst[len(lst)-lst[::-1].index("***diameter values***"):]), dtype=np.uint8)
        dms_m.append(dms_manual)

        core = file.split("/")[-1].split(".txt")[0]

        #manual_files.append("manual_"+file.split("/")[-1].split(".txt")[0])


        for f_path in RES_FILES:

            f_res = RES_PATH+f_path

            # print(file_path)
            # print(f_path)
            # print("yo", f_path.split(".txt")[0][:-3])

            if f_res.endswith(".txt") and file_path in f_res:
                print(f_res)

               
                lst = []
                with open(f_res) as f:
                    for line in f:
                        lst.append(line.strip("\n").split(":"))
                lst = sum(lst, [])
                
                dms_measured = np.array((lst[len(lst)-lst[::-1].index("***diameter values***"):lst.index("***time taken***")]), dtype = np.uint8)
                
                #classical_files.append("classical_"+f_res.split("/")[-1].split(".txt")[0])




                # for man1_path in MAN1_FILES:

                #     man1_res = MAN1_PATH+man1_path

                #     if man1_res.endswith(".txt") and file_path in man1_res:
                #         lst = []
                #         with open(man1_res) as f:
                #             for line in f:
                #                 lst.append(line.strip("\n").split(":"))
                #         lst = sum(lst, [])
                
                #         lst = [float(val) for val in lst[1:]]
                #         lst.insert(0,"***diameter values***")    
                #         dms_man1 = np.array((lst[len(lst)-lst[::-1].index("***diameter values***"):]), dtype = np.uint8)
               

                #         for man2_path in MAN2_FILES:

                #             man2_res = MAN2_PATH+man1_path

                #             if man2_res.endswith(".txt") and file_path in man2_res:
                #                 lst = []
                #                 with open(man2_res) as f:
                #                     for line in f:
                #                         lst.append(line.strip("\n").split(":"))
                #                 lst = sum(lst, [])
                #                 lst = [float(val) for val in lst[1:]]
                #                 lst.insert(0,"***diameter values***")
            
                #                 dms_man2 = np.array((lst[len(lst)-lst[::-1].index("***diameter values***"):]), dtype = np.uint0)
            
                        #u_files.append("unet_"+u_res.split("/")[-1].split(".txt")[0])
        # Binned hg-s #

        # fig, ax = plt.subplots()
        # plt.style.use('seaborn-deep')
        #         #print(dms_measured)
       
        # ltd_len = min(len(dms_measured), len(dms_manual)) #, len(dms_man1), len(dms_man2))

        # dms_measured = dms_measured[:ltd_len]
        # # dms_unet = dms_unet[:ltd_len]
        # dms_manual = dms_manual[:ltd_len]
        # dms_man1 = dms_man1[:ltd_len]
        # dms_man2 = dms_man2[:ltd_len]


        # bins = np.linspace(np.amin((np.min(dms_manual), np.min(dms_measured), np.min(dms_man1),np.min(dms_man2))), np.amax(( np.max(dms_manual), np.max(dms_measured), np.max(dms_man1), np.max(dms_man2))), 10, dtype=np.uint16)

        # plt.hist([dms_manual, dms_measured, dms_man1, dms_man2], bins, label=["Generated dm-s", "Automated dm-s", "Manual 1 dm-s", "Manual 2 dm-s"])
        # plt.legend(loc='upper right')
                

        # Density hg-s #
        # Create a combined dataframe
        #df = pd.DataFrame({'Generated dm-s': dms_manual, 'Automated dm-s': dms_measured,'Manual 1 dm-s': dms_man1, 'Manual 2 dm-s': dms_man2})

        # Plot the density plot
        #sns.histplot(data=df, x='Generated dm-s', y='Measured dm-s', shade=True, shade_lowest=False, legend =True)

        # Kde plots # 

        # sns.kdeplot() for stacked 
        # # Plot the stacked density plot without bins

        sns.kdeplot(data=dms_manual, common_norm=True, fill=True, alpha=0.5, label="Generated dm-s")  # plot first data series
        sns.kdeplot(data=dms_measured, common_norm=True, fill=True, alpha=0.5, label="Automated dm-s")  # plot second data series
        # sns.kdeplot(data=dms_man1, common_norm=True, fill=True, alpha=0.5,color ="yellow", label="Manual 1 dm-s")
        # sns.kdeplot(data=dms_man2, common_norm=True, fill=True, alpha=0.5,label="Manual 2 dm-s")

        # box plots # 
        #sns.boxplot(data=df)



#sns.kdeplot(data=dms_m, common_norm=True, fill=True, alpha=0.5)
#plt.ylabel("Fiber diameter (pixels)")
        plt.ylabel("Relative frequency")
        plt.xlabel("Fiber diameter (pixels)")
        plt.title('Generated vs automatic measured dm-s')
        #plt.xlim((0,2000))
        #ax.set_xticks(bins)
        plt.legend()
        #plt.xlim(left=0)
        #plt.show()
        plt.savefig(f"{VIS_PATH}kde_500_{core}.png")
        plt.clf()