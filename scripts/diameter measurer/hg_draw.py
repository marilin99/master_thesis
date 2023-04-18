import os
import matplotlib.pyplot as plt
import numpy as np
import re
import seaborn as sns
import pandas as pd


ORIG_PATH = "/home/marilin/Documents/ESP/data/fiber_tests/fiber_test_3/original_data_3/"
RES_PATH = "/home/marilin/Documents/ESP/data/fiber_tests/fiber_test_3/classical_results/"
U_PATH =  "/home/marilin/Documents/ESP/data/fiber_tests/fiber_test_3/unet_results/"
VIS_PATH = "/home/marilin/Documents/ESP/data/fiber_tests/fiber_test_3/visuals/"

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
        # Binned hg-s #

        fig, ax = plt.subplots()
        plt.style.use('seaborn-deep')
        
        dms_measured_c = dms_measured[:len(dms_manual)]
        dms_unet_c = dms_unet[:len(dms_manual)]


        bins = np.linspace(np.amin((np.min(dms_manual), np.min(dms_measured_c), np.min(dms_unet_c))), np.amax(( np.max(dms_manual), np.max(dms_measured_c), np.max(dms_unet_c))), 10, dtype=np.uint16)

        plt.hist([dms_manual, dms_measured_c, dms_unet_c], bins, label=["Manual dm-s", "Classical dm-s", "U-Net dm-s"])
        plt.legend(loc='upper right')

        # Density hg-s #
        # Create a combined dataframe
        # df = pd.DataFrame({'Generated dm-s': dms_manual, 'Classical dm-s': dms_measured,'U-Net dm-s': dms_measured})

        # # Plot the density plot
        # sns.histplot(data=df, x='Generated dm-s', y='Measured dm-s', shade=True, shade_lowest=False, legend =True)

        # Kde plots # 

        # sns.kdeplot() for stacked 
        # # Plot the stacked density plot without bins

        # sns.kdeplot(data=dms_manual, common_norm=False, fill=True, alpha=0.5, label="Manual dm-s")  # plot first data series
        # sns.kdeplot(data=dms_measured[:len(dms_manual)], common_norm=False, fill=True, alpha=0.5, label="Classical dm-s")  # plot second data series
        # sns.kdeplot(data=dms_unet[:len(dms_manual)], common_norm=False, fill=True, alpha=0.5, label="U-Net dm-s")

        #plt.ylabel("Relative frequency")
        plt.ylabel("Frequency")
        plt.xlabel("Fiber diameter (nm)")
        plt.title('Histogram of manual vs automatic measured dm-s')
        #plt.xlim((0,2000))
        ax.set_xticks(bins)
        plt.legend()
        #plt.show()
        plt.savefig(f"{VIS_PATH}binned_{core}.png")
        plt.clf()