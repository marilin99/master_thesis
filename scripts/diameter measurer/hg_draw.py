import os
import matplotlib.pyplot as plt
import numpy as np
import re
import seaborn as sns
import pandas as pd

# fetch values between ***diameter values*** and ***time taken*** from the txt file
ORIG_PATH = "/home/marilin/Documents/ESP/data/fiber_tests/synthesised_fibers/2_tone_fibers/"
RES_PATH = "/home/marilin/Documents/ESP/data/fiber_tests/synthesised_fibers/2_tone_fibers_autom_results/"
VIS_PATH = "/home/marilin/Documents/ESP/data/fiber_tests/synthesised_fibers/visuals/"

FILES = os.listdir(ORIG_PATH)
RES_FILES = os.listdir(RES_PATH)
#print(RES_FILES)


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

        #print(file)
        core = file.split("/")[-1].split(".txt")[0]

        for f_path in RES_FILES:

            f_res = RES_PATH+f_path

            if f_res.endswith(".txt") and core in f_res:

                lst = []

                with open(f_res) as f:

                    for line in f:
                        lst.append(line.strip("\n").split(":"))
                
                lst = sum(lst, [])
                
                dms_measured = np.array((lst[len(lst)-lst[::-1].index("***diameter values***"): lst.index("***time taken***")]), dtype = np.uint0)

        # Binned hg-s #

        # fig, ax = plt.subplots()
        # plt.style.use('seaborn-deep')
        # bins = np.linspace(np.minimum(np.min(dms_gen), np.min(dms_measured)), np.maximum( np.max(dms_gen), np.max(dms_measured)), 10)

        # plt.hist([dms_gen, dms_measured], bins, label=["Generated dm-s", "Measured dm-s"])
        # plt.legend(loc='upper right')

        # Density hg-s #
        # # Create a combined dataframe
        # df = pd.DataFrame({'Generated dm-s': dms_gen, 'Measured dm-s': dms_measured})

        # # Plot the density plot
        # sns.histplot(data=df, x='Generated dm-s', y='Measured dm-s', shade=True, shade_lowest=False, legend =True)

        # Kde plots # 

        # sns.kdeplot() for stacked 
        # Plot the stacked density plot without bins
        # sns.kdeplot(data=dms_gen, common_norm=False, fill=True, alpha=0.5, label="Generated dm-s")  # plot first data series
        # sns.kdeplot(data=dms_measured, common_norm=False, fill=True, alpha=0.5, label="Measured dm-s")  # plot second data series

        plt.ylabel("Relative frequency")
        plt.xlabel("Fiber diameter (pixels)")
        plt.title('Histogram of generated vs measured dm-s')
        plt.legend()
        # plt.show()
        plt.savefig(f"{VIS_PATH}kde_{core}.png")
        plt.clf()