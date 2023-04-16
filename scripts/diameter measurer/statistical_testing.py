import numpy as np
import os
from scipy import stats

# fetch values between ***diameter values*** and ***time taken*** from the txt file
ORIG_PATH = "/home/marilin/Documents/ESP/data/fiber_tests/synthesised_fibers/2_tone_fibers/"
RES_PATH = "/home/marilin/Documents/ESP/data/fiber_tests/synthesised_fibers/2_tone_fibers_autom_results/"
VIS_PATH = "/home/marilin/Documents/ESP/data/fiber_tests/synthesised_fibers/visuals/"

FILES = os.listdir(ORIG_PATH)
RES_FILES = os.listdir(RES_PATH)
#print(RES_FILES)

gen_p, gen_conf_low, gen_conf_high = [], [], []
measured_p, measured_conf_low, measured_conf_high = [], [], []


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

        #print(f"confidence for {file}", stats.ttest_1samp(dms_gen, popmean=20).confidence_interval(confidence_level=0.99))
        gen_p.append(stats.ttest_1samp(dms_gen, popmean=20).pvalue)
        gen_conf_low.append(stats.ttest_1samp(dms_gen, popmean=20).confidence_interval(confidence_level=0.99).low)
        gen_conf_high.append(stats.ttest_1samp(dms_gen, popmean=20).confidence_interval(confidence_level=0.99).high)

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

                print(f"confidence for {f_res}",stats.ttest_1samp(dms_measured, popmean=20).confidence_interval(confidence_level=0.99))

                # The bounds of the 95% confidence interval are the minimum and maximum values of the parameter popmean for which the p-value of the test would be 0.05
                measured_p.append(stats.ttest_1samp(dms_measured, popmean=20).pvalue)
                measured_conf_low.append(stats.ttest_1samp(dms_measured, popmean=20).confidence_interval(confidence_level=0.99).low)
                measured_conf_high.append(stats.ttest_1samp(dms_measured, popmean=20).confidence_interval(confidence_level=0.99).high)

