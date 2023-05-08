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
# MAN2_PATH = "/home/marilin/Documents/ESP/data/fiber_tests/manual_2/"qq
# RES_PATH = "/home/marilin/Documents/ESP/data/fiber_tests/synthesised_fibers/2_tone_fibers_autom_results/"
# VIS_PATH = "/home/marilin/Documents/ESP/data/fiber_tests/synthesised_fibers/visuals/"

ORIG_PATH =  "/home/marilin/Documents/ESP/data/fiber_tests/synthesised_fibers/three_dm_fibers_sub/"
RES_PATH =  "/home/marilin/Documents/ESP/data/fiber_tests/synthesised_fibers/three_dm_fibers_autom_results_100/"
VIS_PATH = "/home/marilin/Documents/ESP/data/fiber_tests/synthesised_fibers/visuals/"

#2k,10k,15k,20k
var = "sub_1"

# RES_PATH =  f"/home/marilin/Documents/ESP/data/fiber_tests/seed_time_test/classical_results/{var}/"
# # #RES_PATH =  "/home/marilin/Documents/ESP/data/fiber_tests/seed_time_test/three_dm_fibers_autom_results/"
# VIS_PATH = "/home/marilin/Documents/ESP/data/fiber_tests/seed_time_test/visuals/"


FILES = os.listdir(ORIG_PATH)
# MAN1_FILES = os.listdir(MAN1_PATH)
# MAN2_FILES = os.listdir(MAN2_PATH)
RES_FILES = os.listdir(RES_PATH)
# U_FILES =os.listdir(U_PATH)
dms_m, dms_gen = [],[]

for file_path in FILES:

	#start_time = time.time()
	file = ORIG_PATH+file_path


	if file.endswith(".txt"): # and re.search("(_2k_|_5k_|2k)", file) == None:
		print(file)
		lst = []
		with open(file) as f:
			for line in f:
				lst.append(line.strip("\n").split(":"))

		lst = sum(lst, [])
		#lst = [float(val) if len(val.split(".")[0])>1 else float(val) * 1000 for val in lst[1:]]
		#lst.insert(0,"***diameter values***")

		#print(lst)
		#lst.index("***time taken***")
		dms_manual = np.array((lst[len(lst)-lst[::-1].index("***diameter values***"):]), dtype=np.int16)
		dms_m.append(dms_manual)

		core = file.split("/")[-1].split(".txt")[0]

		#manual_files.append("manual_"+file.split("/")[-1].split(".txt")[0])


		for f_path in RES_FILES:

			f_res = RES_PATH+f_path
			print("res", f_res)

			# print(file_path)
			# print(f_path)
			# print("yo", f_path.split(".txt")[0][:-3])

			if f_res.endswith(".txt"):  #and file_path in f_res:
				#print(f_res)
				
				
				lst = []
				with open(f_res) as f:
					for line in f:
						lst.append(line.strip("\n").split(":"))
				lst = sum(lst, [])
				
				dms_measured = np.array((lst[len(lst)-lst[::-1].index("***diameter values***"):lst.index("***time taken***")]), dtype = np.int16)
				dms_gen.append(dms_measured)
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
# for i in range(len(dms_gen)):
#     sns.histplot(data=dms_gen[i], multiple="stack",label=i+1)

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
import matplotlib.pylab as pylab
params = {
		'axes.labelsize': 14,
		'xtick.labelsize':14,
		'ytick.labelsize':14}
pylab.rcParams.update(params)

man = [10,20,10,10,15,15,15,20,15,15,15,20,10,15,20,20,20,15,10,20,15,20,20,10,15,10,10,15,20,20]
#man = [10,15,15,20,20,10,15,20,10,10,20,10,20,10,10,10,10,15,15,15,20,20,10,15,15,20,10,10,20,20]

### CURVED 750 ###
#gen = [21, 17, 21, 12, 18, 10, 17, 16, 10, 23, 23, 23, 21, 23, 20, 46, 18, 16, 13, 32, 10, 21, 28, 18, 22, 21, 21, 10, 21, 11, 20, 23, 21, 11, 20, 6, 13, 10, 42, 20, 28, 12, 23, 21, 17, 21, 17, 10, 23, 23, 16, 12, 22, 22, 10, 24, 23, 18, 18, 40, 35, 21, 21, 37, 24, 16, 12, 21, 26, 23, 21, 43, 46, 21, 12, 41, 23, 16, 43, 21, 21, 10, 21, 20, 16, 19, 21, 4, 20, 41, 41, 21, 15, 10, 16, 27, 21, 10, 21, 10, 19, 10, 16, 21, 16, 15, 26, 10, 10, 31, 24, 17, 17, 16, 23, 17, 10, 23, 21, 12, 25, 21, 21, 10, 24, 21, 23, 16, 10, 21, 29, 30, 21, 16, 18, 10, 14, 27, 16, 21, 21, 18, 10, 10, 21, 12, 21, 18, 24, 21, 32, 21, 21, 18, 20, 28, 16, 21, 19, 17, 18, 21, 17, 48, 21, 21, 21, 24, 10, 18, 17, 21, 16, 16, 16, 17, 16, 10, 16, 21, 20, 21, 32, 16, 21, 18, 17, 21, 18, 27, 16, 10, 10, 16, 19, 23, 10, 26, 12, 21, 17, 21, 24, 15, 21, 21, 12, 21, 29, 10, 28, 17, 21, 22, 17, 17, 14, 42, 16, 21, 21, 17, 13, 18, 21, 10, 43, 21, 14, 10, 12, 20, 22, 17, 16, 13, 10, 22, 17, 16, 23, 21, 37, 32, 17, 21, 32, 21, 20, 11, 16, 14, 21, 10, 15, 10, 27, 17, 12, 17, 44, 21, 16, 25, 12, 10, 12, 24, 16, 16, 12, 21, 30, 13, 21, 21, 16, 13, 21, 21, 21, 21, 42, 21, 17, 23, 32, 21, 24, 10, 10, 12, 12, 10, 14, 17, 17, 10, 15, 21, 16, 16, 21, 21, 12, 21, 10, 17, 11, 20, 37, 21, 21, 21, 17, 16, 10, 15, 20, 18, 10, 30, 21, 34, 17, 34, 23, 21, 11, 12, 40, 21, 12, 25, 10, 25, 18, 10, 23, 13, 11, 21, 19, 24, 31, 16, 21, 21, 11, 12, 16, 21, 10, 14, 22, 23, 21, 16, 20, 21, 15, 20, 16, 48, 12, 18, 17, 24, 34, 17, 18, 10, 19, 33, 21, 10, 21, 17, 21, 19, 28, 24, 10, 10, 16, 21, 23, 19, 24, 27, 21, 17, 21, 25, 24, 24, 26, 26, 23, 10, 21, 16, 16, 10, 16, 26, 29, 26, 21, 10, 11, 24, 21, 16, 16, 21, 46, 28, 21, 24, 13, 10, 26, 16, 32, 22, 24, 21, 21, 33, 21, 21, 30, 20, 32, 21, 16, 21, 21, 17, 13, 16, 24, 26, 17, 23, 10, 34, 12, 12, 10, 20, 21, 21, 27, 12, 33, 23, 10, 21, 20, 10, 20, 24, 21, 21, 18, 18, 10, 20, 21, 22, 21, 21, 41, 10, 32, 18, 7, 19, 13, 21, 10, 32, 10, 25, 23, 21, 10, 21, 34, 12, 20, 30, 16, 21, 16, 11, 21, 16, 24, 6, 19, 16, 13, 19, 21, 10, 17, 16, 21, 21, 21, 21, 21, 20, 13, 21, 19, 17, 16, 21, 35, 31, 32, 21, 35, 17, 26, 20, 10, 23, 12, 21, 16, 12, 16, 21, 12, 25, 21, 21, 10, 32, 18, 10, 21, 16, 22, 21, 21, 20, 30, 11, 28, 16, 34, 16, 18, 20, 21, 22, 16, 19, 21, 21, 21, 22, 27, 27, 32, 37, 11, 10, 40, 21, 20, 20, 12, 19, 21, 21, 22, 11, 21, 21, 10, 21, 16, 12, 20, 31, 8, 40, 46, 49, 10, 14, 19, 10, 16, 16, 16, 16, 21, 17, 21, 25, 19, 20, 26, 16, 27, 40, 24, 29, 12, 21, 21, 10, 10, 21, 17, 21, 18, 21, 11, 24, 41, 21, 32, 25, 23, 10, 23, 24, 10, 27, 37, 5, 21, 22, 21, 21, 20, 19, 26, 12, 21, 25, 21, 21, 10, 19, 21, 22, 10, 10, 11, 10, 10, 12, 17, 34, 10, 43, 22, 31, 40, 24, 26, 21, 17, 10, 13, 17, 21, 18, 21, 21, 21, 20, 10, 21, 26, 26, 21, 29, 22, 17, 17, 10, 23, 18, 20, 16, 14, 28, 11, 17, 11, 21, 22, 16, 16, 21, 16, 37, 10, 21, 17, 10, 21, 12, 13, 37, 21, 20, 21, 24, 16, 22, 13, 18, 12, 16, 21, 20, 13, 16, 17, 21, 12, 18, 28, 10, 19, 24, 13, 18, 21, 10, 10, 18, 18, 16, 53, 10, 16]

#gen =[16,10,21,29,12,27,21,21,10,10,20,20,10,30,10,13,16,21,14,32,16,20,11,20,16,19,15,10,12,10,16,16,21,10,10,11,16,19,15,10,10,10,10,22,17,16,10,21,16,17,21,12,12,11,20,12,11,21,11,21,37,12,10,16,30,21,21,16,10,30,10,16,16,10,10,21,12,10,11,24,32,21,16,21,11,18,16,26,30,17,21,29,22,16,21,10,21,21,10]

## STRAIGHT ##
#gen = [19,25,20,10,21,21,11,10,29,17,21,19,26,39,17,21,16,9,11,11,16,16,21,10,21,21,20,26,32,29,10,21,21,12,21,10,24,16,21,17,16,21,10,28,21,16,16,19,11,10,21,16,35,21,24,21,19,18,18,16,21,10,16,6,6,10,11,21,34,20,11,21,16,16,21,21,16,11,21,12,36,11,16,10,21,18,21,11,17,11,21,16,16,16,17,16,11,11,10,10]
gen = [21,34,21,29,10,32,35,25,10,16,16,11,10,16,21,16,21,10,21,21,21,23,22,17,11,21,16,21,17,10,20,21,21,21,11,21,30,31,21,21,10,10,16,21,21,21,17,11,20,20,13,16,16,25,37,16,21,11,9,16,16,11,20,11,20,14,14,21,11,27,31,25,11,16,16,11,11,21,11,20,21,21,21,16,21,10,4,11,16,11,23,16,10,11,30,21,40,21,20,1,11,21,17,16,16,27,16,11,11,21,16,10,16,10,16,21,21,30,19,21,16,21,10,21,16,16,21,16,11,23,21,11,11,21,21,31,34,17,21,24,24,51,28,10,11,21,11,17,11,16,38,16,21,10,11,26,21,11,37,16,17,11,10,12,13,21,21,17,21,16,19,19,34,17,16,21,21,10,6,21,3,17,17,10,16,22,11,11,16,10,16,21,16,11,21,23,20,6,21,16,32,21,11,11,17,16,11,11,16,16,21,21,11,21,21,21,21,29,10,11,15,21,17,16,21,21,32,16,21,21,26,21,11,21,21,21,30,11,25,20,14,19,17,10,19,21,17,10,17,19,10,10,21,31,16,16,25,11,16,10,21,20,11,11,12,10,16,21,21,21,16,21,21,11,15,17,16,20,11,21,10,21,21,21,24,16,11,21,21,24,40,21,21,16,17,16,11,21,17,22,10,30,21,16,16,16,21,10,20,21,16,20,16,21,21,10,16,10,21,21,17,16,21,39,21,25,10,17,16,21,16,22,5,11,16,10,10,10,21,21,19,21,16,22,16,19,16,21,12,21,21,26,21,10,17,12,19,21,10,16,17,16,19,11,10,18,16,21,23,21,18,10,10,21,1,11,30,16,22,22,16,16,11,21,21,11,16,21,16,21,34,21,11,16,19,10,26,17,21,22,21,16,17,21,12,16,21,12,10,11,21,10,15,21,16,19,24,10,27,17,17,26,11,16,29,21,10,21,31,21,24,22,11,21,15,21,34,11,6,21,10,17,34,21,16,16,19,17,19,30,11,11,21,16,21,11,11,21,17,21,16,16,16,32,21,22,16,21,11,16,16,16,10,17,21,22,21,20,30,17,40,22,21,19,10,21,10,11,11,21,16,16,16,27,21,10,11,21,25,21,10,10,21,20,10,17,21,25,16,16,21,16,16,17,11,11,21,16,33,11,12,11,21,21,21,11,16,21,11,21,17,11,16,22,16,18,21,28,11,17,10,19,12,21,20,37,34,21,16,21,17,26,11,20,16,19,17,10,27,11,26,21,16,16,10,12,11,11,20,16,30,22,19,38,10,21,20,16,12,11,25,20,9,16,10,21,16,17,21,21,17,31,10,12,11,16,16,16,21,21,20,25,10,17,16,21,21,19,17,21,19,22,11,10,11,16,32,17,11,27,25,21,16,21,16,21,11,16,10,21,17,17,26,10,20,24,19,23,21,10,11,21,11,10,11,16,10,11,11,21,16,21,21,24,21,21,17,19,16,21,39,16,11,21,22,21,16,11,25,22,11,16,26,11,21,21,21,40,12,21,21,10,21,18,7,11,21,16,21,11,21,16,21,16,11,24,26,21,16,16,19,29,27,10,11,21,21,20,21,21,10,11,21,22,21,21,16,21,17,16,22,16,21,22,16,16,10,30,26,17,28,11,11,11,21,16,37,16,16,21,18,10,25,15,20,10,21,17,17]
sns.kdeplot(man, common_norm=True, fill=True, alpha=0.4, label="Generated dm-s")  # plot first data series
sns.kdeplot(gen, common_norm=True, fill=True, alpha=0.4, label="Automatic dm-s")  # plot second data series

			# import matplotlib.pylab as pylab
			# params = {
			# 		'axes.labelsize': 14,
			# 		'xtick.labelsize':14,
			# 		'ytick.labelsize':14}
			# pylab.rcParams.update(params)

			# sns.histplot(dms_manual, common_norm=True, label="Generated dm-s")
			# sns.histplot(dms_measured, common_norm=True, label="Automatically measured dm-s")
        # sns.kdeplot(data=dms_man1, common_norm=True, fill=True, alpha=0.5,color ="yellow", label="Manual 1 dm-s")
        # sns.kdeplot(data=dms_man2, common_norm=True, fill=True, alpha=0.5,label="Manual 2 dm-s")

        # box plots # 
        #sns.boxplot(data=df)


#df = pd.DataFrame({"Iteration": np.arange(1,11), "Fiber diameter (nm)": dms_gen})
#fig,ax=plt.subplots()

#test1 = [664,272,545,545,664,664,545,391,545,664,1173,664,272,936,664,1055,1091,272,272,1055,1055,545,391,936,664,545,545,272,664,664,664,272,664,1173,545,782,545,391,664,818,664,664,545,818,1055,664,545,545,782,272,818,936,272,664,391,818,272,664,664,664,391,1564,936,391,664,664,664,545,391,818,818,272,1091,545,664,818,664,664,664,391,545,391,818,545,545,545,391,664,391,936,545,818,272,545,391,545,545,545,936,391]
#df = df.melt(id_vars="index")
#test2 = [272,664,782,545,782,391,272,936,664,782,664,664,664,391,664,391,664,818,391,664,545,1055,936,545,664,545,664,545,782,664,936,1055,664,272,664,545,272,272,818,391,664,664,391,782,664,545,272,545,1173,664,664,545,664,936,664,664,664,272,664,664,545,545,272,272,1209,782,545,936,664,545,664,664,272,545,545,545,1055,1091,272,272,782,272,545,391,545,545,936,664,545,545,391,664,391,545,664,545,664,664,664,664]
#sns.histplot((test1,test2),common_norm=True, multiple="layer").patch.set_alpha(0.5)

# colors=["#FF0000", "#FFA500","#FFFF00","#008000", "#0000FF", "#4B0082","#EE82EE", "#FF00FF", "#FF69B4", "#FF7F50"]
# colors = ["#FFC300","#E67E22", "#E74C3C", "#9B59B6", "#3498DB", "#1ABC9C","#2ECC71", "#F1C40F", "#F39C12", "#C0392B"] #,"#FFFFFF" ]



# for i,val in enumerate(dms_gen):
# 	plt.hist(dms_gen[i], alpha=0.5, color=colors[i], label=i+1)


#plt.hist(dms_gen[2], bins=12,alpha=0.3, histtype="stepfilled", color=colors[2])
#plt.hist(dms_gen[3], bins=12,alpha=0.3, histtype="stepfilled", color=colors[3])
#p = sns.histplot(dms_gen,common_norm=True, stat="proportion", element="bars", multiple="layer") #,label=i+1)


import matplotlib.pylab as pylab
params = {
		'axes.labelsize': 14,
		'xtick.labelsize':14,
		'ytick.labelsize':14}
pylab.rcParams.update(params)

# for i in range(len(dms_m)):
#     sns.kdeplot(data=dms_m[i], common_norm=True, fill=True, alpha=0.5, label=i+1)

# #plt.ylabel("Fiber diameter (pixels)")
# legend = ax.get_legend()
# handles = legend.legendHandles
# legend.remove()
# ax.legend(handles, ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"], title='Iteration #', loc="upper right", fontsize = 14)
plt.xlim(left=0) #, right=1000)
plt.ylabel("Relative frequency")
plt.xlabel("Fiber diameter (pixels)")
#plt.title('Generated vs automatically measured dm-s')
#plt.xlim((0,2000))
#ax.set_xticks(bins)
plt.legend(fontsize=14)
#plt.xlim(left=0)
#plt.show()
plt.savefig(f"{VIS_PATH}kde_straight_750.png", bbox_inches="tight")
plt.clf()