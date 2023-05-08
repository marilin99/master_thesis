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
RES_PATH =  "/home/marilin/Documents/ESP/data/fiber_tests/synthesised_fibers/three_dm_fibers_autom_results_4000/"
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

			sns.kdeplot(data=dms_manual, common_norm=True, fill=True, alpha=0.4, label="Generated dm-s")  # plot first data series
			sns.kdeplot(data=dms_measured, common_norm=True, fill=True, alpha=0.4, label="Automatically measured dm-s")  # plot second data series

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
			plt.ylabel("Frequency")
			plt.xlabel("Fiber diameter (pixels)")
			#plt.title('Generated vs automatically measured dm-s')
			#plt.xlim((0,2000))
			#ax.set_xticks(bins)
			plt.legend(fontsize=14)
			#plt.xlim(left=0)
			plt.show()
			#plt.savefig(f"{VIS_PATH}kde_new{var}.png", bbox_inches="tight")
			#plt.clf()