import matplotlib.pyplot as plt 
VIS_PATH = "/home/marilin/Documents/ESP/data/fiber_tests/fiber_test_1/visuals/"
lines = [391,272,391,272,391,272,391,272,391,545,391,391,545,272,272,272,272,272,272,272,272,391,272,272,391,272,272,272,272,391,391,272,272,391,272,272,272,272,272,272,272,545,272,664,391,545,272,391,272,272,272,391,272,391,391,272,272,936,272,391,272,272,545,272,272,545,272,272,272,272,391,272,272,272,391,818,272,272,545,272,272,272,272,391,272,272,272,272,818,272,272,272,272,272,272,272,391,272,391,391]
# with open("dm_info_2.txt", "r+") as file:
#     for line in file:
#         lines.append(int(line))


import seaborn as sns
import pandas as pd
import numpy as np

reds_0 = [0,11,4,27,3,3,1,4,12,1,3,15,2,5,2,0,13]
greens_0 = [0,0,1,5,5,0,5,12,11,4,12,14,2,3,3,0,0]
# greens 
reds_24 = [3,8,3,1,1,9,0,9]
greens_24 = [3,22,3,5,5,3,0,7]

# 
y = [305,293,287,602,308,242,240,219,539,274,338,272,288,442,301]
x= np.repeat([1,2,3,4,5], 3)

# plc 

# plc + peo

#df = pd.DataFrame({'0h_syto_PI': reds_0+greens_0, '24h_syto_PI': reds_24+greens_24})

#p = sns.stripplot(data=df, size=8)
#sns.boxplot(data=df, medianprops={'visible': True}, whiskerprops={'visible': False},showbox=False, showcaps=False, showfliers=False, ax=p)

import matplotlib.pylab as pylab
params = {
         'axes.labelsize': 14,
         'xtick.labelsize':14,
         'ytick.labelsize':14}

pylab.rcParams.update(params)

#regr line
# b, a = np.polyfit(greens_0,reds_0, deg=1)
# xseq = np.linspace(0,len(reds_0))


# b2, a2 = np.polyfit(greens_24, reds_24, deg=1)
# xseq2 = np.linspace(0,len(reds_0))
#sns.kdeplot(reds_0)
shifted_data = np.array(reds_0) - np.min(np.array(reds_0))

# Create the kdeplot with the shifted data
sns.kdeplot(shifted_data)
sns.despine()  # remove the top and right spines
sns.set_style("whitegrid")  # add horizontal grid lines
plt.xlim([0, np.max(shifted_data)]) 
# plt.scatter(greens_0, reds_0, label="0h_syto_PI", facecolors="none", edgecolors="b")
# plt.scatter(greens_24, reds_24, label ="24h_syto_PI", facecolors="y", edgecolors="y")
# plt.plot(xseq, a + b * xseq, color="b")
# plt.plot(xseq2, a2 + b2 * xseq, color="y")

#plt.scatter(x,y)
# plt.hist(lines)
#plt.title("Fiber diameter measurements (100)")
plt.ylabel("Amount of dead bacteria")
plt.xlabel("Amount of alive bacteria")
plt.legend(fontsize=14)
#plt.savefig(f"{VIS_PATH}box_plc.png", bbox_inches="tight")
# plt.clf()
plt.show()