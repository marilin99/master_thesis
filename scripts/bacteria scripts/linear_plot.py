import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
# x = np.linspace(0,180,180)
# y = x

# # my approach
# t = [8,31,13,95,99,101,98,4,5,14,173,80,122,177,10,85,12,53,6,12,11,93,49,15,9,100,41,11,96,4]
# u = [7,31,12,84,91,90,92,4,5,14,139,72,105,141,10,77,12,51,6,12,11,83,48,15,9,91,41,11,92,4]

# # imagej approach
# k = [8,31,13,95,99,101,98,4,5,14,173,80,122,177,10,85,12,53,6,12,11,93,49,15,9,100,41,11,96,4]
# l = [7,31,12,84,90,94,91,4,5,14,151,72,114,141,10,83,12,50,6,12,11,83,48,15,9,96,41,11,90,4]


# reds 
reds_0 = [0,11,4,27,3,3,1,4,12,1,3,15,2,5,2,0,13]
greens_0 = [0,0,1,5,5,0,5,12,11,4,12,14,2,3,3,0,0]
# greens 
reds_24 = [3,8,3,1,1,9,0,9]
greens_24 = [3,22,3,5,5,3,0,7]

reds_peo = [4,0,4,2,1,1,9,1,22,15]
greens_peo = [0,0,0,7,7,5,0,7,4,13]

reds_pcl= [1,11,9,4,17,3,7,4,4,5]
greens_pcl = [1,5,3,2,12,0,1,7,4,11]

df = pd.DataFrame({'Amount of red bacteria': reds_peo, 'Amount of green bacteria': greens_peo})
df1 = pd.DataFrame({'Amount of red bacteria': reds_pcl, 'Amount of green bacteria': greens_pcl})


import matplotlib.pylab as pylab
params = {
         'axes.labelsize': 14,
         'xtick.labelsize':14,
         'ytick.labelsize':14}
pylab.rcParams.update(params)


sns.kdeplot(data=df, x='Amount of red bacteria', y='Amount of green bacteria', shade=True, legend =True)
plt.xlabel('Amount of red bacteria')
plt.ylabel('Amount of green bacteria')
plt.savefig("/home/marilin/Documents/ESP/data/bacteria_tests/kde_dens_peo.png")
plt.clf()

sns.kdeplot(data=df1, x='Amount of red bacteria', y='Amount of green bacteria', shade=True, legend =True)
plt.savefig("/home/marilin/Documents/ESP/data/bacteria_tests/kde_dens_pcl.png")

# plt.scatter(reds_0,greens_0, label="0h after staining")
# plt.scatter(reds_24, greens_24, label ="24h after staining")
# #plt.plot(x, y, '--k',label='Ideal case')
# #plt.title('Graph of y=2x+1')
# plt.xlabel('Amount of red bacteria')
# plt.ylabel('Amount of green bacteria')
# plt.legend(loc='upper left')
#plt.show()


## % differences 
# print( np.mean(np.abs(( np.array( t)- np.array( u) ) / np.array(t)) *100))
# print( np.mean(np.abs(( np.array( k)- np.array( l) ) / np.array(k)) *100))

# [12.5         0.          7.69230769 11.57894737  8.08080808 10.89108911
#   6.12244898  0.          0.          0.         19.65317919 10.
#  13.93442623 20.33898305  0.          9.41176471  0.          3.77358491
#   0.          0.          0.         10.75268817  2.04081633  0.
#   0.          9.          0.          0.          4.16666667  0.        ]

# [12.5         0.          7.69230769 11.57894737  9.09090909  6.93069307
#   7.14285714  0.          0.          0.         12.71676301 10.
#   6.55737705 20.33898305  0.          2.35294118  0.          5.66037736
#   0.          0.          0.         10.75268817  2.04081633  0.
#   0.          4.          0.          0.          6.25        0.        ]