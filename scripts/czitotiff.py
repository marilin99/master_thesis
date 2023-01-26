from posixpath import splitext
import matplotlib.pyplot as plt 
import numpy as np 
import aicspylibczi 
import os
import sys, glob 
from pathlib import Path
from aicsimageio import AICSImage

# assuming the data folder is in the same folder as this script
#FILES = os.listdir("/home/marilin/Documents/ESP/data/16.08.22_dead_live") # change this once files in nextcloud or sth 

# PATH = "/home/marilin/Documents/ESP/data/16.08.22_dead_live/"

# FILES = os.listdir(PATH)
# FIN = []

# for file in FILES:
# 	if "txt" not in file:
# 		file = PATH + file
# 		FIN.append(file)

# print(FIN)

FIN = ["/home/marilin/Documents/ESP/data/16.08.22_dead_live/control_50killed_syto_PI_1.czi"]

# function taken from https://github.com/AllenCellModeling/aicspylibczi
def norm_by(x, min_, max_):
    norms = np.percentile(x, [min_, max_])
    i2 = np.clip((x - norms[0]) / (norms[1] - norms[0]), 0, 1)
    return i2




# for file in FIN:

#     test = aicspylibczi.CziFile(file)
#     _, shp = test.read_image()
#     tmp_list = [x for x,_ in shp]
#     print(test.size)
#     print(tmp_list)

#     test_dict = dict(map(lambda i,j : (i,j) , tmp_list, test.size))
#     # assuming that the img shape has B C Y X param, rgb conversion based on channel
#     # z - z stacks and s - saturation values, T - time
#     if tmp_list == ['B', 'C', 'Y', 'X'] or tmp_list == ['B', 'T', 'C', 'Z', 'Y', 'X'] :
#         for slice in range(test_dict["Z"]):
#             img, _ = test.read_image(Z=slice)
#             # assuming that the img shape has B C Y X param
#             c1 = (norm_by(img[0, 0, 0, 0, :,:], 50, 99.8) * 256).astype(np.uint8)
#             c2 = (norm_by(img[0, 0, 1, 0, :,:], 50, 99.8) * 256).astype(np.uint8)
#             c3 = (norm_by(img[0, 0, 2, 0, :,:], 50, 99.8) * 256).astype(np.uint8)
#             rgb = np.stack((c1, c2, c3), axis=2)
#             # turning blue channel off?
#             rgb[:,:,2] = np.zeros([rgb.shape[0], rgb.shape[1]])
#             # image saving 
#             plt.imshow(rgb)
#             plt.axis("off")
#             plt.savefig("{fname}_z{s}.tiff".format(fname=splitext(file)[0], s=slice), format="tiff", bbox_inches="tight", pad_inches=0)
#             plt.show()

#     else: # pass for now 
#         pass


