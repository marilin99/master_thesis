from posixpath import splitext
import matplotlib.pyplot as plt 
import numpy as np 
import aicspylibczi 
import os
import sys, glob 
from pathlib import Path

# assuming the data folder is in the same folder as this script
# FILES = os.listdir("/home/marilin/Documents/ESP/data") # change this once files in nextcloud or sth 

PATH = "data/15june2022/"

FILES = os.listdir(PATH)
FIN = []

for file in FILES:
	if "txt" not in file:
		file = PATH + file
		FIN.append(file)

print(FIN)

# function taken from https://github.com/AllenCellModeling/aicspylibczi
def norm_by(x, min_, max_):
    norms = np.percentile(x, [min_, max_])
    i2 = np.clip((x - norms[0]) / (norms[1] - norms[0]), 0, 1)
    return i2


for file in FIN:

    test = aicspylibczi.CziFile(file)
    img, shp = test.read_image()
    tmp_list = [x for x,_ in shp]
    print(tmp_list)
    # assuming that the img shape has B C Y X param, rgb conversion based on channel
    # z - z stacks and s - saturation values, T - time
    if tmp_list == ['B', 'C', 'Y', 'X'] or tmp_list == ['B', 'T', 'C', 'Z', 'Y', 'X'] :
        print("i")
        # assuming that the img shape has B C Y X param
        c1 = (norm_by(img[0, 0, 0, 0, :,:], 50, 99.8) * 255).astype(np.uint8)
        c2 = (norm_by(img[0, 0, 1, 0, :,:], 50, 99.8) * 255).astype(np.uint8)
        c3 = (norm_by(img[0, 0, 2, 0, :,:], 50, 99.8) * 255).astype(np.uint8)
        rgb = np.stack((c1, c2, c3), axis=2)

        # image saving 
        plt.imshow(rgb)
        plt.axis("off")
        plt.savefig("{}.tiff".format(splitext(file)[0]), format="tiff", bbox_inches="tight", pad_inches=0)
        plt.show()

    else: # pass for now 
        pass


