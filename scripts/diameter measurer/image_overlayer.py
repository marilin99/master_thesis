from posixpath import splitext
import matplotlib.pyplot as plt
import numpy as np
import aicspylibczi
import os
import sys
import glob
from pathlib import Path
import cv2

PATH = "/home/marilin/Documents/ESP/data/dataset/"

FILES = os.listdir(PATH)
BG_IMG = []
FG_IMG = []

for file in FILES:
    if "Foreground" in file: 
        file = PATH + file 
        FG_IMG.append(file)
    else:
        if "Background" in file:
            file = PATH + file
            BG_IMG.append(file)


merged_image = np.zeros_like(cv2.imread("/home/marilin/Documents/ESP/data/dataset/task-1-annotation-6-by-1-tag-Foreground-42.png",0))

for file in FG_IMG:

    file = cv2.imread(file,0)
    merged_image  = cv2.bitwise_or(file, merged_image, mask = None)



cv2.imwrite("/home/marilin/Documents/ESP/data/dataset/foreground.png", merged_image)

merged_image = np.zeros_like(cv2.imread("/home/marilin/Documents/ESP/data/dataset/task-1-annotation-6-by-1-tag-Foreground-42.png",0))

for file in BG_IMG:

    file = cv2.imread(file,0)
    merged_image  = cv2.bitwise_or(file, merged_image, mask = None)

cv2.imwrite("/home/marilin/Documents/ESP/data/dataset/background.png", merged_image)
