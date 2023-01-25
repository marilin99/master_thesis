import numpy as np
import cv2 
import os 
import difflib
import matplotlib.pyplot as plt
import imagej
import scyjava as sj

PATH_1 = "/home/marilin/Documents/ESP/data/SEM/EcN_II_PEO_131120_GML_15k_01.tif"
plugins_dir = '/home/marilin/Documents/ESP/diameterJ_test/ImageJ/plugins'

sj.config.add_option(f'-Dplugins.dir={plugins_dir}')
ij = imagej.init()


# load a sample image
image = ij.io().open(PATH_1)

plugin = 'DiameterJ'

ij.py.run_plugin(plugin)

result = ij.WindowManager.getCurrentImage()
result = ij.py.show(result)