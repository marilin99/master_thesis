import numpy as np
import cv2 
import os 
import difflib
import matplotlib.pyplot as plt
import imagej
import scyjava as sj

for k, v in os.environ.items():
	if k.startswith("QT_") and "cv2" in v:
	    del os.environ[k]


# https://pyimagej.readthedocs.io/en/latest/Initialization.html
PATH_1 = "/home/marilin/Documents/ESP/data/SEM/EcN_II_PEO_131120_GML_15k_01.tif"
plugins_dir = '/home/marilin/Documents/ESP/diameterJ_test/ImageJ/plugins'

sj.config.add_option(f'-Dplugins.dir={plugins_dir}')
ij = imagej.init('sc.fiji:fiji')


# load a sample image
image = ij.io().open(PATH_1)

if ij.WindowManager.getIDList() is None:
    ij.py.run_macro('newImage("dummy", "8-bit", 1, 1, 1);')

# imageplus 
imp = ij.py.to_imageplus(image)
imp.setTitle("imp_fib")

# srm
# ij.IJ.run(imp, "Statistical Region Merging", "q=100 showaverages")


# ij.IJ.run("8-bit")
# ij.py.show(image, "gray")

# 

#print(ij.WindowManager.getIDList())

#print(ij.WindowManager.getImage("imp_fib") is not None)





#ij.py.show(imp)

#print(imp.getAllStatistics())

# huang thresholding from imagejops
#thuang = ij.op().threshold().huang(image)

# uses pyplot for showing
# huang thresholding from imagejops
#ij.py.show(thuang, "gray")

# despeckle 

# erode 
#dilate

# for showing UI
# ij.ui().showUI()
# ij.ui().show(image)

#print("hey" if ij.ui().isHeadless() else "ney")

plugin = 'DiameterJ'
ij.py.run_plugin(plugin,)


result = ij.WindowManager.getCurrentImage()
ij.py.show(result)