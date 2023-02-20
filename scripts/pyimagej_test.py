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
plugins_dir = '/home/marilin/Documents/ESP/diameterJ_test/ImageJ/plugins/DiameterJ/'

sj.config.add_option(f'-Dplugins.dir={plugins_dir}')

ij = imagej.init('sc.fiji:fiji')
#ij = imagej.init()

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

#huang thresholding from imagejops
#ij.op().threshold().huang(image)

#HyperSphereShape = sj.jimport("net.imglib2.algorithm.neighborhood.CenteredRectangleShape")

#ij.IJ.run(imp, "Kuwahara Filter", "sampling=1") # Look ma, a Fiji plugin!
# #ij.IJ.run(imp, "Statistical Region Merging", "q=12 showaverages")
# #ij.IJ.run(imp, "8-bit")
#mask = ij.op().run("threshold.shanbhag", imp)

# macro.jim recorded from imagej
# macro = """

# """
# py recorded from imagej tool 
# #ij.IJ.run(imp, "DiameterJ Segment", "do=Yes image=1024 image_0=768 top=0 top_0=0 bottom=1024 bottom_0=650 stat. do_0=No choose=/home/marilin/Documents/ESP/diameterJ_test/sem_test")


# ij.IJ.run(imp, "Remove Outliers...", "radius=3 threshold=50 which=Dark")
# ij.IJ.run(imp, "Remove Outliers...", "radius=3 threshold=50 which=Bright")
# ij.IJ.run(imp, "Remove Outliers...", "radius=3 threshold=50 which=Dark")
# ij.IJ.run(imp, "Remove Outliers...", "radius=3 threshold=50 which=Bright")

# ij.IJ.run(imp, "Erode")
# ij.IJ.run(imp, "Dilate")

#mask = ij.op().run("")
#mask = ij.op().run("morphology.fillHoles", mask)
ij.py.show(mask, "gray")



#ij.op().threshold().huang(imp)

# Prefs = sj.jimport('ij.Prefs')
# Prefs.blackBackground = False
# ij.IJ.setAutoThreshold(imp, "Otsu dark")
# mask = ij.to_imageplus()("cells-mask", imp.createThresholdMask())
# ij.IJ.run(imp, "Close", "")
# ij.py.show(mask)

#ij.py.show(imp, "gray")


# get numpy version of object 
# np_thuang = ij.py.from_java(thuang)

# print((np_thuang).shape)


#ij.py.run_macro("""run("Erode");""")

#ij.py.show(image, "gray")


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

# plugin = "DiameterJ Segment"
# ij.py.run_plugin(plugin, {"do":"Yes", "image":1024, "image_0":768, "top":0, "top_0":0, "bottom":1024, "bottom_0":650, "stat. do_0":"No", "choose":"/home/marilin/Documents/ESP/diameterJ_test/sem_test"}, imp = imp)


# import imageio
# import imagej

# # Start ImageJ
# ij = imagej.init('1.5.3')

# # Load image
# image = imageio.imread(PATH_1)

# # Convert image to 8-bit
# image = (image / image.max() * 255).astype('uint8')

# # Run DiameterJ macro
# macro = """
# run("DiameterJ Segment ", "do=Yes image=1024 image_0=768 top=0 top_0=0 bottom=1024 bottom_0=650 stat. do_0=No choose=/home/marilin/Documents/ESP/diameterJ_test/sem_test");
# """
# ij.py.run_macro(macro, {'Image': image})

# # Stop ImageJ
# ij.exit()

# result = ij.WindowManager.getCurrentImage()
# ij.py.show(result)