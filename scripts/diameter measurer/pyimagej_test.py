import numpy as np
import cv2 
import os 
import difflib
import matplotlib.pyplot as plt
import imagej
import scyjava as sj
import skimage.morphology 
from pykuwahara import kuwahara
from plantcv import plantcv as pcv
import math 
from scipy import ndimage
from scipy.ndimage import * 

# https://pyimagej.readthedocs.io/en/latest/Initialization.html

#PATH_1 = "/home/marilin/Documents/ESP/data/SEM/EcN_II_PEO_131120_GML_15k_03.tif"

# test set 1 (22.03)
PATH = "/home/marilin/Documents/ESP/data/fiber_tests/original_img/"
TARGET_PATH =  "/home/marilin/Documents/ESP/data/fiber_tests/segmented_img_class/"
FILES = os.listdir(PATH) 


###### huang func in py 

# https://github.com/dnhkng/Huang-Thresholding/blob/master/analysis.py

def huang(data):
    """Implements Huang's fuzzy thresholding method 
        Uses Shannon's entropy function (one can also use Yager's entropy function) 
        Huang L.-K. and Wang M.-J.J. (1995) "Image Thresholding by Minimizing  
        the Measures of Fuzziness" Pattern Recognition, 28(1): 41-51"""
    threshold = -1
    first_bin = 0
    for ih in range(254):
        if data[ih] != 0:
            first_bin = ih
            break
    last_bin = 254
    for ih in range(254, -1, -1):
        if data[ih] != 0:
            last_bin = ih
            break
    term = 1.0 / (last_bin - first_bin)
    mu_0 = np.zeros(shape=(254, 1))
    num_pix = 0.0
    sum_pix = 0.0
    for ih in range(first_bin, 254):
        sum_pix += (ih * data[ih])
        num_pix += data[ih]
        mu_0[ih] = sum_pix / num_pix  # NUM_PIX cannot be zero !
    mu_1 = np.zeros(shape=(254, 1))
    num_pix = 0.0
    sum_pix = 0.0
    for ih in range(last_bin, 1, -1):
        sum_pix += (ih * data[ih])
        num_pix += data[ih]
        mu_1[ih - 1] = sum_pix / num_pix  # NUM_PIX cannot be zero !
    min_ent = float("inf")
    for it in range(254):
        ent = 0.0
        for ih in range(it):
            # Equation (4) in Reference
            mu_x = 1.0 / (1.0 + term * math.fabs(ih - mu_0[it]))
            if not ((mu_x < 1e-06) or (mu_x > 0.999999)):
                # Equation (6) & (8) in Reference
                ent += data[ih] * (-mu_x * math.log(mu_x) - (1.0 - mu_x) * math.log(1.0 - mu_x))
        for ih in range(it + 1, 254):
            # Equation (4) in Ref. 1 */
            mu_x = 1.0 / (1.0 + term * math.fabs(ih - mu_1[it]))
            if not ((mu_x < 1e-06) or (mu_x > 0.999999)):
                # Equation (6) & (8) in Reference
                ent += data[ih] * (-mu_x * math.log(mu_x) - (1.0 - mu_x) * math.log(1.0 - mu_x))
        if ent < min_ent:
            min_ent = ent
            threshold = it
 
    return threshold

### NUMPY part #####
for f in FILES: 
	#print(file)
	file = PATH+f
	img_data = cv2.imread(file, 0)

	# kuwahara filter 
	kuwahara_img = kuwahara(img_data, method='mean', radius=2)

	histogram, bin_edges = np.histogram(kuwahara_img, bins=range(256))
	huang_th = huang(histogram)
	np_thuang = np.where(img_data > huang_th, 1, 0)


	# for k, v in os.environ.items():
	# 	if k.startswith("QT_") and "cv2" in v:
	# 	    del os.environ[k]

	threshold = cv2.erode(np.uint8(np_thuang),None, iterations=3)
	merged = cv2.dilate(threshold,None, iterations=3)

	merged = np.uint8(cv2.medianBlur(np.uint8(np_thuang), 15)) 
	merged[merged>0] = 255
    
	# saving tmp object map to folder 
	cv2.imwrite(f"{TARGET_PATH}{f}.png", np.uint8(merged))

	########

	#PATH_2 = cv2.medianBlur(cv2.imread("/home/marilin/Documents/ESP/data/unet_test/unet3_image_0.6.png",0), 15)

	# thinned = skimage.morphology.medial_axis(merged).astype(np.uint8)
	# thinned[thinned == 1] = 255

	# # removing skeleton hairs - https://plantcv.readthedocs.io/en/stable/prune/
	# pruned_skeleton, segmented_img, segment_objects = pcv.morphology.prune(skel_img=thinned, size=50)

	# object_map = pruned_skeleton

	# #removing ind floating areas from skeleton (8-component neighborhood)
	# object_map, count = ndimage.label(pruned_skeleton, structure = generate_binary_structure(2,2))

	# def pixelcount(regionmask):	return np.sum(regionmask)
	# props = skimage.measure.regionprops(object_map, extra_properties=(pixelcount,))

	# idxs = np.argwhere(np.array([props[val].pixelcount for val in range(len(props))]) == 1).ravel()
	# for val in idxs:	object_map[object_map == val] = 0 

	# object_map[object_map != 0] = 255
        
	# saving tmp object map to folder 


###### visuals ######

# cv2.imshow('segmented im', merged)
# cv2.imshow('orig im', img_data)
# cv2.imshow('thinned orig', thinned)
# # cv2.imshow('kuwahara', kuwahara_img)
# cv2.imshow('thinned im', np.uint8(object_map))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

####################

# HyperSphereShape = sj.jimport("net.imglib2.algorithm.neighborhood.CenteredRectangleShape")
# shape = HyperSphereShape(3, True)

#open = ij.op().morphology().open(tHuang, [shape])


#ij.py.show(median, "gray")

#print([op for op in ij.op().ops()])

# srm
#ij.IJ.run("8-bit")
#ij.IJ.run(imp, "Statistical Region Merging", "q=100 showaverages")

#ij.IJ.run(image, "Erode")

## FIJI ##
# ij.IJ.run(imp, "Kuwahara Filter", "sampling=1") # Look ma, a Fiji plugin!

# Prefs = sj.jimport('ij.Prefs')
# Prefs.blackBackground = True

# ij.IJ.setAutoThreshold(imp, "Huang ignore_white white")
# #mask = ij.py.to_imageplus(imp.createThresholdMask())
# ij.IJ.run(imp, "Close", "")
# #ij.py.show(mask)

# #ij.IJ.run(imp, "threshold.huang")
# #tHuang = ij.op().threshold().huang(k_im)

# # ij.IJ.run("8-bit")
# ij.py.show(imp, "gray")

# 

#print(ij.WindowManager.getIDList())

#print(ij.WindowManager.getImage("imp_fib") is not None)





#ij.py.show(imp, "gray")

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
#ij.py.show(mask, "gray")



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

###
# ## other way 
# macro = """
# run("DiameterJ Segment", "do=Yes image=1024 image_0=768 top=0 top_0=0 bottom=1024 bottom_0=650 stat. do_0=No choose=/home/marilin/Documents/ESP/diameterJ_test/sem_test");
# """
# # # Convert image to 8-bit
# ij.py.run_macro(macro, {'Image': image})

# ij.py.show(imp, "gray")