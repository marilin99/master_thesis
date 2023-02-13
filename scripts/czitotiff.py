from posixpath import splitext
import matplotlib.pyplot as plt 
import numpy as np 
import aicspylibczi 
import os
import sys, glob 
from pathlib import Path
from aicsimageio import AICSImage
from aicsimageio import writers
from aicsimageio import transforms
import cv2
import czifile
import tifffile
from scipy import ndimage
from scipy.ndimage import *

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

FIN = ["/home/marilin/Documents/ESP/data/16.08.22_dead_live/sack_fibers_NOgrowth_syto_PI_1.czi"]
#"/home/marilin/Documents/ESP/data/SYTO_PI/sack_fibers_NOgrowth_syto_PI_1-Image Export-10_z01c1-3.jpg"]

# function taken from https://github.com/AllenCellModeling/aicspylibczi
def norm_by(x, min_, max_):
    norms = np.percentile(x, [min_, max_])
    i2 = np.clip((x - norms[0]) / (norms[1] - norms[0]), 0, 1)
    return i2


for file in FIN:
    aics = AICSImage(file)
    # does not include units
    #print(aics.physical_pixel_sizes)
    print(aics.ome_metadata) # == .metadata
    
    # # xarray's dataarray object
    #aics.get_xarray_stack().shape - (1, 1, 3, 16, 512, 512) - I,T, C,Z,Y,X

    # first stack of Z
    arr1= aics.get_xarray_stack()[0,0,:,0,:,:] 
    print(arr1.shape)
    # #print(arr1.shape)
    # # reshaping I,T,C,Y,X to Y,X,C,I,T - outputs red channel for all channels for this reader
    np_arr = arr1.to_numpy().astype(np.uint8).transpose(1,2,0)

    (G,R,T) = cv2.split(np_arr)
    merged = cv2.cvtColor(T, cv2.COLOR_GRAY2BGR)
    merged[:,:,1] = 0
    merged[:,:,1] += G
    merged[:,:,2] = 0
    merged[:,:,2] += R
    merged[:,:,0] = 0
    merged = cv2.cvtColor(merged, cv2.COLOR_BGR2GRAY)



    # thresh = cv2.adaptiveThreshold(merged, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,9,2)
    # edged = cv2.erode(thresh, (3,3), iterations=1)

    # normalizing ?
    # linear transform 

    # #  adding G+R
    # merged = np.zeros((T.shape))
    # merged[:,1] += G
    # merged[:,2] += R

    # #print(np_arr)
    # #print(trans_arr1)
    # r_channel = np_arr[:,:,0,0,0]
    # g_channel = np_arr[:,:,1,0,0]
    # b_channel = np_arr[:,:,2,0,0]

    ############

    #czi = czifile.CziFile(file)

    # czi global metadata 
    # metadatadict_czi = czi.metadata()
    # f =  open("myxmlfile.xml", "w")
    # f.write(metadatadict_czi)
    # f.close()
    # # memory mappable tiff file
    # czitiff = czifile.czi2tif(file)


    #czi_str = czi.__str__()
    #print(czi_arr)

# file2 = "/home/marilin/Documents/ESP/data/16.08.22_dead_live/sack_fibers_NOgrowth_syto_PI_1.czi.tif"

# # shape is C, Z, Y, X
# tif_file  =tifffile.tifffile.imread(file2)

# print(tif_file.shape)

# rgb = tif_file[:,0,:,:].transpose(1,2,0)
# (G,R,T) = cv2.split(rgb)

# merged = cv2.cvtColor(T, cv2.COLOR_GRAY2BGR)
# merged[:,:,1] += G
# merged[:,:,2] += R

# #cv2.imshow("test", rgb)
# cv2.imwrite("/home/marilin/Documents/ESP/data/16.08.22_dead_live/sack_fibers_NOgrowth_syto_PI_1_slice_1_merged_wrong.tif", merged)
#tif_page = tifffile.TiffPage(file2,0)
# from libtiff import TIFF, TIFFfile, TIFFimage

# tif = TIFF.open(file2, mode='r')

# img = tif.read_image(tif)

# #print(img)

# #for bgr in range(0,len(list(tif.iter_images()),3)):

# B = list(tif.iter_images())[6]
# G = list(tif.iter_images())[7]
# R = list(tif.iter_images())[8]

# merged = cv2.merge([B, G, R])

# (B, G, R) = cv2.split(merged)

# tif_file = TIFFfile(file2)
# samples, sample_names = tif_file.get_samples()


#cv2.imread(file)
# # show each channel individually
cv2.imshow("green", G)
cv2.imshow("red", R)
cv2.imwrite("green_intensities_24h_growth.png", G)
cv2.imwrite("red_intensities_24h_growth.png", R)
#cv2.imwrite("merged_intensities_incub_4h_1.png", merged)
#cv2.imshow("transmission",T)
#cv2.imshow("Merged", merged)
#cv2.imshow("res", result)
# cv2.imshow("thresh", thresh)
# cv2.imshow("edges", edged)
#cv2.imshow("orig", cv2.imread(FIN[-1]))
cv2.waitKey(0)
cv2.destroyAllWindows()

# file = cv2.imread("testing2.tif")
# cv2.imshow("orig", file)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

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


