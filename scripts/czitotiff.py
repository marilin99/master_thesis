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
import webcolors


# assuming the target dir-s exist 

# assuming the data folder is in the same folder as this script
PATH = "/home/marilin/Documents/ESP/data/SYTO_PI/" #, "/home/marilin/Documents/ESP/data/FM_SYTO/"]  # change this once files in the network server
TARGET_PATH = "/home/marilin/Documents/ESP/data/SYTO_PI_conversion/"
FILES = os.listdir(PATH) 
FIN = []

for file in FILES:
    # in case there is a supportive txt file or other format in the folder
	if "czi" in file:
		file = PATH + file
		FIN.append(file)
                
print(FIN)


for file in FIN:

    aics = AICSImage(file)

    ##### czi global metadata collection #####
    colors = []
    czi = czifile.CziFile(file)
    metadatadict_czi = czi.metadata(raw = False)
    
    # physical size assertion
    sanity_x  = metadatadict_czi['ImageDocument']['Metadata']['Experiment']['ExperimentBlocks']['AcquisitionBlock']['AcquisitionModeSetup']['ScalingX']
    sanity_y  = metadatadict_czi['ImageDocument']['Metadata']['Experiment']['ExperimentBlocks']['AcquisitionBlock']['AcquisitionModeSetup']['ScalingY'] # in meters

    # this is in um 
    scale_y = aics.physical_pixel_sizes[-2] * 10**-6
    scale_x = aics.physical_pixel_sizes[-1] * 10**-6

    # assertion that the aics module physical_pixel_sizes() works 
    assert(round(scale_x, 8) == round(sanity_x, 8) and round(scale_y, 8) == round(sanity_y, 8))

    # conditioning for general struc where the transmission channel is the last one - assuming the pmt structure
    if metadatadict_czi['ImageDocument']['Metadata']['Experiment']['ExperimentBlocks']['AcquisitionBlock']['MultiTrackSetup']['TrackSetup']['Detectors']['Detector'][0]['Name'] != '':
        for i in range(2):
            ch_n_hex_color = metadatadict_czi['ImageDocument']['Metadata']['Experiment']['ExperimentBlocks']['AcquisitionBlock']['MultiTrackSetup']['TrackSetup']['Detectors']['Detector'][i]['Color']
            # lime is not suited with the next piece of code - possibly need some more color conditioning on this side in the future 
            if webcolors.hex_to_name(ch_n_hex_color) == "lime":
                colors.append("green")
            else:
                colors.append(webcolors.hex_to_name(ch_n_hex_color))

        # xarray's dataarray object - making an assumption that the z stack is 3rd last el in tuple 
        # iterating through all of the z-stacks 
        #  print(aics.get_xarray_stack().shape)  # I,T,C,Z,Y,X

        for val in range(aics.get_xarray_stack().shape[-3]):

            arr1= aics.get_xarray_stack()[0,0,:,val,:,:] 
        
            # reshaping I,T,C,Y,X to Y,X,C,I,T - outputs red channel for all channels for this reader
            np_arr = arr1.to_numpy().astype(np.uint8).transpose(1,2,0)

            # only for pmt struc
            (ch1,ch2,T) = cv2.split(np_arr)

            # for merging purposes 
            merged = cv2.cvtColor(T, cv2.COLOR_GRAY2BGR)
            merged[:,:,1] = 0
            merged[:,:,1] += ch1
            merged[:,:,2] = 0
            merged[:,:,2] += ch2
            merged[:,:,0] = 0
            merged = cv2.cvtColor(merged, cv2.COLOR_BGR2GRAY).astype(np.uint8)

            # saving images and xml to specified folder 
                
            # saving xml just in case 
            # f =  open(f"{<TODO>}.xml", "w")
            # f.write(czi.metadata())
            # f.close()


    # color collection - so far (5.03.23) only red and lime aka green are known 

    # saving xml just in case 
    # f =  open("myxmlfile_24_sytopi.xml", "w")
    # f.write(czi.metadata)
    # f.close()





##### visualisation #####

# cv2.imshow("green", G)
# cv2.imshow("red", R)
# cv2.imshow("transmission",T)

##### saving the file #####

#for color in colors: 
#cv2.imwrite("green_intensities_incub4h_3.png", G)
#cv2.imwrite("red_intensities_incub4h_3.png", R)
#cv2.imwrite("transmission_intensities_incub4h_3.png", T)

###### 

cv2.waitKey(0)
cv2.destroyAllWindows()


#############################################################
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


