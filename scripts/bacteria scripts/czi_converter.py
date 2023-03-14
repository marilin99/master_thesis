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


# assuming the target dir-s exists

PATH = "/home/marilin/Documents/ESP/data/FM_SYTO/" #, "/home/marilin/Documents/ESP/data/FM_SYTO/"]  # change this once files in the network server
TARGET_PATH = "/home/marilin/Documents/ESP/data/FM_SYTO_conversion/"
#TARGET_PATH = "/home/marilin/Documents/ESP/data/FM_SYTO_conversion_test/"
FILES = os.listdir(PATH) 
#FILES = ["PCL_PEO_fibers_FM_syto_4.czi"]
FIN = []
#FIN = ["/home/marilin/Documents/ESP/data/FM_SYTO/PCL_fibers_FM_syto_9.czi"]
for f in FILES:
    # in case there is a supportive txt file or other format in the folder
    if "czi" in f:
        file = PATH + f
        FIN.append(file)
#         print(f)
#print(FIN)
#for file in FIN:
        
        aics = AICSImage(file)

        ##### czi global metadata collection #####
        colors = []
        czi = czifile.CziFile(file)
        metadatadict_czi = czi.metadata(raw = False)

        # could also fetch distance between z-stacks if needed
        
        # physical size assertion
        sanity_x  = metadatadict_czi['ImageDocument']['Metadata']['Experiment']['ExperimentBlocks']['AcquisitionBlock']['AcquisitionModeSetup']['ScalingX']
        sanity_y  = metadatadict_czi['ImageDocument']['Metadata']['Experiment']['ExperimentBlocks']['AcquisitionBlock']['AcquisitionModeSetup']['ScalingY'] # in meters

        # this is in um 
        scale_y = aics.physical_pixel_sizes[-2] * 10**-6
        scale_x = aics.physical_pixel_sizes[-1] * 10**-6

        # assertion that the aics module physical_pixel_sizes() works 
        assert(round(scale_x, 8) == round(sanity_x, 8) and round(scale_y, 8) == round(sanity_y, 8))

        detect_xml = metadatadict_czi['ImageDocument']['Metadata']['Experiment']['ExperimentBlocks']['AcquisitionBlock']['MultiTrackSetup']['TrackSetup']['Detectors']['Detector']
        # conditioning for general struc where the transmission channel is the last one - assuming the pmt structure
        if detect_xml[0]['Name'] != '':
            
            # wrongly assuming the first two channels are correct (could have a combination of Ch1, ChS1, Ch2 etc.) - can check the length of the detectors and iterate through - regex
            for i in range(len(detect_xml)):
                if "ChS*" and "T PMT" not in detect_xml[i]['ImageChannelName']:
                    ch_n_hex_color = detect_xml[i]['Color']
        
                    # lime is not suited with the next piece of code - possibly need some more color conditioning on this side in the future 
                    if webcolors.hex_to_name(ch_n_hex_color) == "lime":
                        colors.append("green")
                    # blue ch not needed in this case study - autofluor from fibers probs 
                    elif webcolors.hex_to_name(ch_n_hex_color) == "blue":
                        continue
                    else:
                        colors.append(webcolors.hex_to_name(ch_n_hex_color))
                    #print(webcolors.hex_to_name(ch_n_hex_color))
    
            ##### debugging area #####
            #print(len(metadatadict_czi['ImageDocument']['Metadata']['Experiment']['ExperimentBlocks']['AcquisitionBlock']['MultiTrackSetup']['TrackSetup']['Detectors']['Detector']))
            # print(aics.get_xarray_stack().shape)  # I,T,C,Z,Y,X
            # print(aics.dims)
            # print(colors)
            #####
            
            # xarray's dataarray object - making an assumption that the z stack is 3rd last el in tuple 
            # iterating through all of the z-stacks 
            for val in range(aics.get_xarray_stack().shape[-3]):

                arr1= aics.get_xarray_stack()[0,0,:,val,:,:] 
            
                # reshaping I,T,C,Y,X to Y,X,C,I,T 
                np_arr = arr1.to_numpy().astype(np.uint8).transpose(1,2,0)
              
                # only for pmt struc
                # 3 channel behavior
                if np_arr.shape[-1] == 3:
                    (ch1,ch2,T) = cv2.split(np_arr)
                # 4 channel behavior
                else: 
                    # metadata specific 
                    if metadatadict_czi['ImageDocument']['Metadata']['Experiment']['ExperimentBlocks']['AcquisitionBlock']['MultiTrackSetup']['TrackSetup']['Detectors']['Detector'][1]['Name'] == '':
                        # sometimes green, red 
                     
                        (ch1,_,ch2,T) = cv2.split(np_arr)

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
                f_name = f.split(".czi")[0]
                f_xml = open(f"{TARGET_PATH+f_name}.xml", "w")
                f_xml.write(czi.metadata())
                f_xml.close()

                ##### saving the img files #####

                cv2.imwrite(f"{TARGET_PATH+f_name}_{colors[0]}_{val}.png", ch1)
                cv2.imwrite(f"{TARGET_PATH+f_name}_{colors[1]}_{val}.png", ch2)
                cv2.imwrite(f"{TARGET_PATH+f_name}_transmission_{val}.png", T)

        else:
            # raise exception?
            print("Such color channel structure is not known")


##### visualisation #####

# cv2.imshow("green", G)
# cv2.imshow("red", R)
# cv2.imshow("transmission",T)
# cv2.waitKey(0)
# cv2.destroyAllWindows()