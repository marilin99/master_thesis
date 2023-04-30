import numpy as np 
import os
from aicsimageio import AICSImage
import cv2
import czifile
from scipy.ndimage import *
import webcolors



        
def converter(file):
    aics = AICSImage(file)

    ##### czi global metadata collection #####
    colors = []
    variables = {}

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

        # creating separate lists for channels based on the colors
        for c in colors:
            variables[f"ch_{c}"] = []
  
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

            variables[f"ch_{colors[0]}"].append(np.uint8(ch1))
            variables[f"ch_{colors[1]}"].append(np.uint8(ch2))

        return variables.items() #[f"ch_{colors[0]}"], variables[f"ch_{colors[1]}"], colors
    
    else:
        # raise exception?
        return ["Such color channel structure is not known"]