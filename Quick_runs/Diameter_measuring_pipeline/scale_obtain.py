# CODE for obtaining the scale from the SEM image #
#!/usr/bin/python3
# libraries
import numpy as np
import cv2
import os
import difflib
import matplotlib.pyplot as plt
import pytesseract
import re

# pot_val, pot_unit, scale length in px
value_unit_scale = []
# list of units
pot_units = ["nm", "um"]
# pot scale values - should be updated if some new scales used!
pot_values = ["1", "2", "3", "4", "10", "20", "30", "100", "200", "400"]
pot_digits = ["0", "1", "2","3","4"]


# INSERT YOUR tesseract.exe PATH here in case having an error
# make sure tesseract OCR is installed too
# uncomment for lab pc
#pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
#pytesseract.pytesseract.tesseract_cmd = r'C:/home/marilin/fibar_tool_venv/bin/pytesseract'



def scale_obtain(file):

    """
    Function for obtaining the scale from the original SEM input image
    """

    img = cv2.imread(file, 0)

    counter, value_unit_scale, four_contours = [], [], []

    ###########################################################################
    # number and unit obtaining

    thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV)[1]

    # assuming the scale is in the LL quarter 
    empty_im = np.zeros_like(thresh)
    cont_thresh = thresh[thresh.shape[0]//2:,:thresh.shape[1]//2]
    #cv2.imshow("cont_thresh", cont_thresh)
    empty_im[empty_im.shape[0]//2:,:empty_im.shape[1]//2]  = cont_thresh




    result = img.copy()
    contours = cv2.findContours(empty_im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    #cv2.imshow("empty_im", np.uint8(empty_im))
    contours = contours[0] if len(contours) == 2 else contours[1]
    #print(contours)

    
    for contour in contours:
        if len(contour) == 4:
            four_contours.append(contour)
    
    #print(four_contours)
    try:
        big_contour = max(four_contours, key=cv2.contourArea)
    except: 
        big_contour = max(contours, key=cv2.contourArea)
    

    # big_contour struc: [a][0][b], where a is the corner idx (runs from upper left + ↓ and upper right + ↓), b is either y [0] or x [1] coord
    # initial flattened big contour list y1 x1 y2 x2 etc.

    y_coords = big_contour.flatten()[::2]
    x_coords = big_contour.flatten()[1::2]

    # for drawing purposesoppenheimer aaron hibell trance remix
    if len(big_contour) > 4:
        big_contour = np.array([ [[min(y_coords), max(x_coords)]], [[min(y_coords), min(x_coords)]], [[max(y_coords), min(x_coords)]], [[max(y_coords), max(x_coords)]] ])

    cv2.drawContours(result, [big_contour], 0, (255,255,255), 5)

    #cv2.imshow("contours", np.uint8(result))


    # # 1px height cross-section
    cross = img[int( (min(x_coords)+ max(x_coords)) //2 ): int( (min(x_coords)+ max(x_coords)) //2 )+1, min(y_coords):max(y_coords)+5]

    ar_z = np.insert(np.nonzero(cross), 0, 0)
    # diff between non-zero values - fetching the zero amount of zeros and fetching the
    # start idx of the longest sub-arr of consecutive zeros
    min_val = ar_z[np.argmax(np.ediff1d(ar_z))]
    # end idx of the zero arr
    max_val = ar_z[np.argmax(np.ediff1d(ar_z)) +1]

    cutting_idx = int(np.mean((min_val, max_val)))


    # extracting unit
    unit = img[min(x_coords):max(x_coords), min(y_coords) + cutting_idx:max(y_coords)]
    unit = np.pad(unit, pad_width = [(1, 1),(1, 1)], mode = "constant")
    #unit = skimage.morphology.medial_axis(unit).astype(np.uint8)
    unit[unit == 1] = 255
    # cv2.imshow("unit", np.uint8(unit))

    # extracting number
    number = img[min(x_coords):max(x_coords), min(y_coords): min(y_coords) + cutting_idx]
    #cv2.imshow("number", np.uint8(unit))
    number = np.pad(number, pad_width = [(1, 1),(1, 1)], mode = "constant")

    ## number to digits ## 
    first_row_of_whites = np.min(np.nonzero(number)[0])
    full_extent = number.shape[1]

    nr_cross = number[first_row_of_whites: first_row_of_whites+1, :full_extent].flatten()
    # print(nr_cross)
    nonz_locs = np.nonzero(nr_cross)[0]
    # print(nonz_locs)

    # print(int(number.shape[0] //2 ))
    
    # nr_cross = number[int(number.shape[0] //2 ): int( number.shape[0] //2 )+1, :number.shape[1]]
    # print(nr_cross)
    nr_ar_z = np.insert(np.nonzero(nr_cross), 0, 0)
    diff_between_zeros = np.ediff1d(nr_ar_z) 
    # print(diff_between_zeros)
    idxs = np.where(diff_between_zeros > 1)[0]
    # print(idxs)
    cuts = []
    for val in idxs[idxs!=0]:
        cuts.append(int(np.mean((nonz_locs[val], nonz_locs[val-1]))))
    
    cuts.insert(0,0)
    cuts.append(number.shape[1])
    custom_config= r'--psm 10' 

    compound_nr = ""
    
    for i, cut in enumerate(cuts):
        if i != len(cuts)-1:
            detected_nr, detected_unit = [], []

            digit = number[:number.shape[0], cut:cuts[i+1]]
            digit = np.pad(digit, pad_width = [(1, 1),(1, 1)], mode = "constant")
             
            # cv2.imshow("digit", np.uint8(digit))
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            for val in range(10,30, 1):
                detected_nr.append(str(pytesseract.image_to_string(cv2.resize(digit, (val, val)), config =custom_config, timeout = 5)).split("y\n\x0c")[0].strip()) # Timeout after 2 seconds
                if i == 0:
                    detected_unit.append(str(pytesseract.image_to_string(cv2.resize(unit, (val, val)), config =custom_config, timeout = 5)).split("y\n\x0c")[0].strip()) # Timeout after 2 seconds

            for unit in detected_unit:
                if unit in pot_units:
                    value_unit_scale.append(unit)

                    # found_unit = max(set(detected_unit), key=detected_unit.count)
                    # if found_unit in pot_units:
                    #     value_unit_scale.append(found_unit)
        
            detected_ints = np.array([int(val) for val in detected_nr if val.isnumeric() and val in pot_digits])
            vals, counts = np.unique(detected_ints, return_counts=True)
            
            compound_nr += str(vals[np.argmax(counts)])


        else:
            pass
    
    value_unit_scale.append(int(compound_nr))
    value_unit_scale = list(np.unique(value_unit_scale))

    # except RuntimeError as timeout_error:
    # # Tesseract processing is terminated
    #     pass
    ################################################################

    # contouring the scale
    th2 = cv2.threshold(img[max(x_coords):, min(y_coords)-5: max(y_coords) + 200], 254, 255, cv2.THRESH_BINARY)[1]

    ## line length in pixels
    try:
        scale_length = np.max(np.nonzero(th2)[1]) - np.min(np.nonzero(th2)[1])
        value_unit_scale.append(scale_length)
        print(value_unit_scale)
        return value_unit_scale
    # in case no unit or number is found
    except:
        return None
    
    
#ORIG_PATH = "/run/user/1000/gvfs/smb-share:server=gaia.domenis.ut.ee,share=mvfa/Automaatika/SEM_processed/"
# ORIG_PATH = "/home/marilin/fibar_tool/Data/SEM_images/"
# FILES = os.listdir("/home/marilin/fibar_tool/Data/SEM_images")

# for file in FILES:
#    scale_obtain(ORIG_PATH+file)


# # scale_obtain("/run/user/1000/gvfs/smb-share:server=gaia.domenis.ut.ee,share=mvfa/Automaatika/SEM_input/3gel 02kit hfip 10k 1.tif")
# cv2.waitKey(0)
# cv2.destroyAllWindows()
