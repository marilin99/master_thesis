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

    result = img.copy()
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    #cv2.imshow("thresh", np.uint8(thresh))

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
    
    #print(big_contour)

    # big_contour struc: [a][0][b], where a is the corner idx (runs from upper left + ↓ and upper right + ↓), b is either y [0] or x [1] coord
    # initial flattened big contour list y1 x1 y2 x2 etc.

    y_coords = big_contour.flatten()[::2]
    x_coords = big_contour.flatten()[1::2]

    # for drawing purposes
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
    #cv2.imshow("unit", np.uint8(unit))

    # extracting number
    number = img[min(x_coords):max(x_coords), min(y_coords): min(y_coords) + cutting_idx]
    number = np.pad(number, pad_width = [(1, 1),(1, 1)], mode = "constant")

    # cv2.imshow("number", np.uint8(number))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # segmentation technique - one line with many possible char-s
    # https://github.com/madmaze/pytesseract
    # https://muthu.co/all-tesseract-ocr-options/
    # takes an image as one single character
    custom_config= r'--psm 10' 

    #try:
    for val in range(10,30, 1):
        detected_nr = str(pytesseract.image_to_string(cv2.resize(number, (val, val)), config =custom_config, timeout = 5)).split("y\n\x0c")[0].strip() # Timeout after 2 seconds
        # give nr options
        #print("numero", detected_nr)
        if detected_nr in pot_values:
            value_unit_scale.append(int(detected_nr))
            # extra list in case more than one value is detected 
            counter.append(1)

        detected_unit = str(pytesseract.image_to_string(cv2.resize(unit, (val, val)), config =custom_config, timeout = 5)).split("y\n\x0c")[0].strip() # Timeout after 2 seconds
        #print(detected_unit)
        if detected_unit in pot_units:
            value_unit_scale.append(detected_unit)
            counter.append(2)

        #print(counter)
        #print(value_unit_scale)
        if len(np.unique(counter)) == 2:
            state_counts = np.unique(value_unit_scale, return_counts=True)[1]
            # checking for unique values and occurring once 
            if np.in1d(state_counts, 1).all():
                value_unit_scale = list(np.unique(value_unit_scale))
            else:
                max_num = np.unique(value_unit_scale)[np.argmax(state_counts)]
                #val = np.unique(value_unit_scale)[max_idx]
                #fin_unit = value_unit_scale[-1]
                # adding only the value
                if isinstance(max_num, int):
                    value_unit_scale = [max_num, detected_unit]
                else:
                    value_unit_scale = list(np.unique(value_unit_scale))

            break


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
