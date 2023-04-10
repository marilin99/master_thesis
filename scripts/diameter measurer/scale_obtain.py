import numpy as np
import cv2 
import os 
import difflib
import matplotlib.pyplot as plt
import pytesseract
import re

# PATH_1 = "/home/marilin/Documents/ESP/data/SEM/EcN_II_PEO_131120_GML_15k_01.tif" # 400
# PATH_1 = "/home/marilin/Documents/ESP/data/SEM/Lactis_PEO_111220_20k04.tif" # 1
# PATH_1 =  "/home/marilin/Documents/ESP/data/SEM/Lactis_PEO_111220_20k03.tif" # 2 
# PATH_1 =  "/home/marilin/Documents/ESP/data/SEM/EcN_II_PEO_131120_GML_5k_02.tif" # small scale

# PATH_1 = "/home/marilin/Documents/ESP/data/SEM/PEO_EcN_I_181220_MM_2k_02.tif" #10
# PATH_1 = "/home/marilin/Downloads/1_15_8020_13_3000(2).tif" # thinner font 10 um 
# PATH_1 = "/home/marilin/Downloads/CE_1_PCL_15KV_121.tif" #thinner font 200 nm
# PATH_1 = "/home/marilin/Downloads/CE_4_PCL_11KV_61.tif" #thinner font for 2
#PATH_1 = "/home/marilin/Downloads/PCL_15_11k_ACDCM_5_5_65%_4k_1.tif" # zeiss format not working

#pytesseract.pytesseract.tesseract_cmd =r"C:/Program Files/Tesseract-OCR/tesseract.exe"

# pot_val, pot_unit, scale length in px
value_unit_scale = []
# list of units
pot_units = ["nm", "um"]
# pot scale values - should be updated if some new scales used!
pot_values = ["10", "1", "2", "400", "200", "3"]


def scale_obtain(file):
     
    img = cv2.imread(file, 0)
    
    counter = []
    value_unit_scale = []

    ###########################################################################
    # number and unit obtaining 

    thresh = cv2.bitwise_not(cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)[1])

    result = img.copy()
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #print(contours)
    contours = contours[0] if len(contours) == 2 else contours[1]
    big_contour = max(contours, key=cv2.contourArea)
    # big_contour struc: [a][0][b], where a is the corner idx (runs from upper left + ↓ and upper right + ↓), b is either y [0] or x [1] coord
    # initial flattened big contour list y1 x1 y2 x2 etc.

    y_coords = big_contour.flatten()[::2]
    x_coords = big_contour.flatten()[1::2]

    # for drawing purposes 
    if len(big_contour) > 4: 
        big_contour = np.array([ [[min(y_coords), max(x_coords)]], [[min(y_coords), min(x_coords)]], [[max(y_coords), min(x_coords)]], [[max(y_coords), max(x_coords)]] ])

    cv2.drawContours(result, [big_contour], 0, (255,255,255), 1)

    # 1px height cross-section
    #cv2.imshow("res", result)
    cross = img[int( (min(x_coords)+ max(x_coords)) //2 ): int( (min(x_coords)+ max(x_coords)) //2 )+1, min(y_coords):max(y_coords)+5]

    #cv2.imshow("cross", cross)
    ar_z = np.insert(np.nonzero(cross), 0, 0)
    # diff between non-zero values - fetching the zero amount of zeros and fetching the 
    # start idx of the longest sub-arr of consecutive zeros
    min_val = ar_z[np.argmax(np.ediff1d(ar_z))]
    # end idx of the zero arr
    max_val = ar_z[np.argmax(np.ediff1d(ar_z)) +1]

    cutting_idx = int(np.mean((min_val, max_val)))
    #print(cutting_idx)


    #print(cross)
    # thinner vs thicker font in scale 
    # if np.count_nonzero(cross) < 20: 
    #     scalar = 20
    # else:
    #     scalar = 30

    # extracting unit
    unit = img[min(x_coords):max(x_coords), min(y_coords) + cutting_idx:max(y_coords)]
    unit = np.pad(unit, pad_width = [(1, 1),(1, 1)], mode = "constant")

    # extracting number
    number = img[min(x_coords):max(x_coords), min(y_coords): min(y_coords) + cutting_idx]
    number = np.pad(number, pad_width = [(1, 1),(1, 1)], mode = "constant")
    # print(np.count_nonzero(number)
    # cv2.imshow("number", number)
    # cv2.imshow("unit", unit) 

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # segmentation technique - one line with many possible char-s
    # https://github.com/madmaze/pytesseract
    # https://muthu.co/all-tesseract-ocr-options/
    custom_config= r'--psm 10' # can provide specific letters/numbers --tessedit_char_whitelist="unm"'

    try:
        for val in range(10,30, 1):
            detected_nr = str(pytesseract.image_to_string(cv2.resize(cv2.threshold(number, 250, 255, cv2.THRESH_BINARY)[1], (val, val)), config =custom_config, timeout = 2)).split("y\n\x0c")[0].strip() # Timeout after 2 seconds
            # give nr options 
            if detected_nr in pot_values:
                value_unit_scale.append(int(detected_nr))
                counter.append(1)

            detected_unit = str(pytesseract.image_to_string(cv2.resize(cv2.threshold(unit, 250, 255, cv2.THRESH_BINARY)[1], (val, val)), config =custom_config, timeout = 2)).split("y\n\x0c")[0].strip() # Timeout after 2 seconds                
            if detected_unit in pot_units:
                value_unit_scale.append(detected_unit)
                counter.append(2)

            if len(np.unique(counter)) == 2:
                value_unit_scale = list(np.unique(value_unit_scale))
                break


    except RuntimeError as timeout_error:
    # Tesseract processing is terminated
        pass
    ################################################################

    # contouring the scale 
    th2 = cv2.threshold(img[max(x_coords):, min(y_coords)-5: max(y_coords) + 200], 254, 255, cv2.THRESH_BINARY)[1]

    ## line length in pixels
    try:
        scale_length = np.max(np.nonzero(th2)[1]) - np.min(np.nonzero(th2)[1])
        value_unit_scale.append(scale_length)
        print(value_unit_scale)
        return value_unit_scale
    
    except: 
        return None


# cv2.waitKey(0)
# cv2.destroyAllWindows()


# if __name__ == "__main__":
#     scale_obtain(PATH_1)
