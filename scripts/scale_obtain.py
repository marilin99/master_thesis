import numpy as np
import cv2 
import os 
import difflib
import matplotlib.pyplot as plt
import pytesseract
import re

PATH_1 = "/home/marilin/Documents/ESP/data/SEM/EcN_II_PEO_131120_GML_15k_05.tif" # 400
#PATH_1 = "/home/marilin/Documents/ESP/data/SEM/Lactis_PEO_111220_20k04.tif" # 1
#PATH_1 =  "/home/marilin/Documents/ESP/data/SEM/Lactis_PEO_111220_20k03.tif" # 2 
#PATH_1 =  "/home/marilin/Documents/ESP/data/SEM/EcN_II_PEO_131120_GML_5k_02.tif" # small scale
#PATH_1 = "/home/marilin/Documents/ESP/data/SEM/PEO_EcN_I_181220_MM_2k_02.tif"
#pytesseract.pytesseract.tesseract_cmd =r"C:/Program Files/Tesseract-OCR/tesseract.exe"
lst = []
# pot_val, pot_unit, scale length in px
value_unit_scale = []
# list of units
pot_units = ["nm", "um"]
# pot scale values - should be updated if some new scales used!
pot_values = ["10", "1", "2", "400"]

trackbar_value = 1

# helper function to find suitable values
def updateValue(new_value):
    # Make sure to write the new value into the global variable
    global trackbar_value
    trackbar_value = new_value

def main():
    #Working with image files stored in the same folder as .py file
    file = PATH_1
    #Load the image from the given location
    img = cv2.imread(file, 0)

    counter = []
    value_unit_scale = []

    ###########################################################################
    # number and unit obtaining 
    thresh = cv2.bitwise_not(cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)[1])

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
    
    unit = img[min(x_coords):max(x_coords), max(y_coords)-30:max(y_coords)]
    unit = np.pad(unit, pad_width = [(1, 1),(1, 1)], mode = "constant")
    number = img[min(x_coords):max(x_coords), min(y_coords):max(y_coords)-30]
    number = np.pad(number, pad_width = [(1, 1),(1, 1)], mode = "constant")

    custom_config= r'--psm 10'

    try:
        for val in range(10, 22, 1):
            detected_nr = str(pytesseract.image_to_string(cv2.resize(cv2.threshold(number, 250, 255, cv2.THRESH_BINARY)[1], (val, val)), config =custom_config, timeout = 0.5)).split("y\n\x0c")[0].strip() # Timeout after 2 seconds
            if detected_nr in pot_values:
                value_unit_scale.append(int(detected_nr))
                counter.append(1)
            detected_unit = str(pytesseract.image_to_string(cv2.resize(cv2.threshold(unit, 250, 255, cv2.THRESH_BINARY)[1], (val, val)), config =custom_config, timeout = 0.5)).split("y\n\x0c")[0].strip() # Timeout after 2 seconds                
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

    th2 = cv2.threshold(img[max(x_coords):, min(y_coords)-5: max(y_coords) + 50], 254, 255, cv2.THRESH_BINARY)[1]
    ## line length in pixels
    scale_length = np.max(np.nonzero(th2)[1]) - np.min(np.nonzero(th2)[1])
    value_unit_scale.append(scale_length)
    print(value_unit_scale)


    # cv2.imshow('Original', img)
    #cv2.imshow("result", result)
    #cv2.imshow("value", number)

    
    # Quit the program when 'q' is pressed
    # if (cv2.waitKey(1) & 0xFF) == ord('q'):
    #     cv2.destroyAllWindows()
    #     break
    


if __name__ == "__main__":
    main()
