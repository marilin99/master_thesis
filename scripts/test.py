import numpy as np
import cv2 
import os 
import difflib
import matplotlib.pyplot as plt

PATH_1 = "/home/marilin/Documents/ESP/data/SEM/EcN_II_PEO_131120_GML_15k_03.tif"

l_trackbar_value = 1
u_trackbar_value = 1

def updateValue1(new_value):
    # Make sure to write the new value into the global variable
    global l_trackbar_value
    l_trackbar_value = new_value*2-1

def updateValue2(new_value):
    # Make sure to write the new value into the global variable
    global u_trackbar_value
    u_trackbar_value = new_value*2-1

def main():

    #Working with image files stored in the same folder as .py file
    file = PATH_1
    #Load the image from the given location
    # removing part with scale
    image = cv2.imread(file, 0) #[:650, :]

    cv2.namedWindow("Threshold")
    cv2.createTrackbar("l_trackbar_value", "Threshold", l_trackbar_value, 255, updateValue1)
    cv2.createTrackbar("u_trackbar_value", "Threshold", u_trackbar_value, 255, updateValue2)

    while True:
        #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(image, (13,13), 0)
        
        #gray = cv2.bilateralFilter(image, 5, 12.0, 16.0)
        #gray = cv2.medianBlur(image, 3)
        # perform edge detection, then perform a dilation + erosion to
        # close gaps in between object edges
        #edged = cv2.Canny(gray, l_trackbar_value, u_trackbar_value)
        edged = cv2.Canny(gray, 32, 0)
        edged = cv2.dilate(edged, None, iterations=1)
        edged = cv2.erode(edged, None, iterations=1)

        cv2.imshow('Original', image)
        cv2.imshow("edge", edged)
        cv2.imwrite("/home/marilin/Documents/ESP/data/unet_test/edged_15_03_ver2.png", edged)


            
        # Quit the program when 'q' is pressed
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            cv2.destroyAllWindows()
            break
     
if __name__ == "__main__":
    main()
