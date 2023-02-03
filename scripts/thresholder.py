#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage import *

thresholds = {"lH": 0,"lS": 37, "lV": 133, "hH": 44, "hS": 255, "hV": 249}

# Open the camera
#camera= cv2.VideoCapture(0)
img = '/home/marilin/Documents/ESP/data/SYTO_PI/control_50killed_syto_PI_2-Image Export-02_c1-3.jpg'

def updateValue_lH(newvalue): #== var1 = cv2.getTrackbarPos (after img[:] = [var1, var2 etc])

    thresholds["lH"] = newvalue
    
    
def updateValue_lS(newvalue):

    thresholds["lS"] = newvalue
    
def updateValue_lV(newvalue):

    thresholds["lV"] = newvalue


def updateValue_hH(newvalue):

    thresholds["hH"] = newvalue

def updateValue_hS(newvalue):

    thresholds["hS"] = newvalue

def updateValue_hV(newvalue):

    thresholds["hV"] = newvalue


def get_values_from_file(path_of_textfile):
    """"
    This function reads the text file and returns all of the trackbar values
    """

    with open(path_of_textfile, "r") as file:
        for line in file:
            pairs = line.strip().split(" ")
            for i in list(thresholds): #show keys, update dictionary
                i = str(pairs[0]) 
                thresholds[i] = int(pairs[1])

    return thresholds

def write_values_to_file(path_of_textfile):

    """"
    This function writes the text file and returns all of the trackbar values
    """
    with open(path_of_textfile, "w+") as file:
        for i in list(thresholds): #getting the keys of the dict as a lst
            file.write(str(i)) #key
            file.write(" ")
            file.write(str(thresholds[i])) #value
            file.write("\n")
            
    print("values written to file")

if os.path.isfile("trackbar_defaults.txt"):
    get_values_from_file("trackbar_defaults.txt")
    
def main():
        
    cv2.namedWindow("Processed")
    cv2.createTrackbar("Low Hue", "Processed", thresholds["lH"], 179, updateValue_lH)
    cv2.createTrackbar("High Hue", "Processed", thresholds["hH"], 179, updateValue_hH)
    cv2.createTrackbar("Low Saturation", "Processed", thresholds["lS"], 255, updateValue_lS)
    cv2.createTrackbar("High Saturation", "Processed", thresholds["hS"], 255, updateValue_hS)
    cv2.createTrackbar("Low Value", "Processed", thresholds["lV"], 255, updateValue_lV)
    cv2.createTrackbar("High Value", "Processed", thresholds["hV"], 255, updateValue_hV)

    
    while True:
       
        frame = cv2.imread(img)
        cv2.imshow("Original", frame) 



        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # green - 25,15,20,70,255,255
        # red - 0,25,26,31,255,249
        lowerLimits = np.array([thresholds["lH"], thresholds["lS"], thresholds["lV"]])
        upperLimits = np.array([thresholds["hH"], thresholds["hS"], thresholds["hV"]])
        thresholded_r = cv2.inRange(frame, lowerLimits, upperLimits)
        cv2.imwrite("thresholded_r.png", thresholded_r)
            
    # Our operations on the frame come here 
        
        #lowerLimits_g = np.array([36,25,26])
        #upperLimits_g = np.array([70,255,249])

        # lowerLimits_g = np.array([thresholds["lH"], thresholds["lS"], thresholds["lV"]])
        # upperLimits_g = np.array([thresholds["hH"], thresholds["hS"], thresholds["hV"]])
        
        #thresholded_g = cv2.inRange(frame, lowerLimits_g, upperLimits_g)
        #cv2.imwrite("thresholded_g.png", thresholded_g)
        # labeling output - labelled array - mat of features

        labeled_array, num_features = label(thresholded_r)
        #labeled_array_g, _ = label(thresholded_g)
        
        # sometimes erosion may help 
        # kernel = np.ones((3,3),np.uint8)
        # erosion = cv2.erode(thresholded_g, kernel, iterations = 1)
        # cv2.imshow("eroded", erosion)

        #print(num_features)

        ### one way of finding the bacteria count 

        # sizes of thresholded objects in px 
        # depends on the value from .xml about px to um
        # let's say 1px is 0.04um for now ca 25 px is one cell
        # excluding background px-s
        bac_in_px = 34
        ## red bac
        # px_counter = np.bincount(labeled_array.flatten())
        # filtered_counter_r = px_counter[np.where(px_counter > bac_in_px)][:-1]

        ## green bac
        # px_counter_g = np.bincount(labeled_array_g.flatten())
        # filtered_counter_g = px_counter_g[np.where(px_counter_g > bac_in_px)][:-1]

        # print("red", np.sum(filtered_counter_r[1:]// bac_in_px))
        # print("green", np.sum(filtered_counter_g[1:]// bac_in_px))

        #idx_from_25 = np.argsort(np.bincount(labeled_array.flatten()))[24:-1]
        #print(each_feat_size[idx_from_25])
        #print(amount_of_reds)
        #print(sorted(filtered_counter))

        #thresholded = cv2.bitwise_not(thresholded)
        frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
        outimage = cv2.bitwise_and(frame, frame, mask = thresholded_r)

        #cv2.imwrite("outimage.png", outimage)

    # contouring 
        # (cnt, _) = cv2.findContours(thresholded_r.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # rgb = cv2.cvtColor(frame, cv2.COLOR_HSV2RGB)
        #cv2.drawContours(rgb, cnt, -1, (0, 255, 0), 2)

        #cv2.imshow("rgb", rgb)
        #(cnt, hierarchy) = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        #print(len(cnt))
        
    # Display the resulting frame - thresholded image (use to check the where the ball is 
        cv2.imshow("Processed", outimage)
        #cv2.imshow("Original", frame) 
        #cv2.imshow("Thresholded", thresholded_g)
        

    # Quit the program when "q" is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
                
    # When everything done, release the capture
    print("closing program")
    cv2.destroyAllWindows()
    write_values_to_file("trackbar_defaults.txt")
    

if __name__ == "__main__":
    main()