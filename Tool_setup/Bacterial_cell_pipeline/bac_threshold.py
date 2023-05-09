# CODE for thresholding the microscopy image #

#libraries 
import cv2
from varname import argname


def works_for_both(data):
    """
    Function for thresholding the grayscale input image
    """
    _, thresh = cv2.threshold(data, 200, 255,cv2.THRESH_TOZERO)

    # https://github.com/pwwang/python-varname
    # change condition if needed - type based on variable name 
    if 'green' in argname("data"):
        ty = "green"

    elif 'red' in argname("data"):
        ty = "red"

    return thresh, ty