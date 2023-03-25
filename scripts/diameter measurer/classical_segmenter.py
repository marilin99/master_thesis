import numpy as np
import cv2 
import os 
import matplotlib.pyplot as plt
import math
from pykuwahara import kuwahara

# PATH = "/home/marilin/Documents/ESP/data/fiber_tests/original_img/"
# TARGET_PATH =  "/home/marilin/Documents/ESP/data/fiber_tests/segmented_img_class/"
# FILES = os.listdir(PATH) 


###### huang func in py 

# https://github.com/dnhkng/Huang-Thresholding/blob/master/analysis.py

def huang(data):
    """Implements Huang's fuzzy thresholding method 
        Uses Shannon's entropy function (one can also use Yager's entropy function) 
        Huang L.-K. and Wang M.-J.J. (1995) "Image Thresholding by Minimizing  
        the Measures of Fuzziness" Pattern Recognition, 28(1): 41-51"""
    threshold = -1
    first_bin = 0
    for ih in range(254):
        if data[ih] != 0:
            first_bin = ih
            break
    last_bin = 254
    for ih in range(254, -1, -1):
        if data[ih] != 0:
            last_bin = ih
            break
    term = 1.0 / (last_bin - first_bin)
    mu_0 = np.zeros(shape=(254, 1))
    num_pix = 0.0
    sum_pix = 0.0
    for ih in range(first_bin, 254):
        sum_pix += (ih * data[ih])
        num_pix += data[ih]
        mu_0[ih] = sum_pix / num_pix  # NUM_PIX cannot be zero !
    mu_1 = np.zeros(shape=(254, 1))
    num_pix = 0.0
    sum_pix = 0.0
    for ih in range(last_bin, 1, -1):
        sum_pix += (ih * data[ih])
        num_pix += data[ih]
        mu_1[ih - 1] = sum_pix / num_pix  # NUM_PIX cannot be zero !
    min_ent = float("inf")
    for it in range(254):
        ent = 0.0
        for ih in range(it):
            # Equation (4) in Reference
            mu_x = 1.0 / (1.0 + term * math.fabs(ih - mu_0[it]))
            if not ((mu_x < 1e-06) or (mu_x > 0.999999)):
                # Equation (6) & (8) in Reference
                ent += data[ih] * (-mu_x * math.log(mu_x) - (1.0 - mu_x) * math.log(1.0 - mu_x))
        for ih in range(it + 1, 254):
            # Equation (4) in Ref. 1 */
            mu_x = 1.0 / (1.0 + term * math.fabs(ih - mu_1[it]))
            if not ((mu_x < 1e-06) or (mu_x > 0.999999)):
                # Equation (6) & (8) in Reference
                ent += data[ih] * (-mu_x * math.log(mu_x) - (1.0 - mu_x) * math.log(1.0 - mu_x))
        if ent < min_ent:
            min_ent = ent
            threshold = it
 
    return threshold

### NUMPY part #####

def classical_segment(file):

    img_data = cv2.imread(file, 0)

    # kuwahara filter 
    kuwahara_img = kuwahara(img_data, method='mean', radius=2)

    histogram, bin_edges = np.histogram(kuwahara_img, bins=range(256))
    huang_th = huang(histogram)
    np_thuang = np.where(img_data > huang_th, 1, 0)

    threshold = cv2.erode(np.uint8(np_thuang),None, iterations=3)
    merged = cv2.dilate(threshold,None, iterations=3)

    merged = np.uint8(cv2.medianBlur(np.uint8(np_thuang), 15)) 
    merged[merged>0] = 255

    # saving tmp object map to folder 
    #cv2.imwrite(f"{TARGET_PATH}{f}.png", np.uint8(merged))

    # return segmented im 
    return np.uint8(merged)
