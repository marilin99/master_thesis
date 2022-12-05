import numpy as np
import cv2 
import os 
import difflib
import matplotlib.pyplot as plt
import pytesseract

#PATH_1 = "/home/marilin/Documents/ESP/data/SEM/EcN_II_PEO_131120_GML_15k_01.tif"
PATH_1 = "/home/marilin/Documents/ESP/data/SEM/Lactis_PEO_111220_20k04.tif"
#pytesseract.pytesseract.tesseract_cmd =r"C:/Program Files/Tesseract-OCR/tesseract.exe"
d  ={}
trackbar_value = 1

def updateValue(new_value):
    # Make sure to write the new value into the global variable
    global trackbar_value
    trackbar_value = new_value

def main():
    #Working with image files stored in the same folder as .py file
    file = PATH_1
    #Load the image from the given location
    img = cv2.imread(file, 0)

    #cv2.namedWindow("Threshold")
    #cv2.createTrackbar("Thresholder", "Threshold", trackbar_value, 100, updateValue)



    while True:

        #Thresholding the image (Refer to opencv.org for more details)
        #ret, thresh = cv2.threshold(img, trackbar_value, 255, cv2.THRESH_BINARY)
        #blur = cv2.GaussianBlur(img, (3,3), 0)
        #ret2,th2 = cv2.threshold(blur,0,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,9,2)
        #th4 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,9,2)
        median = cv2.medianBlur(th3, 3)
        #meidan = cv2.medianBlur(th4, trackbar_value)
        dist = cv2.distanceTransform(median, cv2.DIST_L2, 5)
        # Normalize the distance image for range = {0.0, 1.0}
        # so we can visualize and threshold it
        #print(dist.size)
        #cv2.normalize(dist, dist, 0, 1, cv2.NORM_MINMAX)

        # number 1
        #corner = img[-100:-60, :25]

        # number 2
        corner = img[-80:-55, 10:30]

        # 3-numbers
        #corner = img[-100:-60, :45]

        # text um
        #text = img[-100:-60, 25:75]
        text = img[-80:-55, 30:80]
        # nm
        #text = img[-100:-60, 50:90]

        sq = dist[100:200, 100:200]

        # find max from each row 
        maxes = np.amax(sq, axis = 1)
        arg_max = np.argmax(sq, axis = 1)
        #print(np.histogram(sq))

        #print(arg_max)
        #print(maxes)
        #print(sq.reshape(10,10))
        cv2.normalize(dist, dist, 0, 1, cv2.NORM_MINMAX)
        #cv2.imshow('Distance Transform Image', cv2.resize(sq, (768, 432)))
        #cv2.imshow("co", text)
        #cv2.imshow("thres_co", cv2.resize(cv2.threshold(corner,250, 255, cv2.THRESH_BINARY)[1], (500, 500)))
        #cv2.imshow("thres_co", cv2.cvtColor(corner, cv2.COLOR_GRAY2RGB))
        #custom_config= r'--psm 10'
        custom_config= r'--psm 10'

        try:
            for val in range(25, 50, 1):
                d[f"{val}"] = pytesseract.image_to_string(cv2.resize(cv2.threshold(text,250, 255, cv2.THRESH_BINARY)[1], (val, val)), config =custom_config, timeout = 0.5) # Timeout after 2 seconds

            #print(pytesseract.image_to_string(cv2.resize(cv2.threshold(text,250, 255, cv2.THRESH_BINARY)[1], (500,500)), config =custom_config, timeout = 2))

        except RuntimeError as timeout_error:
        # Tesseract processing is terminated
            pass

        print(d)
        #cv2.imshow('Original', img)
        #cv2.imshow('Threshold_otsu', th2)
        #cv2.imshow('Threshold_ada_mean_cn', th3)
        #cv2.imshow('Threshold_ada_mean', th4)
        #cv2.imshow("Threshold_ada_mean_median", median)
        #cv2.imshow("Threshold_ada_mean_2", meidan)
        #cv2.imwrite("testing.jpg", median)
   
        
        # Quit the program when 'q' is pressed
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            cv2.destroyAllWindows()
            break
    


if __name__ == "__main__":
    main()
