import numpy as np
import cv2 
import os 
import difflib
import matplotlib.pyplot as plt
import pytesseract

PATH_1 = "/home/marilin/Documents/ESP/data/SEM/EcN_II_PEO_131120_GML_15k_01.tif"
#PATH_1 = "/home/marilin/Documents/ESP/data/SEM/Lactis_PEO_111220_20k04.tif"
#pytesseract.pytesseract.tesseract_cmd =r"C:/Program Files/Tesseract-OCR/tesseract.exe"
d  ={}
trackbar_value = 1

def updateValue(new_value):
    # Make sure to write the new value into the global variable
    global trackbar_value
    trackbar_value = new_value*2-1

def main():
    #Working with image files stored in the same folder as .py file
    file = PATH_1
    #Load the image from the given location
    # removing part with scale
    img = cv2.imread(file, 0)[:650, :]

    cv2.namedWindow("Threshold")
    cv2.createTrackbar("Thresholder", "Threshold", trackbar_value, 100, updateValue)



    while True:

        #Thresholding the image (Refer to opencv.org for more details)
        #ret, thresh = cv2.threshold(img, trackbar_value, 255, cv2.THRESH_BINARY)
        #blur = cv2.GaussianBlur(img, (3,3), 0)
        #ret2,th2 = cv2.threshold(blur,0,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,9,2)
        #th4 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,9,2)
        median = cv2.medianBlur(th3, 3)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dil = cv2.dilate(median, kernel)
        edges = cv2.Canny(dil,0,255)
        
        # do connected components processing
        nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(edges, None, None, None, 8, cv2.CV_32S)

        #get CC_STAT_AREA component as stats[label, COLUMN] 
        areas = stats[1:,cv2.CC_STAT_AREA]
        result = np.zeros((labels.shape), np.uint8)

        for i in range(0, nlabels - 1):
            if areas[i] >= 10:   #keep
                result[labels == i + 1] = 255

        #meidan = cv2.medianBlur(edges, 3)
        #ero = cv2.erode(edges, kernel)

        
        #dil = cv2.dilate(edges, kernel)

        #meidan = cv2.medianBlur(th4, trackbar_value)
        dist = cv2.distanceTransform(cv2.bitwise_not(result), cv2.DIST_L2, 5)
        # Normalize the distance image for range = {0.0, 1.0}
        # so we can visualize and threshold it
        #print(np.histogram(dist, np.arange(np.amin(dist), np.amax(dist))))

        # searching for whiter areas by distance transform value
        sq = dist[100:200, 100:200]
        # tst = sq.copy()
        # for row in range(100):
        #     for col in range(100):
        #         if sq[row, col] > 5:
        #             tst[row, col] = 0

       
        # find max from each row 
        maxes = np.amax(sq, axis = 1)
        #print(maxes)
        arg_max = np.argmax(sq, axis = 1)
        #print(np.histogram(sq))

        #print(arg_max)
        #print(maxes)
        #print(sq.reshape(10,10))
        cv2.normalize(dist, dist, 0, 1, cv2.NORM_MINMAX)
        #cv2.imshow('Distance Transform Image sq', cv2.resize(sq, (img.shape[1], img.shape[0])))
        #cv2.imshow('Distance Transform Image', cv2.resize(tst, (img.shape[1], img.shape[0])))

  


        cv2.imshow('Original', img)
        #cv2.imshow('Threshold_otsu', th2)
        #cv2.imshow('Threshold_ada_mean', th3)
        #cv2.imshow('Threshold_ada_gaus', th4)
        #cv2.imshow("Threshold_ada_mean_median", median)
        #cv2.imshow("dilate", dil)
        cv2.imshow("edge", edges)
        #cv2.imshow("d", dist)
        #cv2.imshow("Threshold_ada_mean_2", cv2.bitwise_not(result))
        #cv2.imwrite("testing.jpg", median)
   
        
        # Quit the program when 'q' is pressed
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            cv2.destroyAllWindows()
            break
    


if __name__ == "__main__":
    main()
