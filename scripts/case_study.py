import numpy as np
import cv2 
import matplotlib.pyplot as plt

img = '/home/marilin/Documents/ESP/data/SYTO_PI/control_50killed_syto_PI_2-Image Export-02_c1-3.jpg'
# trackbar_value = 1

# def updateValue(new_value):
#     # Make sure to write the new value into the global variable
#     global trackbar_value
#     trackbar_value = new_value*2-1

# def main(): 

#     #Load the image from the given location
#     # removing part with scale
#     original = cv2.imread(img)
#     image = cv2.imread(img, 0)
#     cv2.namedWindow("Threshold")
#     cv2.createTrackbar("Thresholder", "Threshold", trackbar_value, 100, updateValue)

#     while True:
#         #_, thresh = cv2.threshold(image, trackbar_value, 255, cv2.THRESH_BINARY)

#         thresh = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,9,2)

#         cv2.imshow('Original', original)
#         cv2.imshow("thresh", thresh)


            
#         # Quit the program when 'q' is pressed
#         if (cv2.waitKey(1) & 0xFF) == ord('q'):
#             cv2.destroyAllWindows()
#             break
     
# if __name__ == "__main__":
#     main()

original = cv2.imread(img)
#hsv_img = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
#hue = hsv_img[:,:,0]
#print(np.unique(hue))
#sat = hsv_img[:, :, 1]
#val = hsv_img[:, :, 2]
b,g,r = cv2.split(original)

_, thresh = cv2.threshold(g,210,255,cv2.THRESH_BINARY)
cv2.imshow('Original image', original)
cv2.imshow('b', b)
cv2.imshow('g', g)
cv2.imshow('r', r)
cv2.imshow("thresh", thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()