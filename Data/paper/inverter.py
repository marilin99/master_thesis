import cv2 
import os 
import numpy as np 

PATH = "/home/marilin/Documents/ESP/paper/Segmented Images/"
for file in os.listdir(PATH):
	print(file)
	try:
		arr = cv2.imread(PATH+file)
		arr_cp = arr.copy()
		arr_cp[arr==0] = 255
		arr_cp[arr==255] = 0
		cv2.imwrite(f"/home/marilin/Documents/ESP/paper/inverted_images/{file}", np.uint8(arr_cp))
	except: 
		continue
