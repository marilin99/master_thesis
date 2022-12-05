import numpy as np
import cv2 
import os 
import difflib

PATH_1 = "/home/marilin/Documents/ESP/data/15june2022/spin1_DAPImounting_syto9_stack2/"
PATH_2 = "/home/marilin/Documents/ESP/data/15june2022/spin1_DAPImounting_syto9_stack2/objects/"

# file = cv2.imread("data/15june2022/spin1_DAPImounting_syto9_stack2/spin1_DAPImounting_syto9_stack2_z01c1-4.tif", cv2.COLOR_BGR2RGB)
# file_2 = cv2.imread("data/15june2022/spin1_DAPImounting_syto9_stack2/objects/spin1_DAPImounting_syto9_stack2_z01c1-4_Probabilities_0.tif", 0)
# _ , thresholded = cv2.threshold(file_2, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# pic_2 = cv2.bitwise_and(file, file, mask = thresholded)

# while True:
#     cv2.imshow("og", file)
#     cv2.imshow('img', file_2)
#     cv2.imshow("da", thresholded)
#     cv2.imshow("merge", pic_2)


#     # Quit the program when 'q' is pressed
#     if (cv2.waitKey(1) & 0xFF) == ord('q'):
#         cv2.destroyAllWindows()
#         break


for file in os.listdir(PATH_1):
    if ".tif" in file:
        name = file.split(".tif")[0]
        for pic in os.listdir(PATH_2):
            if name in pic:
                print(PATH_1+file)
                file = cv2.imread(PATH_1+file, cv2.COLOR_BGR2RGB)
                pic = cv2.imread(PATH_2+pic, 0)
                # Otsu's thresholding
                _ , thresholded = cv2.threshold(pic, 0, 255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                pic_2 = cv2.bitwise_and(file, file, mask = thresholded)
                #print(pic_2)
                cv2.imwrite(f"data/15june2022/spin1_DAPImounting_syto9_stack2/masked/{name}_masked.png", pic_2)
