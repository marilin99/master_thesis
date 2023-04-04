# from importlib.metadata import metadata
# from aicsimageio import AICSImage
# from aicsimageio.readers import *
# from aicsimageio.writers import *

import cv2

# img = AICSImage("my_file.tiff")  # selects the first scene found
# img.data  # returns 5D TCZYX numpy array
# img.xarray_data  # returns 5D TCZYX xarray data array backed by numpy
# img.dims  # returns a Dimensions object
# img.dims.order  # returns string "TCZYX"
# img.dims.X  # returns size of X dimension
# img.shape  # returns tuple of dimension sizes in TCZYX order
# img.get_image_data("CZYX", T=0)  # returns 4D CZYX numpy array

# def aicsio_skills(file):
#     img = AICSImage(file)
#     print(img.data, "\n", img.dims, "\n", img.shape, "\n", img.metadata, "\n", img.scenes)

# def czi_reader(file):
#     reader = CziReader(file)
#     print(reader.data, "\n", reader.dims, "\n", reader.shape, "\n", reader.metadata)

# #aicsio_skills("/home/marilin/Documents/ESP/data/15june2022/spin1_DAPImounting_syto9_stack4.czi")
# aicsio_skills("/home/marilin/Documents/ESP/data/15june2022/spin1_DAPImounting_syto9_stack4.czi")
# #OmeTiffWriter.save("/home/marilin/Documents/ESP/data/15june2022/spin1_DAPImounting_syto9_stack4.czi", "my_file.ome.tif", dim_order="TCZYX")

# im = cv2.imread("/home/marilin/Documents/ESP/data/fiber_tests/fiber_test_1/2k_5k_orig_images/EcN_II_PEO_131120_GML_5k_02.tif",0)
# _, thresh1 = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# cv2.imshow("thresh", thresh1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

import csv

PATH = "/home/marilin/Documents/ESP/data/fiber_tests/fiber_test_2/"
column_name = ["File path", "Diameter measures (nm)", "Runtime"]
data = ["/home/marilin/Documents/ESP/data/fiber_tests/fiber_test_2/original_data_2/PCL_MSN_CAM_10k_12031905.tif", \
        "[430, 398, 565, 478, 725, 1081, 398, 350, 398, 382, 239, 239, 430, 1223, 398, 374, 558, 534, 111, 829, 398, 524, 613, 1810, 797, 819, 319, 239, 812, 558, 319, 398, 350, 1012, 319, 246, 406, 445, 613, 478, 853, 667, 374, 524, 294, 613, 239, 1196, 111, 1116, 829, 717, 778, 1129, 717, 971, 1494, 510, 853, 333, 478, 500, 717, 374, 597, 693, 877, 406, 239, 445, 333, 1116, 374, 667, 319, 858, 1617, 1012, 969, 621, 843, 478, 558, 877, 638, 478, 638, 159, 565, 239, 454, 215, 398, 613, 319, 604, 374, 534, 773]", \
        "93.18571972846985"] #the data

with open(f'{PATH}demo.csv', 'w+') as f:
    writer = csv.writer(f) #this is the writer object
    writer.writerow(column_name) # this will list out the names of the columns which are always the first entrries
    writer.writerow(data) #this is the data