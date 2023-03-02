import numpy as np
import sklearn
from sklearn.feature_extraction import image
import skimage
from skimage import io
import cv2
from sklearn.model_selection import train_test_split

one_image = cv2.imread("/home/marilin/Documents/ESP/data/SEM/EcN_II_PEO_131120_GML_15k_01.tif")
# ground
patches = image.extract_patches_2d(one_image, patch_size = (128,128), max_patches = 48, random_state= 2023)
# label
fg_patches =  image.extract_patches_2d(cv2.imread("/home/marilin/Documents/ESP/data/dataset/foreground.png"), (128,128), max_patches = 48, random_state=2023)

np.random.seed(0)
# max = 48, selected 28
train_idxs = np.random.choice(48, 28, replace = False)
# boolean mask 
arr = np.arange(len(patches))
test_idxs = np.delete(arr, train_idxs)

# assumes the folder path exists
for i, img in enumerate(patches[train_idxs]):
    cv2.imwrite("/home/marilin/Documents/ESP/data/unet_test/train_2/images/{}.png".format(i), img.astype(np.uint8))

for i, img in enumerate(patches[test_idxs]):
    cv2.imwrite("/home/marilin/Documents/ESP/data/unet_test/validation_2/images/{}.png".format(i), img.astype(np.uint8))

for i, img in enumerate(fg_patches[train_idxs]):
    cv2.imwrite("/home/marilin/Documents/ESP/data/unet_test/train_2/masks/{}.png".format(i), img.astype(np.uint8))

for i, img in enumerate(fg_patches[test_idxs]):
    cv2.imwrite("/home/marilin/Documents/ESP/data/unet_test/validation_2/masks/{}.png".format(i), img.astype(np.uint8))
