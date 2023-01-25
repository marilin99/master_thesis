import numpy as np
import sklearn
from sklearn.feature_extraction import image
import skimage
from skimage import io
import cv2
from sklearn.model_selection import train_test_split

one_image = cv2.imread("/home/marilin/Documents/ESP/data/SEM/EcN_II_PEO_131120_GML_15k_01.tif")
# ground
patches = image.extract_patches_2d(one_image, patch_size = (24,32), max_patches = 1024, random_state= 2023)
# label
fg_patches =  image.extract_patches_2d(cv2.imread("/home/marilin/Documents/ESP/data/dataset/foreground.png"), patch_size = (24,32), max_patches = 1024, random_state=2023)

np.random.seed(0)
train_idxs = np.random.choice(1024, 768, replace = False)
# boolean mask 
arr = np.arange(len(patches))
test_idxs = np.delete(arr, train_idxs)

for i, img in enumerate(patches[train_idxs]):
    cv2.imwrite("/home/marilin/Documents/ESP/data/unet_test/train/images/{}.png".format(i), img.astype(np.uint8))

for i, img in enumerate(patches[test_idxs]):
    cv2.imwrite("/home/marilin/Documents/ESP/data/unet_test/validation/images/{}.png".format(i), img.astype(np.uint8))

for i, img in enumerate(fg_patches[train_idxs]):
    cv2.imwrite("/home/marilin/Documents/ESP/data/unet_test/train/masks/{}.png".format(i), img.astype(np.uint8))

for i, img in enumerate(fg_patches[test_idxs]):
    cv2.imwrite("/home/marilin/Documents/ESP/data/unet_test/validation/masks/{}.png".format(i), img.astype(np.uint8))
