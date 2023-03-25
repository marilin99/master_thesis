from plantcv import plantcv as pcv
import cv2 
import numpy as np
import skimage.morphology 

def thinner(PATH_1):
     dist = cv2.distanceTransform(PATH_1, cv2.DIST_L2, 3)

     thinned = skimage.morphology.medial_axis(PATH_1).astype(np.uint8)
     thinned[thinned == 1] = 255

     #### pre-processing of the thinned image ####

     # removing skeleton hairs - https://plantcv.readthedocs.io/en/stable/prune/
     pruned_skeleton, segmented_img, segment_objects = pcv.morphology.prune(skel_img=thinned, size=50)


     thinned = np.uint8(pruned_skeleton)

     ### white speck removal ### erroneous
     # removing ind floating areas from skeleton 
     # object_map, count = ndimage.label(pruned_skeleton, structure = generate_binary_structure(2,2))

     # def pixelcount(regionmask):	return np.sum(regionmask)
     # props = skimage.measure.regionprops(object_map, extra_properties=(pixelcount,))

     # idxs = np.argwhere(np.array([props[val].pixelcount for val in range(len(props))]) == 1).ravel()
     # for val in idxs:	object_map[object_map == val] = 0 

     # object_map[object_map != 0] = 255

     # thinned = np.uint8(object_map)

     return dist, thinned