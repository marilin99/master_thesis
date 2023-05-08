from plantcv import plantcv as pcv
import cv2 
import numpy as np
import skimage.morphology 

def thinner(PATH_1):
     dist = cv2.distanceTransform(PATH_1, cv2.DIST_L2, 3)
     
     thinned = skimage.morphology.medial_axis(PATH_1).astype(np.uint8)
     thinned[thinned == 1] = 255

     ## visuals for testing ##
     # cv2.imshow("thi", thinned)
     # # dist visual



     #### pre-processing of the thinned image ####

     # removing skeleton hairs - https://plantcv.readthedocs.io/en/stable/prune/
     pruned_skeleton, segmented_img, segment_objects = pcv.morphology.prune(skel_img=thinned, size=50)
     thinned = np.uint8(pruned_skeleton)

     # testing w/o pruning for 2k and 5k images 

     #cv2.imshow("pruned", np.uint8(pruned_skeleton))


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

def thinner_2k_5k(PATH_1):
     dist = cv2.distanceTransform(PATH_1, cv2.DIST_L2, 3)
     
     thinned = skimage.morphology.medial_axis(PATH_1).astype(np.uint8)
     thinned[thinned == 1] = 255

     return dist, thinned

# if __name__ == "__main__":
#      im = cv2.imread("/home/marilin/Documents/ESP/data/fiber_tests/fiber_test_1/classical_results_2/EcN_II_PEO_131120_GML_15k_01_segmented.png",0)
#      thinner(im)
#     _, thresh1 = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#    dist, thinned = thinner(im)
#     cv2.imwrite("/home/marilin/Documents/ESP/data/fiber_tests/fiber_test_1/segmented_img_class_tmp/EcN_PEO_111220_2k06_thresh.png", thresh1)
    #cv2.imwrite("/home/marilin/Documents/ESP/data/fiber_tests/fiber_test_1/segmented_img_class_tmp/EcN_PEO_111220_2k06_thinned.png", thinned)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

