import cv2
import os 
import numpy as np 
from PIL import Image

# import warnings
# warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras

# model 
#MODELCHECKPOINT 

# some test file as inp

def net_prediction(file_path:str):
#test = os.path.join("EcN_II_PEO_131120_GML_15k_03.jpg") 
    model = keras.models.load_model("/home/marilin/Documents/ESP/data/unet_test/unet3_model.h5")
  
    test = np.array(Image.open(file_path)).astype(float)
    
    #print("Test im size: " + str(test.shape))

    M,N = 256,256
    # slicing test image into 256x256 tiles - 256 was input size for modMel 
    #https://stackoverflow.com/questions/5953373/how-to-split-image-into-multiple-pieces-in-python
    tiles = np.array([test[x:x+M,y:y+N] for x in range(0,test.shape[0],M) for y in range(0,test.shape[1],N)])


    ## testing for 2k and 5k images - rescaling as the model was trained on 15k image, so initially we would like to have ##
    # for 2k: 7.5x less info and then resize to 256x256 (256/8), for 5k it is around 3x less info (256/4)
    # M,N = int(256/8), int(256/8)
    # tiles = np.array([cv2.resize(test[x:x+M,y:y+N], (256,256)) for x in range(0,test.shape[0],M) for y in range(0,test.shape[1],N)])

    # # prediction
    # preds_train = np.squeeze(model.predict(tiles, verbose=1, batch_size=4)) # Batch size here is important because otherwise you will run out of memory
    
    # # # testing for 2k and 5k images
    # preds_train = np.array([cv2.resize(slice, (M,N)) for slice in preds_train])
    
    ###

    preds_train = model.predict(tiles, verbose=1, batch_size=4)

    #thresholded_predictions = np.linspace(0.5, 1, 12)

    sec = int(test.shape[1] / M) # nr of sections created in the width part

    # # have chosen 0.6 as th based on visual comparison from /home/marilin/Documents/ESP/data/fiber_tests/segmented_img_class/unet_colab/masked_w_edge
    th = 0.6
    #for th in thresholded_predictions: 
    # probabilities
    preds_train_t = (preds_train > th).astype(np.uint8)

    # sewing the slices back together
    layers = [np.concatenate(preds_train_t[i:i+sec], axis = 1) for i in range(0, len(preds_train_t), sec)]
    new_im = np.concatenate(layers, axis =0) * 255

    # median blur w 15 
    blurred_im = cv2.medianBlur(np.uint8(new_im), 15)

    return np.uint8(blurred_im)
    
    #     # saving each thresholded image to a specified PATH - probably needs some reviewing to choose the best threshold 
    #     #cv2.imwrite(f" /home/marilin/Documents/ESP/data/unet_test/th_images/unet3_image_{round(th,1)}.png", np.uint8(new_im))

## testing arena
# cv2.imshow("net", net_prediction("/home/marilin/Documents/ESP/data/fiber_tests/fiber_test_1/2k_5k_orig_images/EcN_II_PEO_131120_GML_5k_02.tif"))
# #cv2.imshow("net", net_prediction("/home/marilin/Documents/ESP/data/fiber_tests/fiber_test_1/original_img/EcN_II_PEO_131120_GML_15k_03.tif"))
cv2.imwrite("/home/marilin/Documents/ESP/data/fiber_tests/fiber_test_1/segmented_img_class_tmp/EcN_PEO_111220_2k06_net_imp_new.png", net_prediction("/home/marilin/Documents/ESP/data/fiber_tests/fiber_test_1/2k_5k_orig_images/EcN_PEO_111220_2k06.tif"))
# cv2.waitKey(0)
# cv2.destroyAllWindows()