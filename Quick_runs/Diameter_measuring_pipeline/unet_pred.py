# Code for using a pre-trained U-Net model for image segmentation #

# libraries
import cv2
import os 
import numpy as np 
from PIL import Image
import tensorflow as tf
from tensorflow import keras


def net_prediction(file_path:str):
    #UNET file path to be INSERTED HERE #
    MODEL_PATH = "/home/marilin/fibar_tool/Quick_runs/Diameter_measuring_pipeline/unet3_model.h5"
    model = keras.models.load_model(MODEL_PATH)
  
    test = np.array(Image.open(file_path)).astype(float)

    M,N = 256,256
    # slicing test image into 256x256 tiles - 256 was input size for modMel 
    #https://stackoverflow.com/questions/5953373/how-to-split-image-into-multiple-pieces-in-python
    tiles = np.array([test[x:x+M,y:y+N] for x in range(0,test.shape[0],M) for y in range(0,test.shape[1],N)])

    preds_train = model.predict(tiles, verbose=1, batch_size=4)

    #thresholded_predictions = np.linspace(0.5, 1, 12)

    sec = int(test.shape[1] / M) # nr of sections created in the width part

    
    th = 0.6

    preds_train_t = (preds_train > th).astype(np.uint8)

    # sewing the slices back together
    layers = [np.concatenate(preds_train_t[i:i+sec], axis = 1) for i in range(0, len(preds_train_t), sec)]
    new_im = np.concatenate(layers, axis =0) * 255

    # median blur w 15 
    blurred_im = cv2.medianBlur(np.uint8(new_im), 15)

    return np.uint8(blurred_im)