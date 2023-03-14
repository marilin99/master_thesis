import cv2
import os 
import numpy as np 
from PIL import Image
import joblib

# model 
model = joblib.load("/home/marilin/Documents/ESP/data/unet_test/unet3_adam_marilin_cnn.joblib")

# some test file as inp
test = os.path.join("EcN_II_PEO_131120_GML_15k_03.jpg") 
test = np.array(Image.open(test)).astype(float)
print("Test im size: " + str(test.shape))

M,N = 256,256
# slicing test image into 256x256 tiles - 256 was input size for model 
#https://stackoverflow.com/questions/5953373/how-to-split-image-into-multiple-pieces-in-python
tiles = np.array([test[x:x+M,y:y+N] for x in range(0,test.shape[0],M) for y in range(0,test.shape[1],N)])

# prediction
preds_train = model.predict(tiles, verbose=1, batch_size=4) # Batch size here is important because otherwise you will run out of memory

thresholded_predictions = np.linspace(0.5, 1, 12)

sec = test.shape[1] / M # nr of sections created in the width parth

for th in thresholded_predictions: 
  # probabilities
  preds_train_t = (preds_train > th).astype(np.uint8)

  # sewing the slices back together
  layers = [np.concatenate(preds_train_t[i:i+sec], axis = 1) for i in range(0, len(preds_train_t), sec)]
  new_im = np.concatenate(layers, axis =0) * 255
  
  # saving each thresholded image to a specified PATH - probably needs some reviewing to choose the best threshold 
  cv2.imwrite(f"th_images/unet3_image_{round(th,1)}.png", np.uint8(new_im))


