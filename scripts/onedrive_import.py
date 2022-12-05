import os 
import requests
import glob
from aicsimageio import AICSImage
from aicsimageio.readers import CziReader

import napari

def aicsio_skills(file):
    img = AICSImage(file)
    print(img.data, "\n", img.dims, "\n", img.shape)

#aici = aicsio_skills("/home/marilin/Documents/ESP/data/15june2022/150622_spin2_agarose48h_syto9_stack2.czi")
#print(glob.glob("google-drive://marilon.moor@gmail.com/2021-03-01"))

def czi_reader(file):
    reader = CziReader(file)
    print(reader.data, "\n", reader.dims, "\n", reader.shape)

img = AICSImage("/home/marilin/Documents/ESP/data/15june2022/150622_spin2_agarose48h_syto9_stack2.czi")
#print(img.get_image_data("ZYX", C=1, S=0, T=1))  # returns 3D ZYX numpy array

viewer = napari.Viewer()
viewer.open("/home/marilin/Documents/ESP/data/15june2022/150622_spin2_agarose48h_syto9_stack2.czi")
