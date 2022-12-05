from importlib.metadata import metadata
from aicsimageio import AICSImage
from aicsimageio.readers import *
from aicsimageio.writers import *


# img = AICSImage("my_file.tiff")  # selects the first scene found
# img.data  # returns 5D TCZYX numpy array
# img.xarray_data  # returns 5D TCZYX xarray data array backed by numpy
# img.dims  # returns a Dimensions object
# img.dims.order  # returns string "TCZYX"
# img.dims.X  # returns size of X dimension
# img.shape  # returns tuple of dimension sizes in TCZYX order
# img.get_image_data("CZYX", T=0)  # returns 4D CZYX numpy array

def aicsio_skills(file):
    img = AICSImage(file)
    print(img.data, "\n", img.dims, "\n", img.shape, "\n", img.metadata, "\n", img.scenes)

def czi_reader(file):
    reader = CziReader(file)
    print(reader.data, "\n", reader.dims, "\n", reader.shape, "\n", reader.metadata)

#aicsio_skills("/home/marilin/Documents/ESP/data/15june2022/spin1_DAPImounting_syto9_stack4.czi")
aicsio_skills("/home/marilin/Documents/ESP/data/15june2022/spin1_DAPImounting_syto9_stack4.czi")
#OmeTiffWriter.save("/home/marilin/Documents/ESP/data/15june2022/spin1_DAPImounting_syto9_stack4.czi", "my_file.ome.tif", dim_order="TCZYX")