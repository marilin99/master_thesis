### testing segmentation methods for bacteria images and imitating fiber segmentation in imagej

import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage import segmentation, color, io
from skimage.future import graph
from skimage.segmentation import felzenszwalb, mark_boundaries
from skimage.util import img_as_float

#img = '/home/marilin/Documents/ESP/data/SYTO_PI/control_50killed_syto_PI_2-Image Export-02_c1-3.jpg'
img = "/home/marilin/Documents/ESP/data/SEM/EcN_II_PEO_131120_GML_15k_01.tif"

sample_image = cv2.imread(img)
img = cv2.cvtColor(sample_image,cv2.COLOR_BGR2HSV)

# ### k means segmentation 
# # 3d to 2d
# twoDimage = img.reshape((-1,3))
# twoDimage = np.float32(twoDimage)

# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# K = 4
# attempts=10

# ret,label,center=cv2.kmeans(twoDimage,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
# center = np.uint8(center)
# res = center[label.flatten()]
# result_image = res.reshape((img.shape))


# ### contour detection 
# #img=cv2.resize(img,(256,256))
# gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
# _,thresh = cv2.threshold(gray, np.mean(gray), 255, cv2.THRESH_BINARY_INV)
# edges = cv2.dilate(cv2.Canny(thresh,0,255),None)

# cnt = sorted(cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2], key=cv2.contourArea)[-1]
# mask = np.zeros((512,512), np.uint8)
# masked=cv2.drawContours(mask,[cnt],-1,255,-1)
# dst = cv2.bitwise_and(img, img, mask=mask)
# segmented = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)

# ####color masking 
# low_green = np.array([36,25,26])
# high_green = np.array([70,255,249])

# low_red =  np.array([0,25,26])
# high_red = np.array([31,255,249])

# thresholded = cv2.inRange(img, low_green, high_green)

# result = cv2.bitwise_and(sample_image, sample_image, mask = mask)

# ### felzenszwalb segmentation
# img = img_as_float(img[::2, ::2])

# segments_fz = felzenszwalb(img, scale=100, sigma=0.5, min_size=10)

#print(f'Felzenszwalb number of segments: {len(np.unique(segments_fz))}')

### rag skimage
# https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_rag_merge.html#sphx-glr-auto-examples-segmentation-plot-rag-merge-py

def _weight_mean_color(graph, src, dst, n):
    """Callback to handle merging nodes by recomputing mean color.

    The method expects that the mean color of `dst` is already computed.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    n : int
        A neighbor of `src` or `dst` or both.

    Returns
    -------
    data : dict
        A dictionary with the `"weight"` attribute set as the absolute
        difference of the mean color between node `dst` and `n`.
    """

    diff = graph.nodes[dst]['mean color'] - graph.nodes[n]['mean color']
    diff = np.linalg.norm(diff)
    return {'weight': diff}


def merge_mean_color(graph, src, dst):
    """Callback called before merging two nodes of a mean color distance graph.

    This method computes the mean color of `dst`.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    """
    graph.nodes[dst]['total color'] += graph.nodes[src]['total color']
    graph.nodes[dst]['pixel count'] += graph.nodes[src]['pixel count']
    graph.nodes[dst]['mean color'] = (graph.nodes[dst]['total color'] /
                                      graph.nodes[dst]['pixel count'])

# nr of area based
labels = segmentation.slic(sample_image, compactness=30,n_segments=100,start_label=1)
g = graph.rag_mean_color(sample_image, labels)
labels2 = graph.merge_hierarchical(labels, g, thresh=40, rag_copy=False,
                                   in_place_merge=True,
                                   merge_func=merge_mean_color,
                                   weight_func=_weight_mean_color)

out = color.label2rgb(labels2, sample_image, kind='avg', bg_label=0)
result = segmentation.mark_boundaries(out, labels2, (0, 0, 0)) 

# gray scale value based?



cv2.imshow('result', np.uint8(result))
cv2.imshow('original', sample_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
