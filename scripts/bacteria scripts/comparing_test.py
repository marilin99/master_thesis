import cv2 
import numpy as np 

red = cv2.imread("/home/marilin/Documents/ESP/data/bacteria_tests/test_pipeline_res/sack_fibers_NOgrowth_syto_PI_11_th_r.png", 0)
green = cv2.imread("/home/marilin/Documents/ESP/data/bacteria_tests/test_pipeline_res/sack_fibers_NOgrowth_syto_PI_11_th_g.png", 0)

print(np.nonzero(green))
print(np.nonzero(red))


print(red[324,214])

red = cv2.imread("/home/marilin/Documents/ESP/data/FM_SYTO_conversion/PCL_fibers_FM_syto_8_red_6.png",0)
trans = cv2.imread("/home/marilin/Documents/ESP/data/FM_SYTO_conversion/PCL_fibers_FM_syto_8_transmission_6.png",0)

red = cv2.blur(red, (3,3))
cv2.imshow("subtract", np.uint8(trans-red))
cv2.waitKey(0)
cv2.destroyAllWindows()
# print(np.unique(red))
# print(np.unique(green))
# green_c = tuple(zip(*np.where(green>0)))
# red_c = tuple(zip(*np.where(red>0)))

# if not (green_c == red_c):
#     print(np.unique(np.maximum(green,red)))
# else: 
#     print(np.where(green_c, green_c == red_c))

# print(green[green>0])
# print(red[red>0])
# # print(len(green[green>red]))
# # print(len(red[red>green]))
# print(np.nonzero((np.maximum(green,red))))

