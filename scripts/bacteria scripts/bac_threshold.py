import cv2
from varname import argname


def works_for_both(data):
    _, thresh = cv2.threshold(data, 200, 255,cv2.THRESH_TOZERO)
    # https://github.com/pwwang/python-varname
    # change condition if needed - type based on variable name 
    if 'green' in argname("data"):
        ty = "green"

    elif 'red' in argname("data"):
        ty = "red"

    return thresh, ty


# if __name__ == "__main__":
#      im = cv2.imread("/home/marilin/Documents/ESP/data/SYTO_PI_conversion/sack_fibers_NOgrowth_syto_PI_2_red_6.png",0)
#      cv2.imshow("thresh", works_for_both(im)[0])
#      cv2.waitKey(0)
#      cv2.destroyAllWindows()