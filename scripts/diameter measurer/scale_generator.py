import numpy as np
import cv2

image_size = 512


# Set the length and width of the scale
TARGET_PATH = "/home/marilin/Documents/ESP/data/fiber_tests/synthesised_fibers/horizontal_bars/"
scale_width = 10

for scale_length in np.linspace(50,100, 6,dtype=np.uint8):
    
    image = np.zeros((image_size, image_size), dtype=np.uint8)

    x1 = int((image_size - scale_width) / 2)
    x2 = int((image_size + scale_width) / 2)

    y1 = int((image_size - scale_length) / 2)
    y2 = int((image_size + scale_length) / 2)

    midpoint_x = int((x1 + x2) / 2)
    midpoint_y = int((y1 + y2) / 2)

    # Draw the scale on the image
    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), -1)

    line_len = scale_length // 2

    cv2.line(image, (x1-( line_len//2 - (midpoint_x-x1) ), y1), (x2 + (line_len//2 - (midpoint_x-x1)), y1), (255, 255, 255), 2)
    cv2.line(image, (x1-(line_len//2 - (midpoint_x-x1)), y2), (x2 + (line_len//2 - (midpoint_x-x1)), y2), (255, 255, 255), 2)

    cv2.imwrite(f"{TARGET_PATH}bar_{scale_length}.png", image.T)

    # cv2.imshow("Scale", image.T)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
