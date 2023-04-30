import numpy as np
import cv2

TARGET_PATH = "/home/marilin/Documents/ESP/data/bacteria_tests/synthesised_images/"

width, height = 512,512

for val in range(8):
    image = np.zeros((height, width, 3), np.uint8)


    num_ellipses = np.random.choice(100,1)
    for i in range(*num_ellipses):
        center_x, center_y = np.random.randint(0, width), np.random.randint(0, height)
        axes_x, axes_y = np.random.randint(3,10), np.random.randint(3,10)
        angle = np.random.randint(0, 180)
        start_angle, end_angle = 0, 360
        color = (0, 255, 0)
        thickness = -1
        cv2.ellipse(image, (center_x, center_y), (axes_x, axes_y), angle, start_angle, end_angle, color, thickness)


    num_circles = np.random.choice(100,1)
    for i in range(*num_circles):
        center_x, center_y = np.random.randint(0, width), np.random.randint(0, height)
        radius = np.random.randint(3,10)
        color = (0, 255, 0)
        thickness = -1
        cv2.circle(image, (center_x, center_y), radius, color, thickness)

    cv2.imwrite(f"{TARGET_PATH}syn_{num_ellipses+num_circles}_green.png", image)
# cv2.imshow("Red Ellipses and Circles", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
