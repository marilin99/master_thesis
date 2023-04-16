import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# Set the size of the output image
image_size = 1024
TARGET_PATH = "/home/marilin/Documents/ESP/data/fiber_tests/synthesised_fibers/2_tone_fibers/"

# Create a blank black image
image = np.zeros((image_size, image_size), dtype=np.uint8)

for m in range(7):

    # Generate random diameters for the lines from a normal distribution
    mean_diameter = 20
    std_diameter = 5
    diameters_co = []
    tmp_img = np.zeros((image_size, image_size), dtype=np.uint8)

    # Loop through the lines
    for j in range(2):

        diameters = np.random.normal(loc=mean_diameter, scale=std_diameter, size=15)
        orientations = np.random.rand(15) * (np.pi)

        image = np.zeros((image_size, image_size), dtype=np.uint8)

        for i in range(len(diameters)):

            orientation = orientations[i]
            diameter = diameters[i]
            
            # Calculate the start and end points of the line
            x1 = 0
            y1, y2 = int(np.random.rand() * image_size), int(np.random.rand() * image_size)
            x2 = image_size - 1
            width = int(diameter)
            diameters_co.append(width)
            
            # Draw the line on the image
            cv2.line(image, (x1, y1), (x2, y2), (255, 255, 255), width)

    if j == 1:
        tmp_img = cv2.bitwise_or(tmp_img, image.T)

        with open(f"{TARGET_PATH}binary_{m}.txt", "w+") as file:

            file.write("***diameter values***")
            file.write("\n")
            for val in diameters_co:
                file.write(f"{val}")
                file.write("\n")



    tmp_img = cv2.bitwise_or(tmp_img, image)

    cv2.imwrite(f"{TARGET_PATH}binary_{m}.png", tmp_img[:768, :])
