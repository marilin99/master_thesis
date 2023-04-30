import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

### unordered straight lines ###
# Set the size of the output image
# image_size = 1024
# TARGET_PATH = "/home/marilin/Documents/ESP/data/fiber_tests/synthesised_fibers/one_dm_fibers/"

# # Create a blank black image
# image = np.zeros((image_size, image_size), dtype=np.uint8)

# #for m in range(5):

# # Generate random diameters for the lines from a normal distribution
# # mean_diameter = 20
# # std_diameter = 5
# diameters_co = []
# tmp_img = np.zeros((image_size, image_size), dtype=np.uint8)
# #[5,10,15], [10,15,20], [15,20,25], [20,25,30], [25,30,35]
# n = [25,30,35]

# # Loop through the lines
# for j in range(2):

#     #diameters = np.random.normal(loc=mean_diameter, scale=std_diameter, size=15)
#     diameters = np.random.choice(n, 15)
#     orientations = np.random.rand(15) * (np.pi)

#     image = np.zeros((image_size, image_size), dtype=np.uint8)

#     for i in range(len(diameters)):

#         orientation = orientations[i]
#         diameter = diameters[i]

#         # Calculate the start and end points of the line
#         x1 = 0
#         y1, y2 = int(np.random.rand() * image_size), int(np.random.rand() * image_size)
#         x2 = image_size - 1
#         width = int(diameter)
#         diameters_co.append(width)

#         # Draw the line on the image
#         cv2.line(image, (x1, y1), (x2, y2), (255, 255, 255), width)

# if j == 1:
#     tmp_img = cv2.bitwise_or(tmp_img, image.T)

#     with open(f"{TARGET_PATH}binary_unordered_{n}.txt", "w+") as file:

#         file.write("***diameter values***")
#         file.write("\n")
#         for val in diameters_co:
#             file.write(f"{val}")
#             file.write("\n")



# tmp_img = cv2.bitwise_or(tmp_img, image)

# cv2.imwrite(f"{TARGET_PATH}binary_unordered_{n}.png", tmp_img[:768, :])

### unordered curved lines ###

# import numpy as np
# import cv2

# # Define the image size
# height, width, image_size = 1024,1024,1024

# # Define the number of curved lines to generate
# num_lines = 15
# TARGET_PATH = "/home/marilin/Documents/ESP/data/fiber_tests/synthesised_fibers/one_dm_fibers/"


# #for n in [[5,10,15], [10,15,20], [15,20,25], [20,25,30], [25,30,35]]:
# for n in [[10], [20],[30],[40],[50]]:
#     # Create a black image
#     image = np.zeros((image_size, image_size), dtype=np.uint8)
#     tmp_img = np.zeros((image_size, image_size), dtype=np.uint8)
#     diameters_co = []
#     # Define the thickness of the lines


#     # # Loop through the lines
#     for j in range(2):
#         # Generate random curved lines

#         for i in range(num_lines):
#             thickness = np.random.choice(n)

#             diameters_co.append(thickness)
#             # Define the start and end points of the line
#             edge1 = np.random.randint(4)
#             edge2 = (edge1 + np.random.randint(1, 4)) % 4
#             if edge1 == 0:
#                 start_point = (np.random.randint(width), 0)
#             elif edge1 == 1:
#                 start_point = (width-1, np.random.randint(height))
#             elif edge1 == 2:
#                 start_point = (np.random.randint(width), height-1)
#             else:
#                 start_point = (0, np.random.randint(height))
#             if edge2 == 0:
#                 end_point = (np.random.randint(width), 0)
#             elif edge2 == 1:
#                 end_point = (width-1, np.random.randint(height))
#             elif edge2 == 2:
#                 end_point = (np.random.randint(width), height-1)
#             else:
#                 end_point = (0, np.random.randint(height))

#             # Define random control points for the curve
#             control_point1 = (np.random.randint(width), np.random.randint(height))
#             control_point2 = (np.random.randint(width), np.random.randint(height))

#             # Define a random curve function
#             curve_func = np.random.choice(['quadratic', 'cubic'])

#             # Generate a series of points that define the curve
#             if curve_func == 'quadratic':
#                 num_points = 100
#                 t = np.linspace(0, 1, num_points)
#                 x = (1 - t) ** 2 * start_point[0] + 2 * (1 - t) * t * control_point1[0] + t ** 2 * end_point[0]
#                 y = (1 - t) ** 2 * start_point[1] + 2 * (1 - t) * t * control_point1[1] + t ** 2 * end_point[1]
#             elif curve_func == 'cubic':
#                 num_points = 200
#                 t = np.linspace(0, 1, num_points)
#                 x = (1 - t) ** 3 * start_point[0] + 3 * (1 - t) ** 2 * t * control_point1[0] + 3 * (1 - t) * t ** 2 * control_point2[0] + t ** 3 * end_point[0]
#                 y = (1 - t) ** 3 * start_point[1] + 3 * (1 - t) ** 2 * t * control_point1[1] + 3 * (1 - t) * t ** 2 * control_point2[1] + t ** 3 * end_point[1]
#             points = np.column_stack((x.astype(int), y.astype(int)))

#             # Draw a line between the points to create the curved line
#             for j in range(num_points - 1):
#                 cv2.line(tmp_img, tuple(points[j]), tuple(points[j+1]), color=255, thickness=thickness)


#         if j == 1:
#             tmp_img = cv2.bitwise_or(tmp_img, image.T)

#     with open(f"{TARGET_PATH}binary_unordered_curved_{n}.txt", "w+") as file:

#         file.write("***diameter values***")
#         file.write("\n")
#         for val in diameters_co:
#             file.write(f"{val}")
#             file.write("\n")



    # tmp_img = cv2.bitwise_or(tmp_img, image)

    # cv2.imwrite(f"{TARGET_PATH}binary_unordered_curved_{n}.png", tmp_img[:768, :])

## ordered lines ## 

# import numpy as np
# import cv2

# # Define the image size
# height, width, image_size = 1024,1024,1024

# # Define the number of curved lines to generate
# num_lines = 15
# TARGET_PATH = "/home/marilin/Documents/ESP/data/fiber_tests/synthesised_fibers/three_dm_fibers/"

# # Define the number of lines to generate
# num_lines = 15

# # Define the gap between the lines
# line_gap = height // (num_lines + 1)

# # Create a black image
# #img = np.zeros((height, width), dtype=np.uint8)

# for n in [[5,10,15], [10,15,20], [15,20,25]]:
#     # Create a black image
#     image = np.zeros((image_size, image_size), dtype=np.uint8)
#     tmp_img = np.zeros((image_size, image_size), dtype=np.uint8)
#     diameters_co = []
#     # Define the thickness of the lines

#     # # # Loop through the lines
#     for i in range(num_lines * 2):

#         if i % 2 == 0:  # Generate horizontal line
#             # Define the start and end points of the line
#             start_point = (0, (i//2+1) * line_gap)
#             end_point = (width-1, (i//2+1) * line_gap)
#         else:  # Generate vertical line
#             # Define the start and end points of the line
#             start_point = ((i//2+1) * line_gap, 0)
#             end_point = ((i//2+1) * line_gap, height-1)
        
#          # Define the thickness of the lines
#         thickness = np.random.choice(n)
#         diameters_co.append(thickness)


#         # Draw a line between the points
#         cv2.line(tmp_img, start_point, end_point, color=255, thickness=thickness)

#     with open(f"{TARGET_PATH}binary_ordered_{n}.txt", "w+") as file:

#         file.write("***diameter values***")
#         file.write("\n")
#         for val in diameters_co:
#             file.write(f"{val}")
#             file.write("\n")

#     #tmp_img = cv2.bitwise_or(tmp_img, image)

#     cv2.imwrite(f"{TARGET_PATH}binary_ordered_{n}.png", tmp_img[:768, :])


        # # Display the image
        # cv2.imshow("Straight Lines", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


