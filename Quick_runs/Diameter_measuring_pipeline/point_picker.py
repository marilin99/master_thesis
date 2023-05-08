# CODE for obtaining the starting locations from the image randomly #

import numpy as np

def point_picker(PATH_1, amount_of_points):
     """
     Picking points from the segmented image, when a specific amount of points is found, the search stops
     """
     coords = []

     while True:
          rnd_idx = np.random.randint(0, (len(np.where(PATH_1 > 0)[0])), 1)
          x, y = np.where(PATH_1 > 0)[0][rnd_idx][0], np.where(PATH_1 > 0)[1][rnd_idx][0]

          #relative to px pos
          try:
               # the location needs to be at least one pixel away from all of the borders
               U = PATH_1[x-1][y]
               R = PATH_1[x][y+1]
               L = PATH_1[x][y-1]
               B = PATH_1[x+1][y]
          #if too much in the border - find a better location 
          except:
               continue

          # safe check - exclusive or 
          if (U == 0 and L == 0) ^ (U == 0 and R == 0) ^ (B == 0 and L == 0) ^ (B == 0 and R == 0):
          # uniqueness (was one pair at 1000 points)
               if amount_of_points < 1000:
                    coords.append((x,y))
               else:
                    if (x,y) not in coords:
                         coords.append((x,y))

          # amount of points chosen
          if len(coords) == amount_of_points:
               break
          

     return coords