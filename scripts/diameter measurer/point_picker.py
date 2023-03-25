import numpy as np

def point_picker(PATH_1, amount_of_points):
     # temp seed
     np.random.seed(42)
     coords = []

     while True:
          rnd_idx = np.random.randint(0, (len(np.where(PATH_1 > 0)[0])), 1)
          x, y = np.where(PATH_1 > 0)[0][rnd_idx][0], np.where(PATH_1 > 0)[1][rnd_idx][0]

          
          #choosing idx of white px - randint should be excluding high values
          

          # refacto this   (temp safety net: 34<x<h-y, 34 < y < w-y)
          # while not (n2 < x < (h-n2)) and (n2 < y < (w-n2) ):
          #      rnd_idx = np.random.randint(0, (len(np.where(PATH_1 > 0)[0])), 1)
          #      x, y = np.where(PATH_1 > 0)[0][rnd_idx][0], np.where(PATH_1 > 0)[1][rnd_idx][0]


          #neighboring px-s from white should be black (U+L, U+R / B+L, B+R) but majority of kernel should be white - no points on straight line to avoid erroneous measurements (such as end stomps etc.)

          #relative to px pos
          try:
               U = PATH_1[x-1][y]
               R = PATH_1[x][y+1]
               L = PATH_1[x][y-1]
               B = PATH_1[x+1][y]
          #if too much in the border - find a better location 
          except:
               continue

          # safe check - exclusive or to avoid obscure situations or stick to or?
          if (U == 0 and L == 0) ^ (U == 0 and R == 0) ^ (B == 0 and L == 0) ^ (B == 0 and R == 0):
          #if (U == 0 and L == 0) or (U == 0 and R == 0) or (B == 0 and L == 0) or (B == 0 and R == 0):
          # uniqueness (was one pair at 1000 points)
               #if (x,y) not in coords:
               coords.append((x,y))

          # amount of points chosen
          if len(coords) == amount_of_points:
               break
          

     return coords