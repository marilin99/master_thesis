import numpy as np

#### helper functions ####
# https://stackoverflow.com/questions/45225474/find-nearest-white-pixel-to-a-given-pixel-location-opencv
def find_nearest_white(img, origin):
     """
     more naive version using euc distance
     """

     nonzero =  np.transpose(np.nonzero(img))
     # euc dist
     distances = np.sqrt((nonzero[:,0] - origin[0]) ** 2 + (nonzero[:,1] - origin[1]) ** 2)
     nearest_index = np.argmin(distances)

     return nonzero[nearest_index]


def find_nearest_whites(img, origin):
     """
     for polynomial fitting 
     outputs idx-s of 15 nearest whites
     """
     nonzero =  np.transpose(np.nonzero(img))
     # euc dist
     distances = np.sqrt((nonzero[:,0] - origin[0]) ** 2 + (nonzero[:,1] - origin[1]) ** 2)
     # selecting 15 closest white points
     nearest_index = np.argsort(distances)[:15]
     return nonzero[nearest_index]

#####

def dm_finder(thinned:np.ndarray, dist:np.ndarray, PATH_1:np.ndarray, pt_s:list, h:int, w:int, nano_per_px:float):
     dm_s,exc_cases,coords = [], [], []


     for x,y in pt_s: 

     # ### direction choosing ###
          n = 13
          if (y+n) > w or (x+n) > h or (y-n) < 0 or (x-n) < 0:
               n = np.min((abs(0-x), (h-x), (w-y), abs(0-y)))
          
          # #Creating a 13x13/nxn kernel where x,y is the midpoint 
          kernel_1 = PATH_1[x-(n//2):x+(n//2+1), y-(n//2):y+(n//2+1)]

          # # find biggest sum
          UL = np.sum(kernel_1[:n//2, :n//2])
          UR = np.sum(kernel_1[:n//2, n//2+1:])
          LL = np.sum(kernel_1[n//2+1:, :n//2])
          LR = np.sum(kernel_1[n//2+1:, n//2+1:])


          # returning strings of quarter 
          quarters = np.array(["UL", "UR", "LL", "LR"])
     #      # https://stackoverflow.com/questions/17568612/how-to-make-numpy-argmax-return-all-occurrences-of-the-maximum
          winners = quarters[np.flatnonzero(np.array([UL, UR, LL, LR]) == np.max(np.array([UL, UR, LL, LR])))]

     # ####

          #print(winners)
     # # # if 2 quarters have high sums - px will become in the middle of the outer edge of a new kernel 
     # # # depending on the side of higher values, the px will be on one of the edges
     # # # if one quarter will have higher values - turning the kernel according to the location of the quarter 

     # #### conditioning for kernel edge size ####

          #h,w = PATH_1.shape[0],PATH_1.shape[1]
          
          n2 = int(np.ceil(np.max(dist)))
          #print(n2)
          # edge cases (if x or y are too close to the edge 
          if (y+n2) > w or (x+n2) > h or (y-n2) < 0 or (x-n2) < 0:
               n2 = np.min((abs(0-x), (h-x), (w-y), abs(0-y)))
          
               # point in close upper right corner - very unlikely case 

     # ####

     # # potential combos - UL+LL, UR+LR, UL+UR, LL+LR, and all quarters separately 
     # flooring w even kernel can become uneven - heads-up!
     # # refacto try-excepts please :) 
          # edge cases
          
  
          try:
               if len(winners) == 2:
                    if (np.array(["UR", "LR"]) == winners).all():

                         if n2 == int(np.ceil(np.max(dist))):
                              kernel_2 = thinned[ x-(n2//2): x+(n2//2 ), y:(y+n2)]
                              # function input origin is quarter specific 
                              x_new = x + (find_nearest_white(kernel_2, [n2//2, 0])[0] - n2//2)
                              y_new = y + find_nearest_white(kernel_2, [n2//2, 0])[1]

                         else:
                              n3 = int(np.ceil(np.max(dist)))

                              # upper right corner or lower right corner or right middle edge
                              if ((y+n3 > w) and (x-n3<0)) or ((y+n3) > w and (x+n3)>h) or ((y+n3) > w):
                                   kernel_2 = thinned[ x-(n2//2): x+(n2//2), y :]
                                   x_new = x + (find_nearest_white(kernel_2, [n2//2, 0])[0] - n2//2)
                                   y_new = y + find_nearest_white(kernel_2, [n2//2, 0])[1] 

                              # upper layer - upper left corner or upper middle or lower left corner or middle lower part of left middle
                              elif ((y-n3) < 0 and (x-n3) < 0) or ((x-n3) < 0) or ((x+n3) > h and (y-n3) < 0) or ((x+n3) > h) or (y-n3 <0):
                                   kernel_2 = thinned[ x-(n2//2): x+(n2//2), y : y+n3]

                                   x_new = x + (find_nearest_white(kernel_2, [n2//2, 0])[0] - n2//2)
                                   y_new = y + find_nearest_white(kernel_2, [n2//2, 0])[1] 


                    elif (np.array(["UL", "LL"]) == winners).all():
                       
                         if n2 == int(np.ceil(np.max(dist))):
                              kernel_2 = thinned[ x-(n2//2) : x+(n2//2), (y-n2):y]

                              x_new = x + (find_nearest_white(kernel_2, [n2//2, n2])[0] - n2//2)
                              y_new = y + (find_nearest_white(kernel_2, [n2//2, n2])[1] - n2)
                         
                         else:
                              n3 = int(np.ceil(np.max(dist)))
                 
                              # upper layer - upper left corner or lower left corner or middle left 
                              if ((y-n3) < 0 and (x-n3) < 0) or ( (y-n3) < 0 and (x+n3)>h) or ((y-n3) < 0):
                                   
                                   kernel_2 = thinned[ x-(n2//2) : x+(n2//2), :y]
                                   x_new = x + (find_nearest_white(kernel_2, [n2//2, y])[0] - n2//2)
                                   y_new = y + (find_nearest_white(kernel_2, [n2//2, y])[1] - y)


                              #  upper right corner or upper mid or  lower right corner or lower mid or middle right layer
                              elif  ((x-n3) <0 and (y+n3) >w) or ((x-n3) < 0) or ( (y+n3) > w and (x+n3)>h) or ((x+n3)>h) or ( (y+n3) > w):

                                   kernel_2 = thinned[  x-(n2//2) : x+(n2//2), (y-n3): y]

                                   x_new = x + (find_nearest_white(kernel_2, [n2//2, n3])[0] - n2//2)
                                   y_new = y + (find_nearest_white(kernel_2, [n2//2, n3])[1] - n3)
                                   

                    elif (np.array(["LL", "LR"]) == winners).all():
                         # some edge cases are kinda spreading in the wrong dir - have created bigger kernels than actually needed
               
                         if n2 == int(np.ceil(np.max(dist))):
                              kernel_2 = thinned[ x : (x+n2), y-(n2//2) :y + (n2//2)]

                              x_new = x + find_nearest_white(kernel_2, [0, n2//2])[0] 
                              y_new = y + (find_nearest_white(kernel_2, [0, n2//2])[1] - n2//2)

                         else: # edge case
                              n3 = int(np.ceil(np.max(dist)))

                              # lower layer - right corner or lower left corner or lower mid
                              if ((y+n3 > w) and (x+n3 > h)) or ((y-n3 < 0) and (x+n3 > h)) or (x+n3 >h):

                                   kernel_2 = thinned[ x:, y-(n2//2) :y + (n2//2)]

                                   x_new = x + find_nearest_white(kernel_2, [0, n2//2])[0] 
                                   y_new = y + (find_nearest_white(kernel_2, [0, n2//2])[1] - n2//2)

                              # upper right corner or upper left corner or left-mid or upper mid or right mid 
                              elif ((y+n3 > w) and (x-n3<0)) or ((y-n3 < 0) and (x-n3 <0)) or (y-n3 < 0) or (x-n3 < 0) or (y+n3 > w):

                                   kernel_2 = thinned[ x:x+n3, y-(n2//2) :y + (n2//2)]

                                   x_new = x + find_nearest_white(kernel_2, [0, n2//2])[0] 
                                   y_new = y + (find_nearest_white(kernel_2, [0, n2//2])[1] - n2//2)
                         


                    elif (np.array(["UL", "UR"]) == winners).all():

                         if n2 == int(np.ceil(np.max(dist))):
                              kernel_2 = thinned[ (x-n2) : x, y-(n2//2) :y + (n2//2)]

                              x_new = x + (find_nearest_white(kernel_2, [n2, n2//2])[0] - n2)
                              y_new = y + (find_nearest_white(kernel_2, [n2, n2//2])[1] - n2//2)

                         else: 
                              n3 = int(np.ceil(np.max(dist)))

                              # lower layer - right corner or lower left corner or lower-mid or left-mid or right-mid
                              if ((y+n3 > w) and (x+n3 > h)) or ((y-n3 < 0) and (x+n3 > h)) or ((x+n3 >h)) or (y-n3 < 0) or (y+n3 > w):

                                   kernel_2 = thinned[ (x-n3) :x,  y-(n2//2) :y + (n2//2)]

                                   x_new = x + (find_nearest_white(kernel_2, [n3, n2//2])[0] - n3)
                                   y_new = y + (find_nearest_white(kernel_2, [n3, n2//2])[1] - n2//2)


                              # upper right corner or upper left corner or upper mid
                              elif ((y+n3 > w) and (x-n3<0)) or ((y-n3 < 0) and (x-n3 <0)) or (x-n3 < 0):

                                   kernel_2 = thinned[ :x, y-(n2//2) :y + (n2//2)]

                                   x_new = x + (find_nearest_white(kernel_2, [x, n2//2])[0] - x)
                                   y_new = y + (find_nearest_white(kernel_2, [x, n2//2])[1] - n2//2)


               elif len(winners) == 1: # 1 winner or 3 which is unlikely but still sth to look out for 
                    if 'UR' in winners:    
                         if n2 == int(np.ceil(np.max(dist))):
                              kernel_2 = thinned[ x-n2: x, y:y+n2]

                              x_new = x + (find_nearest_white(kernel_2, [n2, 0])[0] - n2)
                              y_new = y + find_nearest_white(kernel_2, [n2, 0])[1]

                         else:
                              n3 = int(np.ceil(np.max(dist)))
                              

                              # upper right corner 
                              if ((y+n3 > w) and (x-n3<0)):

                                   kernel_2 = thinned[ :x, y:]

                                   x_new = x + (find_nearest_white(kernel_2, [x, 0])[0] - x)
                                   y_new = y + find_nearest_white(kernel_2, [x, 0])[1] 

                              # lower layer - right corner or right+mid
                              elif ((y+n3 > w) and (x+n3 > h)) or ((y+n3 > w)):

                                   kernel_2 = thinned[ (x-n3): x, y:]

                                   x_new = x + (find_nearest_white(kernel_2, [n3, 0])[0] - n3)
                                   y_new = y + find_nearest_white(kernel_2, [n3, 0])[1]

                              # upper left corner 
                              elif ((y-n3 < 0) and (x-n3 <0)) or ((x-n3 < 0)):
                                                                 
                                   kernel_2 = thinned[:x, y:y+n3]

                                   x_new = x + (find_nearest_white(kernel_2, [x, 0])[0] - x)
                                   y_new = y + find_nearest_white(kernel_2, [x, 0])[1] 

                              # lower left corner or lower mid or left mid
                              elif ((y-n3 < 0) and (x+n3 > h)) or (x+n3 >h) or (y-n3 < 0):
                                   kernel_2 = thinned[ x-n3: x, y:y+n3]

                                   x_new = x + (find_nearest_white(kernel_2, [n3, 0])[0] - n3)
                                   y_new = y + find_nearest_white(kernel_2, [n3, 0])[1]



                    elif 'UL' in winners:
                         # std cases
                         if n2 == int(np.ceil(np.max(dist))):
                              kernel_2 = thinned[ x-n2: x, y-n2:y]

                              x_new = x + (find_nearest_white(kernel_2, [n2, n2])[0] - n2)
                              y_new = y + (find_nearest_white(kernel_2, [n2, n2])[1] - n2)

                         else: 
                              n3 = int(np.ceil(np.max(dist)))

                              # upper left corner
                              if ((x-n3<0) and (y-n3<0)):
                                   kernel_2 = thinned[ : x, :y]

                                   x_new = x + (find_nearest_white(kernel_2, [x, y])[0] - x)
                                   y_new = y + (find_nearest_white(kernel_2, [x, y])[1] - y)

                              # upper mid or upper right corner
                              if (x-n3<0) or ((x-n3<0) and (y+n3>w)):
                                   kernel_2 = thinned[ : x, y-n3:y]

                                   x_new = x + (find_nearest_white(kernel_2, [x, n3])[0] - x)
                                   y_new = y + (find_nearest_white(kernel_2, [x, n3])[1] - n3)

                              # lower left corner or left mid
                              elif ((y-n3<0) and (x+n3>h)) or (y-n3<0):
                                   kernel_2 = thinned[ x-n3: x, :y]

                                   x_new = x + (find_nearest_white(kernel_2, [n3, y])[0] - n3)
                                   y_new = y + (find_nearest_white(kernel_2, [n3, y])[1] - y)

                              # right mid or lower mid or lower right corner
                              elif (y+n3>w) or (x+n3>h) or ((y+n3>w) and (x+n3>h)):
                                   kernel_2 = thinned[ x-n3: x, y-n3:y]

                                   x_new = x + (find_nearest_white(kernel_2, [n3, n3])[0] - n3)
                                   y_new = y + (find_nearest_white(kernel_2, [n3, n3])[1] - n3)



                    elif 'LL' in winners:

                         if n2 == int(np.ceil(np.max(dist))):
                              kernel_2 = thinned[ x: x + n2, y-n2:y]
                         
                              x_new = x + find_nearest_white(kernel_2, [0, n2])[0] 
                              y_new = y + (find_nearest_white(kernel_2, [0, n2])[1] - n2)

                         else: 
                              n3 = int(np.ceil(np.max(dist)))

                              # upper left corner 
                              if ((x-n3<0) and (y-n3<0)):
                                   kernel_2 = thinned[ : x , :y]
                              
                                   x_new = x + find_nearest_white(kernel_2, [0,y])[0] 
                                   y_new = y + (find_nearest_white(kernel_2, [0,y])[1] - y)

                              # lower mid or lower right corner (→stopped checking from here - 21.02 @14:47←)
                              elif (x+n3>h) or (y+n3>w and x+n3>h):

                                   kernel_2 = thinned[ x:, y-n3:y]

                                   x_new = x + find_nearest_white(kernel_2, [0, n3])[0] 
                                   y_new = y + (find_nearest_white(kernel_2, [0, n3])[1] - n3)

                              # upper mid or upper right corner or right mid
                              elif (x-n3<0) or ((x-n3<0) and (y+n3>w)) or (y+n3>w):

                                   kernel_2 = thinned[x: x + n3, y-n3:y]
                              
                                   x_new = x + find_nearest_white(kernel_2, [0,n3])[0] 
                                   y_new = y + (find_nearest_white(kernel_2, [0,n3])[1] - n3)


                              # lower left corner
                              elif ((y-n3<0) and (x+n3>h)):

                                   kernel_2 = thinned[x:, :y]
                              
                                   x_new = x + find_nearest_white(kernel_2, [0, y])[0] 
                                   y_new = y + (find_nearest_white(kernel_2, [0,y])[1] - y)

                              # left mid
                              elif (y-n3<0):
                                   kernel_2 = thinned[ x:x+n3 , :y]
                              
                                   x_new = x + find_nearest_white(kernel_2, [0,y])[0] 
                                   y_new = y + (find_nearest_white(kernel_2, [0,y])[1] - y)


                    elif 'LR' in winners:
                         if n2 == int(np.ceil(np.max(dist))):
                              kernel_2 = thinned[ x:x+n2, y:y+n2]

                         else: 
                              n3 = int(np.ceil(np.max(dist)))

                              # lower right corner
                              if (y+n3>w and x+n3>h):
                                   kernel_2 = thinned[ x:, y:]

                              # lower mid 
                              elif (x+n3>h):
                                   kernel_2 = thinned[ x:, y:y+n3]

                              # upper right corner or right mid
                              elif ((x-n3<0) and (y+n3>w)) or (y+n3>w):
                                   kernel_2 = thinned[x: x + n3, y:]
                              
                              # lower left corner
                              elif ((y-n3<0) and (x+n3>h)):
                                   kernel_2 = thinned[x:, y:y+n3]
                              
                              # upper left corner or left mid or upper mid
                              elif ((x-n3<0) and (y-n3<0)) or ((x-n3<0)):
                                   kernel_2 = thinned[ x:x+n3, y:y+n3]
                              

                         x_new = x + find_nearest_white(kernel_2, [0,0])[0] 
                         y_new = y + find_nearest_white(kernel_2, [0,0])[1] 



                         

               elif len(winners) == 3: # 3 winners has two cases - just creating bigger rectangles in that case 
                    if (np.array(["UL", "UR", "LL"]) == winners).all(): 
                         if n2 == int(np.ceil(np.max(dist))):
                              kernel_2 = thinned[x-n2 : x+(n2//2), y-n2:y+(n2//2)]

                              x_new = x + (find_nearest_white(kernel_2, [n2,n2])[0] - n2)
                              y_new = y + (find_nearest_white(kernel_2, [n2,n2])[1] - n2)

                         else:
                              n3 = int(np.ceil(np.max(dist)))

                              # upper left corner 
                              if ((x-n3<0) and (y-n3<0)):
                                   kernel_2 = thinned[ :x+(n2//2) , :y+(n2//2)]
                              
                                   x_new = x + (find_nearest_white(kernel_2, [x,y])[0] - x)
                                   y_new = y + (find_nearest_white(kernel_2, [x,y])[1] - y)

                              
                              # upper mid or upper right corner 
                              elif (x-n3<0) or ((x-n3<0) and (y+n3>w)):

                                   kernel_2 = thinned[:x+(n2//2), y-n3:y+(n2//2)]
                              
                                   x_new = x + (find_nearest_white(kernel_2, [x,n3])[0] - x)
                                   y_new = y + (find_nearest_white(kernel_2, [x,n3])[1] - n3)


                              # lower mid or lower right corner or right mid
                              elif (x+n3>h) or (y+n3>w and x+n3>h) or  (y+n3>w):

                                   kernel_2 = thinned[x-n3 : x+(n2//2), y-n3:y+(n2//2)]

                                   x_new = x + (find_nearest_white(kernel_2, [n3,n3])[0] - n3)
                                   y_new = y + (find_nearest_white(kernel_2, [n3,n3])[1] - n3)


                              # lower left corner
                              elif ((y-n3<0) and (x+n3>h)):

                                   kernel_2 = thinned[x:, :y+(n2//2)]
                              
                                   x_new = x + find_nearest_white(kernel_2, [0, y])[0] 
                                   y_new = y + (find_nearest_white(kernel_2, [0,y])[1] - y)

                              # left mid
                              elif (y-n3<0):
                                   kernel_2 = thinned[ x-n3 : x+(n2//2), :y+(n2//2)]
                              
                                   x_new = x + find_nearest_white(kernel_2, [0,y])[0] 
                                   y_new = y + (find_nearest_white(kernel_2, [0,y])[1] - y)




                    elif (np.array(["UL", "UR", "LR"]) == winners).all(): 
                         if n2 == int(np.ceil(np.max(dist))):
                              kernel_2 = thinned[x-n2: x+(n2//2), y-(n2//2) : y+n2]

                              x_new = x + (find_nearest_white(kernel_2, [n2,n2//2])[0] - n2)
                              y_new = y + (find_nearest_white(kernel_2, [n2,n2//2])[1] - n2//2)

                         else:
                              n3 = int(np.ceil(np.max(dist)))

                              # upper right corner 
                              if ((x-n3<0) and (y+n3>w)):

                                   kernel_2 = thinned[:x+(n2//2), y-(n2//2):]
                              
                                   x_new = x + (find_nearest_white(kernel_2, [x,n2//2])[0] - x)
                                   y_new = y + (find_nearest_white(kernel_2, [x,n2//2])[1] - n2//2)

                              # upper left corner or upper mid
                              elif ((x-n3<0) and (y-n3<0)) or (x-n3<0):
                                   kernel_2 = thinned[ :x+(n2//2) ,  y-(n2//2):y+n3]
                              
                                   x_new = x + (find_nearest_white(kernel_2, [x,n2//2])[0] - x)
                                   y_new = y + (find_nearest_white(kernel_2, [x,n2//2])[1] - n2//2)

                              # lower right corner or right mid
                              elif ((y+n3>w) and (x+n3>h)) or (y+n3>w): 

                                   kernel_2 = thinned[ (x-n3):x+(n2//2) ,  y-(n2//2):]

                                   x_new = x + (find_nearest_white(kernel_2, [n3,n2//2])[0] - n3)
                                   y_new = y + (find_nearest_white(kernel_2, [n3,n2//2])[1] - n2//2)

                              # lower mid  or lower left corner or left mid 
                              elif (x+n3>h) or ((y-n3<0) and (x+n3>h)) or (y-n3<0):

                                   kernel_2 = thinned[ (x-n3):x+(n2//2) ,  y-(n2//2):y+n3]

                                   x_new = x + (find_nearest_white(kernel_2, [n3,n2//2])[0] - n3)
                                   y_new = y + (find_nearest_white(kernel_2, [n3,n2//2])[1] - n2//2)

                    
                    elif (np.array(["UL", "LL", "LR"]) == winners).all(): 
                         if n2 == int(np.ceil(np.max(dist))):
                              kernel_2 = thinned[x-(n2//2): x+n2, y-n2:y+(n2//2)]

                              x_new = x + (find_nearest_white(kernel_2, [n2//2,n2])[0] - n2//2)
                              y_new = y + (find_nearest_white(kernel_2, [n2//2,n2])[1] - n2)

                         else:
                              n3 = int(np.ceil(np.max(dist)))

                              # upper right corner or upper mid or lower right corner or right mid
                              if ((x-n3<0) and (y+n3>w)) or (x-n3<0) or (y+n3>w): 

                                   kernel_2 = thinned[x-(n2//2): x+n3, y-n3:y+(n2//2)]

                                   x_new = x + (find_nearest_white(kernel_2, [n2//2,n3])[0] - n2//2)
                                   y_new = y + (find_nearest_white(kernel_2, [n2//2,n3])[1] - n3)

                              
                              # lower left corner 
                              elif ((y-n3<0) and (x+n3>h)):

                                   kernel_2 = thinned[ x-(n2//2):,  :y+(n2//2)]
                              
                                   x_new = x + (find_nearest_white(kernel_2, [n2//2, y])[0] - n2//2)
                                   y_new = y + (find_nearest_white(kernel_2, [n2//2, y])[1] - y)


                              # upper left corner or left mid 
                              elif ((x-n3<0) and (y-n3<0)) or (y-n3<0):
                                   kernel_2 = thinned[ x-(n2//2): x+n3,  :y+(n2//2)]
                              
                                   x_new = x + (find_nearest_white(kernel_2, [n2//2, y])[0] - n2//2)
                                   y_new = y + (find_nearest_white(kernel_2, [n2//2, y])[1] - y)


                              # lower right corner or lower mid 
                              elif ((y+n3>w) and (x+n3>h)) or (x+n3>h):

                                   kernel_2 = thinned[ x-(n2//2):,  (y-n3):y+(n2//2)]
                              
                                   x_new = x + (find_nearest_white(kernel_2, [n2//2, n3])[0] - n2//2)
                                   y_new = y + (find_nearest_white(kernel_2, [n2//2, n3])[1] - n3)




                    elif (np.array(["UR", "LL", "LR"]) == winners).all(): 
                         if n2 == int(np.ceil(np.max(dist))):
                             
                              kernel_2 = thinned[x-(n2//2) : x+n2, y-(n2//2):y+n2]


                         else:
                              n3 = int(np.ceil(np.max(dist)))

                              # upper left corner or mid left or upper mid 
                              if ((x-n3<0) and (y-n3<0)) or (x-n3<0) or (y-n3<0):
                                   kernel_2 = thinned[x-(n2//2): x+n3, y-(n2//2):y+n3]

                              # lower right corner
                              elif ( (y+n3>w) and (x+n3>h) ):
                                   kernel_2 = thinned[x-(n2//2):, y-(n2//2):]


                              # upper right corner or mid right
                              elif ((x-n3<0) and (y+n3>w)) or  (y+n3>w):
                                   kernel_2 = thinned[x-(n2//2):x+n3, y-(n2//2):]


                              # low mid or lower left corner
                              elif (x+n3>h) or ((x+n3>h) and (y-n3<0)):
                                   kernel_2 = thinned[x-(n2//2):, y-(n2//2):y+n3]


                         x_new = x + (find_nearest_white(kernel_2, [n2//2, n2//2])[0] - n2//2)
                         y_new = y + (find_nearest_white(kernel_2, [n2//2, n2//2])[1] - n2//2)
                                                                 


               
               px_dist = dist[x_new][y_new]
               
               # extra condition in case of marker starting at a pointer
               dm = int(2 * px_dist * nano_per_px)
               # for visualisation purposes 
               coords.append((x,y,x_new, y_new))
               if dm != 0:    dm_s.append(dm)               
               
          except: 
               # cases where kernel_2 only has 0-s need to probably create long slits along a specific edge :))
               # re-running the exceptions - get rids of most of them - for some reason unknown
               exc_cases.append((x,y,winners))
               #exc_cases.append((x,y))

     return (dm_s, exc_cases, coords)