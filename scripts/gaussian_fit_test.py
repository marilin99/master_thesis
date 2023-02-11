# https://stackoverflow.com/questions/50559569/how-can-i-make-my-2d-gaussian-fit-to-my-image

import numpy as np
import os
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import cv2


def gaussian(xycoor,x0, y0, sigma, amp):
    '''This Function is the Gaussian Function'''

    x, y = xycoor # x and y taken from fit function.  Stars at 0, increases by 1, goes to length of axis
    A = 1 / (2*sigma**2)
    eq =  amp*np.exp(-A*((x-x0)**2 + (y-y0)**2)) #Gaussian
    return eq


def fit(image):
    med = np.median(image)
    image = image-med
   

    max_index = np.where(image >= np.max(image))
    x0 = max_index[1] #Middle of X axis
    y0 = max_index[0] #Middle of Y axis
    x = np.arange(0, image.shape[1], 1) #Stars at 0, increases by 1, goes to length of axis
    y = np.arange(0, image.shape[0], 1) #Stars at 0, increases by 1, goes to length of axis
    xx, yy = np.meshgrid(x, y) #creates a grid to plot the function over
    sigma = np.std(image) #The standard dev given in the Gaussian
    amp = np.max(image) #amplitude
    guess = [x0, y0, sigma, amp] #The initial guess for the gaussian fitting

    low = [0,0,0,0] #start of data array
    #Upper Bounds x0: length of x axis, y0: length of y axis, st dev: max value in image, amplitude: 2x the max value
    upper = [image.shape[0], image.shape[1], np.max(image), np.max(image)*2] 
    bounds = [low, upper]

    params, pcov = curve_fit(gaussian, (xx.ravel(), yy.ravel()), image.ravel(),p0 = guess, bounds = bounds) #optimal fit.  Not sure what pcov is. 

    return params


def plotting(image, params):
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.scatter(params[0], params[1],s = 10, c = 'red', marker = 'x')
    circle = Circle((params[0], params[1]), 10*params[2], facecolor = 'none', edgecolor = 'red', linewidth = 1)

    ax.add_patch(circle)
    plt.show()


red_chan = cv2.imread("red_intensities.png",0)


# adaptive hg eq - https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html
clahe1 = cv2.createCLAHE(clipLimit=1, tileGridSize=(30,30))
cl1 = clahe1.apply(red_chan)

app_img = cl1
pwr = 2
red_chan_eq = np.uint8((np.float128(app_img)**pwr / np.amax(np.float128(app_img)**pwr) ) * 255)

parameters = fit(red_chan_eq)

#generates a gaussian based on the parameters given
plotting(red_chan_eq, parameters)