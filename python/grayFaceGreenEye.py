#getting the projection of the gradient as a function of the angle 
# defining a pixel as row, col

# get gray face with green lens overlayed - finished version 1
import numpy as np
import cv2
import copy as cp
import os
import cv2.cv as cv
import time
import scipy as sp
from scipy import signal
import sys

PathForRoot = sys.argv[1] + '/assets'

fo = open(PathForRoot + '/' + sys.argv[2], "w+")
fo.write("http://google.com/")
