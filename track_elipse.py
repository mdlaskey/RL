import pygame
import math
import random
import IPython
import numpy as np
from numpy import linalg as LA
import random
from scipy.spatial import ConvexHull
from numpy import linalg as LA
from Tools.TrackPiece import TrackPiece


xs = 600
ys = 450
xt = xs #- 100
yt = ys #+ 100
dt = 1.0
BLACK = (0,0,0)
BLUE = (  0,   0, 255)
a = 1000
b = 500 

NUM_SAMPLES = 100

START = 250
ANGLES = [0,math.pi/2,math.pi,3*math.pi/2]  


class Track():

    seed = 1.0

    def Draw(self,screen,car_pos):
      rect = pygame.Rect(0,0,a,b)
      #pygame.draw.ellipse(screen,BLACK,rect,200)
           
    
