import pygame
import math
import random
import IPython
import numpy as np
from numpy import linalg as LA
import random
from scipy.spatial import ConvexHull
from numpy import linalg as LA


xs = 600
ys = 450
xt = xs #- 100
yt = ys #+ 100
dt = 1.0
BLACK = (0,0,0)
BLUE = (  0,   0, 255)
TRACK_WIDTH = 200

TRACK_LENGTH = 1250
NUM_SAMPLES = 100

START = 250
ANGLES = [0,math.pi/2,math.pi,3*math.pi/2]  


class TrackPiece():

    seed = 1.0
    def __init__(self,p0,p1):
      self.p0 = p0
      self.p1 = p1

      self.length = LA.norm(p1-p0)
    
      self.center = (p1 - p0)/2.0+p0
      #Calculate Angle of Piece
      vec = p1 - p0
      vec = vec/ LA.norm(vec)

      cos_a = np.sum(np.array([0,-1])*vec)
      if(vec[0]>= 0.0):
        self.theta = np.arccos(cos_a)
      else:
        self.theta = cos_a - math.pi/2 


    def distToCenter(self,point):

      a = self.p1[0] - self.p0[0]
      b = self.p0[1] - point[1]
      c = self.p0[1] - point[0]
      d = self.p1[1] - self.p0[1]

      denom = (a*b-c*d)
      dist = denom/self.length
      return dist
   
    def isOnPiece(self,point):
      #Check if point is on both sides of line for the two pairs

      point = point - self.center
      princ_axe_big = (self.p1 - self.center)/LA.norm(self.p1-self.center)

      rot_90 = np.array(([0.0,-1.0],[1.0,0.0]))

      princ_axe_small = np.dot(rot_90,princ_axe_big)
      
      if(np.abs(np.sum(princ_axe_big*point))<self.length/2+TRACK_WIDTH/2 and np.abs(np.sum(princ_axe_small*point))<TRACK_WIDTH/2):
        return True 
      else: 
        return False 

    def projectToPiece(self,cords):
   
      AP = cords   - self.p0
      AB = self.p1 - self.p0  

      pos = self.p0 + np.sum((AP*AB))/np.sum((AB*AB))*AB

      if(not self.isOnPiece(pos)):
        distA = LA.norm(pos - self.p0)
        distB = LA.norm(pos - self.p1)
        if(distA > distB):
          return self.p0
        else: 
          return self.p1 

      return pos

    