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
TRACK_WIDTH = 200

TRACK_LENGTH = 1250
NUM_SAMPLES = 100

START = 250
ANGLES = [0,math.pi/2,math.pi,3*math.pi/2]  


class Track():

    seed = 1.0
    def Load(self):
       #Compose Track of Rectangles 
       
       self.finish = pygame.Rect(TRACK_LENGTH/2,TRACK_LENGTH,200,TRACK_WIDTH)
       self.inbox = False 
       mean = np.zeros(2)+np.array([1000,1000])
       #80000
       cov = np.eye(2)*40000
      
       self.data = np.random.multivariate_normal(mean,cov,NUM_SAMPLES)

       self.lap = 0
       self.hull = ConvexHull(self.data)
       self.genTrack()


    def genTrack(self):
     
      self.track = []
      self.angle = [] 
      self.points = []   
      for i in range(self.hull.nsimplex-1):
        idx1 = self.hull.vertices[i]
        idx2 = self.hull.vertices[i+1]
        p1 = self.data[idx1,:]
        p2 = self.data[idx2,:]

        self.points.append((p1[0],p1[1]))
       
        tr = TrackPiece(p1,p2)
        self.track.append(tr)
      
      idx0 = self.hull.vertices[0]
      tr = TrackPiece(p2,self.data[idx0,:])
      self.track.append(tr)
      self.points.append((p2[0],p2[1]))

    def returnStart(self):
      return self.track[2].theta,self.track[2].center


    def getLap(self,x,y):
      if(self.finish.collidepoint(x,y) and not self.inbox):
        self.lap += 1
        self.inbox = True 
      elif(not self.finish.collidepoint(x,y) and self.inbox):
        self.inbox = False

      return self.lap 


    def closestRectangle(self,car):
      idx = self.closest_rectangle_to_car(car)

      return self.track[idx].theta, self.track[idx].projectToPiece(car.cords)




    def closest_rectangle_to_car(self, car=None, x=None, y=None):
      """
      Returns the index of the closest rectangle to the car
      """
      dist = 1e6
      pos = car.cords
      
      for i in range(self.hull.nsimplex):
        tr = self.track[i]
        cent = tr.center
        # print "CAR POS: ",pos," cent ",cent," tr ",i
        if dist > LA.norm(pos-cent):
          dist = LA.norm(pos-cent)
          index = i
          closest = tr

      return index

    def desired_rectangle_angle(self, car):
      """
      Returns the desired rectangle angle based on the
      car's current location.
      """
      desired_rectangle_index = self.desired_rectangle(car)
      
      return self.track[desired_rectangle_index].theta*57.29

    def distance_from_current_rectangle(self, car, pos):
      desired_rectangle_index = self.desired_rectangle(car)
      tr = self.track[desired_rectangle_index]
      cent = tr.center
      dist = LA.norm(pos-cent)
      return dist

    def desired_rectangle(self, car):
      """
      Returns the rectangle that the car is currently on.
      """
      current_rectangles = []
      for i in range(len(self.track)):
        #print "Track ",i," ",self.track[i].isOnPiece(car.cords)," ",car.cords," ",self.track[i].center
        if self.track[i].isOnPiece(car.cords):
          current_rectangles += [i]
      if len(current_rectangles) == 0:
        current_rectangles = [self.closest_rectangle_to_car(car)]
      if len(current_rectangles) == 1:
        current_rectangles = current_rectangles[0]
      elif len(current_rectangles) == 2:
        # Edge case: Rectangles 0, 3
        if 0 in current_rectangles and self.hull.nsimplex-1 in current_rectangles:
          current_rectangles = 0
        # Otherwise take latest rectangle
        else:
          current_rectangles = max(current_rectangles)
      elif len(current_rectangles) >= 3: 
        current_rectangles = current_rectangles[1]
      # print "current_rectangles", current_rectangles

 
      return current_rectangles

    def IsOnTrack(self,car):
      T = False 
      for tr in self.track: 
        if(tr.isOnPiece(car.cords)):
          T = True

      return T

    def Draw(self,screen,car_pos):

      draw_points = []
      for i in range(len(self.points)-1):
        p0 = self.points[i]
        p1 = self.points[i+1]
        p0 = (p0[0] - car_pos[0], p0[1] - car_pos[1])
        p1 = (p1[0] - car_pos[0], p1[1]- car_pos[1])

        pygame.draw.line(screen, BLACK, p0, p1, TRACK_WIDTH)

      p0 = self.points[0]
      p0 = (p0[0]-car_pos[0], p0[1] - car_pos[1])
      pygame.draw.line(screen, BLACK, p1, p0, TRACK_WIDTH)
      
    
