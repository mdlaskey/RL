import pygame
import math
import random
import IPython
import numpy as np
from numpy import linalg as LA
import random

xs = 600
ys = 450
xt = xs #- 100
yt = ys #+ 100
dt = 1.0
BLACK = (0,0,0)
BLUE = (  0,   0, 255)
TRACK_WIDTH = 200

TRACK_LENGTH = 1250

START = 250
ANGLES = [0,math.pi/2,math.pi,3*math.pi/2]  


class Track():

    seed = 1.0
    def Load(self):
       #Compose Track of Rectangles 
       self.lap = 1
       r1_x_end = START+TRACK_LENGTH
       r1_y_end = START+TRACK_WIDTH

       r1 = pygame.Rect(START,START,TRACK_LENGTH,TRACK_WIDTH)

       r2_x = START+TRACK_LENGTH-TRACK_WIDTH
       r2_y = START

       r2 = pygame.Rect(r2_x,r2_y,TRACK_WIDTH,TRACK_LENGTH)

       r3_x = START
       r3_y = START+TRACK_LENGTH-TRACK_WIDTH

       r3 = pygame.Rect(r3_x,r3_y,TRACK_LENGTH,TRACK_WIDTH)

       r4_x = START
       r4_y = START

       r4 = pygame.Rect(r4_x,r4_y,TRACK_WIDTH,TRACK_LENGTH)

       self.track = [r1,r2,r3,r4]

       center = TRACK_LENGTH+TRACK_WIDTH+300
       self.mid_cords = np.array([float(center)/2.0,float(center)/2.0])

       self.radius = float(center)/2.0
 

       self.finish = pygame.Rect(TRACK_LENGTH/2,r3_y,200,TRACK_WIDTH)
       self.inbox = False 

    def returnStart(self):
      return self.track[2].center

    def center_rec(self,x,y,rect,rec_num):
      y_c = y-rect.centery
      x_c = x-rect.centerx

      if(abs(x_c) < abs(y_c)):
        if(rec_num == 3):
          x_c = - x_c
        return x_c
      else: 
        if(rec_num == 0): 
          y_c = -y_c 
        return y_c 

    def genCars(self,num_cars):
      """
      Randomly generates dummy cars and their
      corresponding positions. Ensures that
      none of them overlap.
      """
      def is_overlapped(car_list, current_car):
        for car in car_list:
          if abs(car[0] - current_car[0]) < 30 and abs(car[1] - current_car[1]) < 30:
            return True
        return False
      cars_per_track = num_cars/4
      car_list = []
      random.seed(self.seed)
      for tr in self.track:
        width = tr.right-50- tr.left+50
        height = tr.top+50 - tr.bottom-50
        widthInt = width/5
        heightInt = height/3
        heightPos = range(tr.bottom-50,tr.top+50,heightInt)
        widthPos = range(tr.left+50,tr.right-50,widthInt)

        for i in range(cars_per_track):
          generating_car = True
          while generating_car:
            hP = random.randint(0,len(heightPos)-1)
            wP = random.randint(0,len(widthPos)-1)
            car = [heightPos[hP],widthPos[wP]]
            if is_overlapped(car_list, car):
              continue
            else:
              generating_car = False
              car_list.append(car)

      return car_list 

    def getCorners(self,x,y):
      dist_corners = np.zeros(4)
      car = np.array([x,y])
      corners = [] 
      
      c0_y = self.track[0].centery
      c0_x = self.track[0].midleft[0] 

      corners.append(np.array([c0_x,c0_y]))

      c1_y = self.track[1].midtop[1]
      c1_x = self.track[1].centerx
      
      corners.append(np.array([c1_x,c1_y]))

      c2_y = self.track[2].centery
      c2_x = self.track[2].midright[0]

      corners.append(np.array([c2_x,c2_y]))

      c3_y = self.track[3].midbottom[1]
      c3_x = self.track[3].centerx

      corners.append(np.array([c3_x,c3_y]))

      for i in range(4):
        dist_corners[i] = np.exp(LA.norm(car - corners[i])**2/(-2*(200)**2))

      return dist_corners  



    def getDistance(self,x,y):
      dist = np.zeros(2)
      rec = 0
      first = False
      for tr in self.track: 
        if(tr.collidepoint(x,y)):
          if(not first):
            dist[0] = self.center_rec(x,y,tr,rec)
            first = True
          elif(first):
            dist[1] = self.center_rec(x,y,tr,rec)
        rec+=1
      if(self.track[0].collidepoint(x,y) and self.track[3].collidepoint(x,y)):
        temp = dist[0]
        dist[0] = dist[1]
        dist[1] = temp 
      if(not first):
        card_cords = np.array([x,y])
        distances = []
        for tr in self.track:
          tr_center = np.array([tr.center[0],tr.center[1]])
          distances.append(LA.norm(tr_center-card_cords))

        distances.sort()
          
        dist[0] = distances[0]
      #print rec,angle
      return dist

    def getLap(self,x,y):
      if(self.finish.collidepoint(x,y) and not self.inbox):
        self.lap += 1
        self.inbox = True 
      elif(not self.finish.collidepoint(x,y) and self.inbox):
        self.inbox = False

      return self.lap 


    def closestRectangle(self,pos):
      dist = 1e6
      for i in range(4):
        tr = self.track[i]
        cent = np.array([tr.centerx,tr.centery])
        if(dist > LA.norm(pos-cent)):
          dist = LA.norm(pos-cent)
          idx = i
          closest = tr

      if(idx == 1 or idx == 3):
        pos[0] = closest.centerx
      else:
        pos[1] = closest.centery

      return ANGLES[idx],pos


    def currentRectangle(self,x,y):
      angle = None
      rec = 0
      first = False
      dist_cent = 0
      for tr in self.track:
        if(tr.collidepoint(x,y)):
          if(not first):
            dist_cent = self.center_rec(x,y,tr,rec)
          
            first = True
            angle = ANGLES[rec]
          elif(first):
            if(self.center_rec(x,y,tr,rec) >= dist_cent or self.center_rec(x,y,tr,rec) >= 90):
              angle = ANGLES[rec]
            
            dist_cent = self.center_rec(x,y,tr,rec)
        rec+=1
      if(self.track[0].collidepoint(x,y) and self.track[3].collidepoint(x,y)):
        if( self.center_rec(x,y,tr,rec) >= 90 or self.center_rec(x,y,self.track[0],0) >= dist_cent):
          angle = ANGLES[0]

      #if(angle == None):
        #IPython.embed()
      #print rec,angle
      # if angle == None:
      #   index = self.closest_rectangle_to_car(x=x, y=y)
      #   angle = ANGLES[index]
      return angle

    def closest_rectangle_to_car(self, car=None, x=None, y=None):
      """
      Returns the index of the closest rectangle to the car
      """
      dist = 1e6
      pos = np.array([car.xc, car.yc])
      for i in range(4):
        tr = self.track[i]
        cent = np.array([tr.centerx,tr.centery])
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
      return ((desired_rectangle_index + 1) % 4) * 90

    def distance_from_current_rectangle(self, car, pos):
      desired_rectangle_index = self.desired_rectangle(car)
      tr = self.track[desired_rectangle_index]
      cent = np.array([tr.centerx,tr.centery])
      dist = LA.norm(pos-cent)
      return dist

    def desired_rectangle(self, car):
      """
      Returns the rectangle that the car is currently on.
      """
      current_rectangles = []
      for i in range(len(self.track)):
        if self.track[i].collidepoint(car.xc, car.yc):
          current_rectangles += [i]
      if len(current_rectangles) == 0:
        current_rectangles = [self.closest_rectangle_to_car(car)]
      if len(current_rectangles) == 1:
        current_rectangles = current_rectangles[0]
      elif len(current_rectangles) == 2:
        # Edge case: Rectangles 0, 3
        if 0 in current_rectangles and 3 in current_rectangles:
          current_rectangles = 0
        # Otherwise take latest rectangle
        else:
          current_rectangles = max(current_rectangles)
      # print "current_rectangles", current_rectangles
      return current_rectangles

    def IsOnTrack(self,car):
      T = False 
      for tr in self.track: 
        if(tr.collidepoint(car.xc,car.yc)):
          T = True

      return T

    def Draw(self,screen,car_pos):
  		
      for tr in self.track:
  			#IPython.embed()
        tl = tr.topleft
        tl_t = [tl[0] - car_pos[0],tl[1]-car_pos[1]]
  			

        tri = tr.bottomright
  			
        width = abs(tl[0] - tri[0])
        height = abs(tl[1] - tri[1])
        dim = [width,height]
        rect = [tl[0] - car_pos[0],tl[1]-car_pos[1],width,height]
        #rect = [tl[0],tl[1],width,height]
      
        pygame.draw.rect(screen,BLACK,rect)
 
      mid_x = int(self.mid_cords[0] - car_pos[0])
      mid_y = int(self.mid_cords[1] - car_pos[1])
      #pygame.draw.circle(screen,BLUE,(mid_x,mid_y),int(self.radius),10)
      #pygame.draw.circle(screen,BLUE,(mid_x,mid_y),int(self.radius / 2),10)
      tr = self.finish
      tl = tr.topleft
      tl_t = [tl[0] - car_pos[0],tl[1]-car_pos[1]]
        

      tri = tr.bottomright
        
      width = abs(tl[0] - tri[0])
      height = abs(tl[1] - tri[1])
      dim = [width,height]
      rect = [tl[0] - car_pos[0],tl[1]-car_pos[1],width,height]
      #rect = [tl[0],tl[1],width,height]
      #print tl,width,height
      pygame.draw.rect(screen,BLUE,rect)
