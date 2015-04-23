import pygame
import math
import random
import IPython
import numpy as np
from numpy import linalg as LA
xs = 600
ys = 450
xt = xs - 100
yt = ys + 100
dt = 1.0
BLACK = (0,0,0)
BLUE = (  0,   0, 255)
TRACK_WIDTH = 200
TRACK_LENGTH = 1250 
START = 0
ANGLES = [0,math.pi/2,math.pi,3*math.pi/2]  
class Track():


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

       self.finish = pygame.Rect(TRACK_LENGTH/2,START,200,TRACK_WIDTH)
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
      cars_per_track = num_cars/4
      car_list = []
      for tr in self.track:
        for i in range(cars_per_track):
          car = [random.randint(tr.left+20, tr.right-20),random.randint(tr.top+20,tr.bottom-20)]
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

      if(angle == None):
        IPython.embed()
      #print rec,angle
      return angle

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
        print tl,width,height
        pygame.draw.rect(screen,BLACK,rect)

      tr = self.finish
      tl = tr.topleft
      tl_t = [tl[0] - car_pos[0],tl[1]-car_pos[1]]
        

      tri = tr.bottomright
        
      width = abs(tl[0] - tri[0])
      height = abs(tl[1] - tri[1])
      dim = [width,height]
      rect = [tl[0] - car_pos[0],tl[1]-car_pos[1],width,height]
      #rect = [tl[0],tl[1],width,height]
      print tl,width,height
      pygame.draw.rect(screen,BLUE,rect)
