import pygame
import math
import random
import numpy as np
import IPython

from numpy import linalg as LA

xs = 600
ys = 450
xt = xs - 100
yt = ys + 100
dt = 1.0
font = pygame.font.Font(None,24)
msg = []
msg += ["STOP"]
msg += ["GEAR 1"]
msg += ["GEAR 2"]
msg += ["GEAR 3"]
msg += ["GEAR 4"]

BLUE = (  0,   0, 255)
GREEN = (  0,   255, 0)

DIST_THRESH = 200 

n = len(msg)
gears = []
for i in range(n):
    gears += [font.render(msg[i],1,(250,250,250))]
speedo = []
laps = []
for i in range(101):
    speedo += [font.render("SPEED "+str(i),1,(250,250,250))]
    laps += [font.render("LAP "+str(i),1,(250,250,250))]
pygame.mixer.init()
sound = pygame.mixer.Sound("sound/racing_car.wav")
sound.set_volume(0)
sound.play(loops=-1)
shiftup_sound = pygame.mixer.Sound("sound/shift_up.wav")
idle_sound = pygame.mixer.Sound("sound/idle_rev.wav")
idle_sound.set_volume(0)
idle_sound.play(loops=-1)
                                
class Sprite():
    def Load(self,path,NF,start):
        self.view = 270
        self.images = []
        self.NF = NF
        self.start = start 
        self.start_v = self.view 
        self.xc = start[0]
        self.yc = start[1]
        self.xf = float(self.xc)
        self.yf = float(self.yc)
        self.speed = 0
        self.gear = 3
        self.wobble = 0
        self.lap = 0
        self.pastId = 0
        self.cords = np.array([self.xf,self.yf])
        self.pre_cords = self.cords
        self.carsHit = 0
        self.timeOffTrack = 0
        for f in range(NF):
            nv = len(str(f+1))
            name = path+'/fr_'
            if nv == 1:
                name += '000'
            if nv == 2:
                name += '00'
            if nv == 3:
                name += '0'
            self.images += [pygame.image.load(name+str(f+1)+'.png')]
    def Draw(self,x,y,screen):
        view = self.view + int(random.gauss(0,self.wobble))

        if view < 0 :
            view = view + 360
        view = view%360

        self.car_pos = [x,y]
        self.screen = screen

        screen.blit(self.images[view],(x-32,y-32))
        #screen.blit(gears[self.gear],(xt,yt))
        indicated = int(10.0*self.speed)
        #screen.blit(speedo[indicated],(xt+100,yt))
        #screen.blit(laps[self.lap],(xt,yt+50))
        if self.lap > MAX_LAPS :
            elapsed_time = font.render(str(frames/24),1,(250,250,250))
            #screen.blit(elapsed_time,(xt+100,yt+50))


    def getRotate(self,d_cords):
        #convert to radians 
        theta = self.view/57.296
        rot_mat = np.matrix([[math.cos(theta),-math.sin(theta)],\
                             [math.sin(theta),math.cos(theta)]])     
        
        dif = np.array([d_cords - self.cords]) 
        dif = dif/LA.norm(dif)
        rot_d_car = dif.T
        rot_d_car_cord = rot_mat*dif.T
        #rot_d_car = rot_d_car/LA.norm(rot_d_car)
        #print "ROT", rot_d_car,LA.norm(rot_d_car)

        perp = np.array([[1,0]])
        perp = rot_mat*perp.T

        #IPython.embed()
        theta = math.acos(perp.T*rot_d_car)
        #return [theta,rot_d_car,perp]

        front = np.array([[0,1]])
        back = np.array([[0,-1]])

        front = rot_mat*front.T
        back = rot_mat*back.T

        dist_f = LA.norm(front-rot_d_car)
        dist_b = LA.norm(back-rot_d_car)



        if(dist_f > dist_b):
            return [theta,rot_d_car,perp]
        else: 
            return [0,rot_d_car,perp]

    def sort_func(self,d):
        return d[0]

    def reset(self,dummycars):
        self.xf = self.start[0]
        self.yf = self.start[1]

        self.view = self.start_v
        for d in dummycars:
            d.xf = d.startXf
            d.yf = d.startYf


    def isCrashed(self,track):
        mid_cords = track.mid_cords
        radius = track.radius  
        dist = LA.norm(mid_cords -self.cords)

        if(dist > radius + 100):
            return True
        else:
            return False 


    def getState(self,track,dummycars):
        state = np.zeros(2) 
        state[0:2] = self.cords
        #state[1:3] = track.getDistance(self.xc,self.yc)
        dist_list = []
        for d_car in dummycars:
            d = np.zeros(2)
            dist = LA.norm(d_car.cords -self.cords)
            theta,vec,perp = self.getRotate(d_car.cords)
          
            if(dist < DIST_THRESH and theta > 0):
                d[0] = dist 
                d[1] = theta 
                #IPython.embed()
                print "ROT",vec
                print "PERP",perp
                print "THETA", theta*57.296
                endpoint = self.car_pos+perp.T*20
                endpoint = [endpoint[0,0],endpoint[0,1]]

                endpoint = self.car_pos+vec.T*20
                endpoint = [endpoint[0,0],endpoint[0,1]]

            dist_list.append(d)


        dist_list = sorted(dist_list,reverse = True, key = self.sort_func)
        cars = dist_list[0:1]
        #for c in cars: 
           # state = np.hstack([state,c])
            
        return state

    def updateStats(self,track,dummycars):
        for d_car in dummycars:
            d = np.zeros(2)
            dist = LA.norm(d_car.cords -self.cords)
            if(dist < 15 and d_car.id != self.pastId):
                self.carsHit += 1
                self.pastId = d_car.id 
        
        if(not track.IsOnTrack(self)):
            self.timeOffTrack +=1 


    def Update(self):
        self.speed = .95*self.speed + .05*(2.5*self.gear)
        print self.gear,'\t',int(10.0*self.speed),'\t',self.lap
        

        theta = self.view/57.296
        if self.wobble :
            idle_sound.set_volume(1.)
        else :
            idle_sound.set_volume(0)

        vx = self.speed*math.sin(theta)
        vy = -self.speed*math.cos(theta)
        self.xf = self.xf + vx*dt
        self.yf = self.yf + vy*dt
        self.pre_cords = self.cords 
        self.cords[0] = self.xf
        self.cords[1] = self.yf
        self.xc = int(self.xf)
        self.yc = int(self.yf)
        
        sound.set_volume(0)
    def Shift_Up(self):
        i=0
