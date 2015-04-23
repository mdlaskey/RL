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
        self.xc = start[0]
        self.yc = start[1]
        self.xf = float(self.xc)
        self.yf = float(self.yc)
        self.speed = 0
        self.gear = 3
        self.wobble = 0
        self.lap = 0
        self.cords = np.array([self.xf,self.yf])
        self.pre_cords = self.cords
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
        screen.blit(self.images[view],(x-32,y-32))
        #screen.blit(gears[self.gear],(xt,yt))
        indicated = int(10.0*self.speed)
        #screen.blit(speedo[indicated],(xt+100,yt))
        #screen.blit(laps[self.lap],(xt,yt+50))
        if self.lap > MAX_LAPS :
            elapsed_time = font.render(str(frames/24),1,(250,250,250))
            #screen.blit(elapsed_time,(xt+100,yt+50))



    def getState(self,track,dummycars):
        state = np.zeros(3) 
        state[0] = self.view 
        state[1:3] = self.cords 
        #state[1:3] = track.getDistance(self.xc,self.yc)
        dist = []
        #for d_car in dummycars:
        #    state = np.hstack((state,d_car.cords))
        #    d = LA.norm(d_car.cords -self.cords)
        #    p_d = LA.norm(d_car.pre_cords - self.pre_cords)
        #   if(d <= p_d):
        #       dist.append(d)


        #dist.sort()
        #state[3:7] = track.getCorners(self.xc,self.yc)
        return state

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
