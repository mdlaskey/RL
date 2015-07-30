import pygame
import math
import random
import IPython
import track
import numpy as np

xs = 600
ys = 450
xt = xs - 100
yt = ys + 100
dt = 1.0

speedo = []
laps = []

pygame.mixer.init()
sound = pygame.mixer.Sound("sound/racing_car.wav")
sound.set_volume(0.0)
sound.play(loops=-1)
shiftup_sound = pygame.mixer.Sound("sound/shift_up.wav")
idle_sound = pygame.mixer.Sound("sound/idle_rev.wav")
idle_sound.set_volume(0)
idle_sound.play(loops=-1)
WIDTH = 14
HEIGHT = 6 


ANGLES = [0,89,179,269]

ANGLES_IDX = [89,179,269,0]

class Static_Sprite():
    @staticmethod
    def initialize_images(path):
        Static_Sprite.images = []
        for a in ANGLES_IDX:
            name = path+'/computer_ '
            Static_Sprite.images += [pygame.image.load(name+str(a)+'.png')]

    @staticmethod
    def draw_car(input_car, s_c, s_y, screen):
        view = input_car.view + int(random.gauss(0,input_car.wobble))

        if view < 0 :
            view = view + 360
        view = view%360
        view = int(view/90)
        input_car.x_s = input_car.xc-s_c-32
        input_car.y_s = input_car.yc-s_y-32
        #print input_car.x_s, input_car.y_s
        Size = screen.get_bounding_rect()
        if(input_car.x_s>0 and input_car.y_s>0 and input_car.x_s<=Size.width and input_car.y_s<=Size.height):
            screen.blit(Static_Sprite.images[view],(input_car.x_s,input_car.y_s))

        indicated = int(10.0*input_car.speed)

class Sprite():
    def return_dir(self,track):
        return track.currentRectangle(self.xc,self.yc)

    def Load(self,path,NF,xs,ys):
        self.view = 270
        self.NF = NF
#       self.xc = 1690
#        self.yc = 2400

        self.xc = xs
        self.yc = ys
        self.id = xs 
        self.xf = float(self.xc)
        self.yf = float(self.yc)
        self.startXf = self.xf
        self.startYf = self.yf
        self.x_s = 0
        self.y_s =0
        self.speed = 8
        self.gear = 1
        self.wobble = 0
        self.lap = 0
        self.cords = np.array([self.xf,self.yf])
        self.path = path


    def Draw(self,s_c,s_y,screen):
        Static_Sprite.draw_car(self, s_c, s_y, screen)
      
    def Update(self,track):
        
        theta = self.return_dir(track)

        if self.wobble :
            idle_sound.set_volume(1.)
        else :
            idle_sound.set_volume(0)
        vx = self.speed*math.cos(theta)
        vy = self.speed*math.sin(theta)
     
        self.view = int(theta*180/math.pi)-90
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
