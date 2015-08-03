import pygame
import math
import random
import numpy as np
import IPython

from numpy import linalg as LA

xs = 600
ys = 450
xt = xs
yt = ys
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

class Static_Sprite():
    @staticmethod
    def initialize_images(NF, path):
        Static_Sprite.images = []
        for f in range(NF):
            name = path+'/human_ '
            Static_Sprite.images += [pygame.image.load(name+str(f)+'.png')]

    @staticmethod
    def draw_car(input_car, x, y, screen, frames, MAX_LAPS):
        view = input_car.view + int(random.gauss(0,input_car.wobble))

        if view < 0 :
            view = view + 360
        view = (view+270)%360

        input_car.car_pos = [x,y]
        input_car.screen = screen

        # screen.blit(Static_Sprite.images[view],(x-32,y-32))
        screen.blit(Static_Sprite.images[view],(x,y))
        indicated = int(10.0*input_car.speed)
        if input_car.lap > MAX_LAPS :
            elapsed_time = font.render(str(frames/24),1,(250,250,250))

class Sprite():
    def Load(self,path,NF,start, car_length=50, car_width=25):
        self.view = 270
        self.NF = NF
        self.start = start 
        self.start_v = self.view 
        self.xc = start[0]
        self.yc = start[1]
        self.xf = float(self.xc)
        self.yf = float(self.yc)
        self.speed = 0
        self.gear = 5 # Acceleration
        self.wobble = 0
        self.lap = 0
        self.pastId = 0
        self.cords = np.array([self.xf,self.yf])
        self.pre_cords = self.cords
        self.carsHit = 0
        self.timeOffTrack = 0
        self.timesHit = 0
        self.path = path
        self.car_length = car_length
        self.car_width = car_width

    def Draw(self,x,y,screen):
        Static_Sprite.draw_car(self, x, y, screen, frames, MAX_LAPS)

    def getRotate(self,d_cords):
        #convert to radians 
        theta = self.view/57.296
        rot_mat = np.matrix([[math.cos(theta),-math.sin(theta)],\
                             [math.sin(theta),math.cos(theta)]])     
        
        dif = np.array([d_cords - self.cords]) 
        dif = dif/LA.norm(dif)
        rot_d_car = dif.T
        rot_d_car_cord = rot_mat*dif.T

        perp = np.array([[1,0]])
        perp = rot_mat*perp.T

        theta = math.acos(perp.T*rot_d_car)

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

        self.carsHit = 0
        self.timesHit = 0
        self.timeOffTrack = 0

    def returnToTrack(self,track):
        view,pos = track.closestRectangle(self.cords)

        self.xf = pos[0]
        self.yf = pos[1]

        self.view = int(57.296*view+90)

    def isCrashed(self,track):
        mid_cords = track.mid_cords
        outer_radius = track.radius
        inner_radius = track.radius / 2
        dist = LA.norm(mid_cords -self.cords)

        if(dist > outer_radius or dist < inner_radius):
            return True
        else:
            return False


    def getState(self,track,dummycars):
        state = np.zeros(2) 
        state[0:2] = self.cords
        dist_list = []
        for d_car in dummycars:
            d = np.zeros(2)
            dist = LA.norm(d_car.cords -self.cords)
            theta,vec,perp = self.getRotate(d_car.cords)

            if(dist < DIST_THRESH and theta > 0):
                d[0] = dist 
                d[1] = theta 

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

        return state

    def collision_rectangle(self):
        if self.view == 0 or self.view == 180 or self.view == 360:
            return pygame.Rect(self.cords[0], self.cords[1], self.car_width, self.car_length)
        elif self.view == 90 or self.view == 270:
            return pygame.Rect(self.cords[0], self.cords[1], self.car_length, self.car_width)

    def collision_points(self):
        # Convert to radians
        theta = self.view / 57.296
        points = []
        signs = [[1,1],[1,-1],[-1,1],[-1,-1]]
        for s1, s2 in signs:
            tempX = self.car_width / 2 * s1
            tempY = self.car_length / 2 * s2
            rotatedX = tempX*math.cos(theta) - tempY*math.sin(theta) + self.xc
            rotatedY = tempX*math.sin(theta) + tempY*math.cos(theta) + self.yc
            points.append(np.array([rotatedX, rotatedY]))
        return points

    def updateStats(self,track,dummycars):
        colliding_cars = self.get_colliding_cars(track, dummycars)
        self.carsHit += len(colliding_cars)

        if(not track.IsOnTrack(self)):
            self.timeOffTrack +=1


    def get_colliding_cars(self, track, dummycars):
        colliding_cars = []
        if self.view % 90 == 0:
            car_bounds = self.collision_rectangle()
            for d_car in dummycars:
                dummy_car_bounds = d_car.collision_rectangle()
                if car_bounds.colliderect(dummy_car_bounds) and d_car.id != self.pastId:
                    colliding_cars.append(d_car)
                    self.pastId = d_car.id
        else:
            car_points = self.collision_points()
            for d_car in dummycars:
                dummy_car_bounds = d_car.collision_rectangle()
                for point in car_points:
                    if dummy_car_bounds.collidepoint(point) and d_car.id != self.pastId:
                        colliding_cars.append(d_car)
                        self.pastId = d_car.id
        return colliding_cars

    def Update(self):
        self.speed = .95*self.speed + .05*(2.5*self.gear)
        #print self.gear,'\t',int(10.0*self.speed),'\t',self.lap
        

        theta = self.view/57.296
        if self.wobble :
            idle_sound.set_volume(0.)
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

    def get_new_pos(self, new_view):
        speed = .95*self.speed + .05*(2.5*self.gear)
        theta = new_view/57.296
        vx = speed*math.sin(theta)
        vy = speed*math.cos(theta)
        xf = self.xf + vx*dt
        yf = self.yf + vy*dt
        cords = [xf, yf]
        return np.array(cords)

    def Shift_Up(self):
        i=0
