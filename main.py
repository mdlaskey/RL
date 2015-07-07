import os
import pygame
import IPython
import dummy_car
from pygame.locals import *
import math
import pdb
import learner 
x = 8
y = 30

os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" %(x,y)
pygame.init()
import car
import pickle 
import numpy as np
import track 
import time
import matplotlib as plt

from Agents.DAgger import Dagger 
from Agents.Soteria import Soteria
#import movie

screen_size = (500,500)

#screen_size = (600,600)
screen = pygame.display.set_mode(screen_size)

MAX_LAPS = 3

car.MAX_LAPS = MAX_LAPS
screen.fill((0,192,0))
clock = pygame.time.Clock()
running = True
red = car.Sprite()
blue = dummy_car.Sprite()
font = pygame.font.Font(None,60)

iterations = 0

car.xs = screen_size[0]/2;
car.ys = screen_size[1]/2;
track_f = pygame.image.load('track.png')
xs = 600
ys = 450

xt = 100

yt = 20

BLUE = (  0,   0, 255)
visible_track = pygame.image.load('track_textured.png')
trap = pygame.Rect(844,1324,140,200)
trk = track_f.get_at((0,0))
Track = track.Track()
Track.Load()
red.Load('red',360,Track.returnStart())

car_list = Track.genCars(5*5)

cars_hit = []
timeOffTrack = [] 
dummy_cars = []
for car_p in car_list:
    d_car = dummy_car.Sprite()
    d_car.Load('purple',360,car_p[0],car_p[1])
    dummy_cars.append(d_car)


inbox = trap.collidepoint(red.xc,red.yc)
lap = 0
#pdb.set_trace()

first_frame = True 
intial_training = False
robot_only = False

agent = Dagger(intial_training)

frames = 0
robot = learner.Learner()
if(not intial_training):
    robot.Load()



while running:
    clock.tick(24)
    frames = frames + 1
    car.frames = frames
    screen.fill((0,0,0))
    past_lap = Track.getLap(red.xc,red.yc)
    red.Update()
    #print red.xc,red.yc
    
    if trap.collidepoint(red.xc,red.yc) == 0:
        if inbox == 1 :
            red.lap += 1
            inbox = 0
    else :
        inbox = 1

    screen.blit(visible_track,(car.xs-red.xc,car.ys-red.yc))
    Track.Draw(screen,(red.xc-car.xs,red.yc-car.ys))
    pygame.draw.circle(screen,BLUE,(int(Track.mid_cords[0]),int(Track.mid_cords[1])),int(Track.radius),0)

    for d_car in dummy_cars:
        d_car.Update(Track,screen)
        d_car.Draw((red.xc-car.xs),(red.yc-car.ys),screen)

    red.Draw(car.xs,car.ys,screen)

    red.updateStats(Track,dummy_cars)
    state = pygame.surfarray.array3d(screen)
    #print "STATE",state

    
   


    if(not intial_training):
        ask_for_help = agent.askForHelp(state)

    key = pygame.key.get_pressed()


    if(len(agent.States)>500 and not intial_training):
        red.reset(dummy_cars)
        
        cars_hit.append(red.carsHit)
        iterations += 1
        timeOffTrack.append(red.timeOffTrack)
        
        agent.updateModel()
        agent.reset()

    if key[K_d] :
        a = np.array([0])
    elif key[K_a]:
        a = np.array([1])
    else:
        a = np.array([2])
    if(iterations == 10):
        IPython.embed()


    if((intial_training or ask_for_help == -1) and not robot_only):

        text = font.render("Human Control",1,(255,0,0))
        screen.blit(text,(xt-20,yt))
        
        if key[K_d] :
            #print "key d pressed"
            red.view = (red.view+2)%360
        
        elif key[K_a]:
            red.view = (red.view+358)%360
        else:
            a = np.array([2])    
    else: 
        a_r = agent.getAction(state)

        text = font.render("Robot Control",1,(0,0,255))
        screen.blit(text,(xt,yt))

        if a_r[0] == 0 :
            red.view = (red.view+2)%360
        
        elif a_r[0] == 1:
            red.view = (red.view+358)%360
        else:
            a = np.array([2])

    pygame.display.flip()

 
    agent.integrateObservation(state,a)  
   
    # if(red.isCrashed(Track)):
    #     red.reset()
    #     IPython.embed()
    #     agent.updateModel()

    if Track.getLap(red.xc,red.yc) > MAX_LAPS :
        if(intial_training):
            agent.newModel()
            intial_training = False 
            red.reset(dummy_cars)
            

    if not Track.IsOnTrack(red) :
        red.wobble = 10
    else: 
        red.wobble = 0 
  

   

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                running = False;
            elif event.key == K_UP:
                red.gear = red.gear + 1
                if red.gear<4 :
                    red.Shift_Up()
                if red.gear>4 :
                    red.gear = 4
            elif event.key == K_DOWN:
                red.gear = red.gear - 1
                if red.gear < 0:
                    red.gear = 0
#            elif event.key == K_RIGHT:
#                red.view = (red.view + 2)%360
#            elif event.key == K_LEFT:
#                red.view = (red.view + 358)%360

