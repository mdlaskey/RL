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



#import movie

screen_size = (500,500)

#screen_size = (600,600)
screen = pygame.display.set_mode(screen_size)
MAX_LAPS = 1

car.MAX_LAPS = MAX_LAPS
screen.fill((0,192,0))
clock = pygame.time.Clock()
running = True
red = car.Sprite()
blue = dummy_car.Sprite()
font = pygame.font.Font(None,60)


car.xs = screen_size[0]/2;
car.ys = screen_size[1]/2;
track_f = pygame.image.load('track.png')
xs = 600
ys = 450

xt = 100

yt = 20


visible_track = pygame.image.load('track_textured.png')
trap = pygame.Rect(844,1324,140,200)
trk = track_f.get_at((0,0))
Track = track.Track()
Track.Load()
red.Load('red',360,Track.returnStart())

car_list = Track.genCars(4*5)

dummy_cars = []
for car_p in car_list:
    d_car = dummy_car.Sprite()
    d_car.Load('purple',360,car_p[0],car_p[1])
    dummy_cars.append(d_car)


inbox = trap.collidepoint(red.xc,red.yc)
lap = 0
#pdb.set_trace()

first_frame = True 
intial_training = True
robot_only = False


frames = 0
robot = learner.Learner()
if(not intial_training):
    robot.Load()



while running:
    clock.tick(24)
    frames = frames + 1
    car.frames = frames
    screen.fill((0,192,0))
    past_lap = Track.getLap(red.xc,red.yc)
    red.Update()
    print red.xc,red.yc
    
    if trap.collidepoint(red.xc,red.yc) == 0:
        if inbox == 1 :
            red.lap += 1
            inbox = 0
    else :
        inbox = 1

    screen.blit(visible_track,(car.xs-red.xc,car.ys-red.yc))
    Track.Draw(screen,(red.xc-car.xs,red.yc-car.ys))
    
    for d_car in dummy_cars:
        d_car.Update(Track,screen)
        d_car.Draw((red.xc-car.xs),(red.yc-car.ys),screen)

    red.Draw(car.xs,car.ys,screen)


    state = red.getState(Track,dummy_cars)
    print "STATE",state

        
    if(not intial_training):
        ask_for_help = robot.askForHelp(state)

    if((intial_training or ask_for_help == -1) and not robot_only):

        text = font.render("Human Control",1,(255,0,0))
        screen.blit(text,(xt-20,yt))
        key = pygame.key.get_pressed()
        if key[K_d] :
            print "key d pressed"
            a = np.array([0])
            red.view = (red.view+2)%360
        
        elif key[K_a]:
            a = np.array([1])
            red.view = (red.view+358)%360
        else:
            a = np.array([2])    
    else: 
        a = robot.getAction(state)

        text = font.render("Robot Control",1,(0,0,255))
        screen.blit(text,(xt,yt))

        if a[0] == 0 :
            a = np.array([0])
            red.view = (red.view+2)%360
        
        elif a[0] == 1:
            a = np.array([1])
            red.view = (red.view+358)%360
        else:
            a = np.array([2])

    pygame.display.flip()

    if((intial_training or ask_for_help == 1 or first_frame) and not robot_only):
        if(first_frame):
            img = pygame.surfarray.array3d(screen)
            States = []
            States.append(img)
            Actions = np.array([a[0]])
            first_frame = False 
        else:
            img = pygame.surfarray.array3d(screen)

            States.append(img)

            Actions = np.vstack((Actions,a))  
   
    if(Track.getLap(red.xc,red.yc)> past_lap and not intial_training and not robot_only):
        first_frame = True;
        robot.updateModel(States,Actions)

    if Track.getLap(red.xc,red.yc) > MAX_LAPS :
        if(intial_training):
            #robot.States = robot.listToMat(States)
            robot.States = np.array(States)
            robot.Actions = np.array(Actions) 
            robot.trainModel(robot.States,robot.Actions)
            intial_training = False

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

