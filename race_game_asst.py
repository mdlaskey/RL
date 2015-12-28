import os
import pygame
import IPython
import dummy_car
from pygame.locals import *
import math
import pdb
import learner
import copy

import car
import pickle
import numpy as np
import track_elipse as track
import time
#import matplotlib as plt
import itertools

from Classes.SteeringWheel import SteeringWheel
from Classes.OilDynamics import OilDynamics

from Classes.Supervisor import Supervisor 
#from Agents.DAgger import Dagger
#from Agents.Soteria import Soteria

from scipy.stats import norm

from numpy import linalg as LA
import cv2

OFFSET = np.array([2100,1125])

T = 500
#T = 10
class RaceGame:
    def __init__(self,samples = 10,human = True,coach = 'false',roboCoach = None,game='winter',timesteps=500):
        print "BEFORE PYGAME INIT"
        #pygame.init()
        #T = timesteps
        self.x = 8
        self.y = 30
        os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" %(self.x,self.y)
        self.screen_size = (400,400)
        self.roboCoach = roboCoach
        self.cost = []
        self.i = 0
        self.human = True
        offset = np.array([350,300])
        self.grad = np.zeros(2)
        self.str_wheel = SteeringWheel(offset)
        self.key_pressed = False

        #self.screen_size = (2000,1040)
        if(coach == 'false'):
            self.use_coach = False
        else: 
            self.use_coach = True
         
        self.screen = pygame.display.set_mode(self.screen_size)#,pygame.FULLSCREEN)
        #pygame.display.toggle_fullscreen()
        self.screen.fill((0,192,0))
        if(game == 'winter'):
            self.visible_track = pygame.image.load('track_textured.png')
            self.oil_on = True
        else:
            self.visible_track = pygame.image.load('track_textured_summer.png')
            self.oil_on = False
        #self.trk = self.track_f.get_at((0,0))

        
        self.cost = np.zeros(T)
        self.inOil = np.zeros(T)
        self.controls = np.zeros([T,2])
        self.states = np.zeros([T,4])
        


        self.car = car
        self.supervisor = Supervisor()

        self.t = 0
        self.oil = OilDynamics(self.supervisor.car.dynamics)
        theta, pos, self.car_state = self.supervisor.getInitialState()
        IPython.embed()
        self.car_state[3] = 5.0
        self.car_state[2] = 2.0


        self.clock = pygame.time.Clock()
        self.running = True
        
        self.red = car.Sprite()
        #self.font = pygame.font.Font(None,60)

        self.iterations = 0

        self.car.xs = self.screen_size[0]/2;
        self.car.ys = self.screen_size[1]/2;

        self.xs = 600
        self.ys = 450
        self.xt = 100
        self.yt = 20

        self.BLUE = (  0,   255, 127)


   
        print "Loading graphics"
        self.Track = track.Track()
       
        
        self.red.Load('car_images',360,pos,theta)
        car.Static_Sprite.initialize_images(self.red.NF, self.red.path)

        self.lap = 0

     
        self.frames = 0
        self.iters = 0
       

    def run_frame(self):
        # Update screen
        
        self.frames += 1
        self.car.frames = self.frames

        self.red.Update(self.car_state)
        print "ITERS", self.iters


        if(self.iters > T-2): 
            self.running = False

        self.clock.tick(24)
        self.screen.fill((0,0,0))

        self.screen.blit(self.visible_track,(self.car.xs-self.red.xc,self.car.ys-self.red.yc))
        self.Track.Draw(self.screen,(self.red.xc-self.car.xs,self.red.yc-self.car.ys))
        self.red.Draw(car.xs,car.ys,self.screen)
        self.str_wheel.drawSteering(self.screen,self.car.xs,self.car.ys)

        # if(self.roboCoach != None):
        #     self.str_wheel.drawCorrections(self.screen,self.car.xs,self.car.ys,self.grad)

        if(self.use_coach and self.oil.inOil(self.car_state)):
            self.str_wheel.drawCorrections(self.screen,self.car.xs,self.car.ys,self.grad)
            #self.str_wheel.optimalCor(self.screen,self.car.xs,self.car.ys,self.oil.help)

        pygame.display.flip()


    def drawGoal(self,screen,x,y):
        points = self.supervisor.listPointsRef()

        for i in range(len(points)): 
            p = points[i]
            points[i] = (p[0]-x+OFFSET[0],p[1]-y+OFFSET[1])
            
        pygame.draw.polygon(screen, self.BLUE, points, 3)

    def control_car(self):
        """
        Controls car using given input sequence.
        Calculates input sequence if none is given and driving_agent is true.
        """  
        self.run_frame()

        if(self.human):
            self.get_control()
        else: 
            self.control_car_step()

        #cv2.imwrite("static/images/game.jpg", pygame.surfarray.array3d(self.screen))

        return pygame.surfarray.array3d(self.screen)

   


    def probs(self,std,x): 
        return np.exp((-x**2/(2*std**2))/(std*np.sqrt(2*math.pi)))

    def get_control(self): 


        angle = self.str_wheel.getSteeringAngle()
        if(not self.str_wheel.start): 
            return

        control = np.zeros(2)
        control[1] = angle 


        key = pygame.key.get_pressed()

        if(key[K_w] and not self.key_pressed):
            control[0] = 1.0
            self.key_pressed = True
        elif(key[K_s] and not self.key_pressed):
            control[0] = -1.0
            self.key_pressed = True
        else: 
            self.key_pressed = False
       

        self.controls[self.iters,:] = control
        self.states[self.iters,:] = self.car_state

        if(self.roboCoach != None): 
            self.grad = self.roboCoach.calGrad(self.car_state,control)
            print "GRAD ", self.grad
            Q = self.roboCoach.evalQ(self.car_state,control)
            print "Q VALUE, ", Q


        #self.car_state = self.supervisor.car.dynamics(self.car_state,)
        self.car_state = self.oil.dynamics(self.car_state,control,self.oil_on)

        #Update Stats
        self.cost[self.iters] =  self.supervisor.getCost(self.car_state)
        self.inOil[self.iters] = self.oil.inOil(self.car_state)
        self.iters += 1

        
       


    def control_car_step(self, key_input=None):
        """
        Takes a control input and updates the environment.
        0 = "d", 1 = "a", 2 = others/none
        """
        # Control
        control = self.supervisor.getControl(self.car_state,self.t)
        self.t = self.t+1 
        self.cost.append(self.supervisor.getCost(self.car_state,self.t))
        if(self.t == self.supervisor.T-1):
            self.running = False 
            self.t = 0
        #print self.car_state
        
        print "TIMESTEP ",self.t
        if(self.t > 17 and self.t < 24): 
            control = self.oil_dynamics(control)
            s_v = 0.03*np.random.randn()
            s_a = 0.03*np.random.randn()
            control[1] = control[1] + s_v
            control[0] = control[0] + s_a

            self.controls_oil[self.i,:] = control
            self.states_oil[self.i,:] = self.car_state
            self.probs_oil[self.i] = self.probs(0.05,s_v)*self.probs(0.05,s_a)



        #self.car_state = self.supervisor.getPos(self.t)
        time.sleep(8e-2)
        self.car_state = self.supervisor.car.dynamics(self.car_state,control)

        if(self.t > 17 and self.t < 24):
            self.cost_oil[self.i] = self.supervisor.getCost(self.car_state,self.t+1)
            self.i+=1

      
           
        

  
