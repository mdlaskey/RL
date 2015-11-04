__author__ = 'wesley'
import os
import pygame
import IPython
import dummy_car
from pygame.locals import *
import math
import pdb
import learner
import copy

pygame.init()
import car
import pickle
import numpy as np
import track_elipse as track
import time
import matplotlib as plt
import itertools

from Classes.SteeringWheel import SteeringWheel

from Classes.Supervisor import Supervisor 
from Agents.DAgger import Dagger
from Agents.Soteria import Soteria

from scipy.stats import norm

from numpy import linalg as LA

class RaceGame:
    def __init__(self,samples = 10,human = True):

        self.x = 8
        self.y = 30
        os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" %(self.x,self.y)
        self.screen_size = (400,400)
        self.cost = []
        self.i = 0
        self.human = True
        offset = np.array([350,300])
        self.str_wheel = SteeringWheel(offset)
        #self.screen_size = (2000,1040)
      
        
        self.screen = pygame.display.set_mode(self.screen_size)#,pygame.FULLSCREEN)
        #pygame.display.toggle_fullscreen()
        self.screen.fill((0,192,0))
        self.track_f = pygame.image.load('track.png')
        self.visible_track = pygame.image.load('track_textured.png')
        self.trk = self.track_f.get_at((0,0))

        
        self.cost_oil = np.zeros(6)
        self.controls_oil = np.zeros([6,2])
        self.states_oil = np.zeros([6,4])
        self.probs_oil = np.zeros(6)


        self.car = car
        self.supervisor = Supervisor()
        self.t = 0

        theta, pos, self.car_state = self.supervisor.getInitialState()

        self.car_state[3] = 5.0
        self.car_state[2] = 2.0


        self.clock = pygame.time.Clock()
        self.running = True
        
        self.red = car.Sprite()
        self.font = pygame.font.Font(None,60)

        self.iterations = 0

        self.car.xs = self.screen_size[0]/2;
        self.car.ys = self.screen_size[1]/2;

        self.xs = 600
        self.ys = 450
        self.xt = 100
        self.yt = 20

        self.BLUE = (  0,   0, 255)


   
        print "Loading graphics"
        self.Track = track.Track()
       
        
        self.red.Load('car_images',360,pos,theta)
        car.Static_Sprite.initialize_images(self.red.NF, self.red.path)

        self.lap = 0

     
        self.frames = 0
        self.iters = 0
       

    def run_frame(self):
        # Update screen
        
        self.iters += 1
        self.frames += 1
        self.car.frames = self.frames

        

        self.red.Update(self.car_state)
        print self.car_state

        self.clock.tick(24)
        self.screen.fill((0,0,0))
        self.screen.blit(self.visible_track,(self.car.xs-self.red.xc,self.car.ys-self.red.yc))
        self.Track.Draw(self.screen,(self.red.xc-self.car.xs,self.red.yc-self.car.ys))
        self.red.Draw(car.xs,car.ys,self.screen)
        self.str_wheel.drawSteering(self.screen,self.car.xs,self.car.ys)
        self.state = pygame.surfarray.array3d(self.screen)
        pygame.display.flip()



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
   

    def oil_dynamics(self,control):
        control[1] = control[1]-0.08
        control[0] = control[0]+1e-1
        return control

    def probs(self,std,x): 
        return np.exp((-x**2/(2*std**2))/(std*np.sqrt(2*math.pi)))

    def get_control(self): 

        angle = self.str_wheel.getSteeringAngle()

        control = np.zeros(2)
        control[1] = angle 


        key = pygame.key.get_pressed()

        if(key[K_w]):
            control[0] = 1.0
        elif(key[K_s]):
            control[0] = -1.0
       
    


        self.car_state = self.supervisor.car.dynamics(self.car_state,control)



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

      
           
        

  
