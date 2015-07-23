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
import track
import time
import matplotlib as plt

from Agents.DAgger import Dagger
from Agents.Soteria import Soteria

class RaceGame:
    def __init__(self, MAX_LAPS=100, graphics=False, input_red=None, input_dummy_cars=None):
        self.graphics = graphics
        self.x = 8
        self.y = 30
        os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" %(self.x,self.y)
        self.screen_size = (400,400)
        if self.graphics:
            self.screen = pygame.display.set_mode(self.screen_size)
            self.screen.fill((0,192,0))
            self.track_f = pygame.image.load('track.png')
            self.visible_track = pygame.image.load('track_textured.png')
            self.trk = self.track_f.get_at((0,0))

        self.MAX_LAPS = MAX_LAPS

        self.car = car
        self.car.MAX_LAPS = self.MAX_LAPS

        self.clock = pygame.time.Clock()
        self.running = True
        self.red = car.Sprite()
        self.blue = dummy_car.Sprite()
        self.font = pygame.font.Font(None,60)

        self.iterations = 0

        self.car.xs = self.screen_size[0]/2;
        self.car.ys = self.screen_size[1]/2;

        self.xs = 600
        self.ys = 450
        self.xt = 100
        self.yt = 20

        self.BLUE = (  0,   0, 255)

        self.trap = pygame.Rect(844,1324,140,200)

        self.Track = track.Track()
        self.Track.Load()
        self.red.Load('car_images',360,self.Track.returnStart())

        self.car_list = self.Track.genCars(6*5)

        self.cars_hit = []
        self.timeOffTrack = []
        self.timeHit = []
        self.dummy_cars = []
        for car_p in self.car_list:
            self.d_car = dummy_car.Sprite()
            self.d_car.Load('car_images',360,car_p[0],car_p[1])
            self.dummy_cars.append(self.d_car)

        self.inbox = self.trap.collidepoint(self.red.xc,self.red.yc)
        self.lap = 0

        self.first_frame = True
        self.intial_training = True
        self.retrain_net = False
        self.robot_only = False

        self.agent = Dagger(self.intial_training)

        self.frames = 0
        self.iters = 0
        self.robot = learner.Learner()
        if not self.intial_training:
            self.robot.Load(retrain_net=self.retrain_net)

        if input_red:
            self.red = input_red
        if input_dummy_cars:
            self.dummy_cars = input_dummy_cars

    def run_frame(self):
        # Update screen
        self.clock.tick(24)
        self.iters += 1
        self.frames += 1
        self.car.frames = self.frames

        self.past_lap = self.Track.getLap(self.red.xc,self.red.yc)
        self.red.Update()

        if self.trap.collidepoint(self.red.xc,self.red.yc) == 0:
            if self.inbox == 1 :
                self.red.lap += 1
                self.inbox = 0
        else:
            self.inbox = 1

        if self.graphics:
            self.screen.fill((0,0,0))
            self.screen.blit(self.visible_track,(self.car.xs-self.red.xc,self.car.ys-self.red.yc))
            self.Track.Draw(self.screen,(self.red.xc-self.car.xs,self.red.yc-self.car.ys))
            self.red.Draw(car.xs,car.ys,self.screen)
            self.state = pygame.surfarray.array3d(self.screen)

        self.red.updateStats(self.Track,self.dummy_cars)

        for d_car in self.dummy_cars:
            if self.graphics:
                d_car.Update(self.Track,self.screen)
                d_car.Draw((self.red.xc-self.car.xs),(self.red.yc-self.car.ys),self.screen)
            else:
                d_car.Update(self.Track)

    def control_car(self, key_input=None, driving_agent=False):
        """
        Takes a control input and updates the environment.
        0 = "d", 1 = "a", 2 = others/none
        """
        if(not self.intial_training):
            ask_for_help = self.agent.askForHelp(self.state)

        # Control
        if key_input or driving_agent:
            if driving_agent:
                key_input = self.driving_agent()
                # print "driving_agent", key_input
            key = {K_f:False, K_d:False, K_a:False}
            if key_input == 0:
                key[K_d] = True
            elif key_input == 1:
                key[K_a] = True
        else:
            key = pygame.key.get_pressed()

        if(key[K_f]):
            time.sleep(20)
        if key[K_d]:
            a = np.array([0])
        elif key[K_a]:
            a = np.array([1])
        else:
            a = np.array([2])
        # if key_input and not driving_agent:
        #     print "a", a
        #     print "self.red.xc, self.red.yc", self.red.xc, self.red.yc
        #     print "self.red.timesHit", self.red.timesHit
        #     print "self.red.carsHit", self.red.carsHit

        # if(self.red.isCrashed(self.Track)):
        if not self.Track.IsOnTrack(self.red):
            self.red.timesHit += 1
            self.red.returnToTrack(self.Track)

        if(self.iters>700 and not self.intial_training):
            self.cars_hit.append(self.red.carsHit)
            self.iterations += 1
            self.timeOffTrack.append(self.red.timeOffTrack)
            self.timeHit.append(self.red.timesHit)
            self.red.reset(self.dummy_cars)
            self.agent.updateModel()
            self.iters = 0



        if (self.intial_training or ask_for_help == -1) and not self.robot_only:
            self.text = self.font.render("Human Control",1,(255,0,0))
            if key[K_d] :
                self.red.view = (self.red.view+2)%360
            elif key[K_a]:
                self.red.view = (self.red.view+358)%360
        else:
            a_r = self.agent.getAction(self.state)
            self.text = self.font.render("Robot Control",1,(0,0,255))
            if a_r[0] == 0 :
                self.red.view = (self.red.view+2)%360
            elif a_r[0] == 1:
                self.red.view = (self.red.view+358)%360

        if self.graphics:
            pygame.display.flip()
            self.agent.integrateObservation(self.state,a)

        if self.Track.getLap(self.red.xc,self.red.yc) > self.MAX_LAPS:
            if self.intial_training:
                self.agent.newModel()
                self.intial_training = False
                self.agent.intial_training = False
                self.iters = 0
                self.red.reset(self.dummy_cars)

        if not self.Track.IsOnTrack(self.red):
            self.red.wobble = 10
        else:
            self.red.wobble = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    self.running = False;
                elif event.key == K_UP:
                    self.red.gear = self.red.gear + 1
                    if self.red.gear<4 :
                        self.red.Shift_Up()
                    if self.red.gear>4 :
                        self.red.gear = 4
                elif event.key == K_DOWN:
                    self.red.gear = self.red.gear - 1
                    if self.red.gear < 0:
                        self.red.gear = 0

    def driving_agent(self, num_steps=20):
        """
        Determines whether to steer left or right
        based on a simulation where it goes straight.
        Returns an action.
        """
        if self.simulate_steps(num_steps, 2) == 0:
            return 2
        else:
            simulated_0 = self.simulate_steps(num_steps, 0)
            if simulated_0 == 0:
                return 0
            else:
                simulated_1 = self.simulate_steps(num_steps, 1)
                return 0 if simulated_0 <= simulated_1 else 1



    def simulate_steps(self, num_steps, initial_input):
        """
        Simulates the game for "steps" number of steps from the current trajectory.
        Returns whether there is a collision or the car goes off the track.
        Cannot display graphics of the simulated game.
        """
        simulated_game = RaceGame(graphics=False, input_red=copy.deepcopy(self.red), input_dummy_cars=copy.deepcopy(self.dummy_cars))
        simulated_game.run_frame()
        simulated_game.control_car(key_input=initial_input, driving_agent=False)
        for _ in range(num_steps):
            simulated_game.run_frame()
            simulated_game.control_car(key_input=2)
        return simulated_game.red.timesHit + simulated_game.red.carsHit