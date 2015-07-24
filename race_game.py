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
import itertools

from Agents.DAgger import Dagger
from Agents.Soteria import Soteria

class RaceGame:
    def __init__(self, MAX_LAPS=100, graphics=False, input_red=None, input_dummy_cars=None, turn_angle=15):
        self.graphics = graphics
        self.turn_angle = turn_angle

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
        if self.graphics:
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

        self.cars_hit = []
        self.timeOffTrack = []
        self.timeHit = []
        self.dummy_cars = []

        if input_red:
            self.red = input_red
        if input_dummy_cars:
            self.dummy_cars = input_dummy_cars

        if self.graphics:
            self.red.Load('car_images',360,self.Track.returnStart())
            self.car_list = self.Track.genCars(6*5)
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

        if self.graphics:
            self.agent = Dagger(self.intial_training)

        self.frames = 0
        self.iters = 0
        self.robot = learner.Learner()
        if not self.intial_training:
            self.robot.Load(retrain_net=self.retrain_net)

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

    def control_car(self, input_sequence=None, driving_agent=False):
        """
        Controls car using given input sequence.
        Calculates input sequence if none is given and driving_agent is true.
        """
        if input_sequence:
            for action in input_sequence:
                self.run_frame()
                self.control_car_step(action)
        elif driving_agent:
            input_sequence = self.driving_agent()
            #print "input_sequence", [i for i in input_sequence]
            #print "used_sequence", input_sequence[0:2]
            for action in input_sequence[0:1]:
                self.run_frame()
                self.control_car_step(action)
        else:
            self.run_frame()
            self.control_car_step()

    def control_car_step(self, key_input=None):
        """
        Takes a control input and updates the environment.
        0 = "d", 1 = "a", 2 = others/none
        """
        if(not self.intial_training):
            ask_for_help = self.agent.askForHelp(self.state)

        # Control
        if key_input != None:
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
                self.red.view = self.calculate_new_angle(self.red.view, 'right')
            elif key[K_a]:
                self.red.view = self.calculate_new_angle(self.red.view, 'left')
        else:
            a_r = self.agent.getAction(self.state)
            self.text = self.font.render("Robot Control",1,(0,0,255))
            if a_r[0] == 0 :
                self.red.view = self.calculate_new_angle(self.red.view, 'right')
            elif a_r[0] == 1:
                self.red.view = self.calculate_new_angle(self.red.view, 'left')

        if self.graphics:
            pygame.display.flip()
            #self.agent.integrateObservation(self.state,a)

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
        #print "coordinates", self.red.xc, self.red.yc
    def calculate_new_angle(self, original_angle, action):
        """
        Given an angle and action, calculates
        the new angle after taking that action.
        """
        if action == 'left':
            return (original_angle - self.turn_angle) % 360
        elif action == 'right':
            return (original_angle + self.turn_angle) % 360
        else:
            return original_angle

    def driving_agent(self, num_steps=5, num_basic_steps=10):
        """
        Determines whether to steer left or right
        based on local trajectory simulation.
        Returns a trajectory that avoids crashing.
        Prioritizes actions that straighten the car.

        Heuristic: Going straight > one turn, then go straight >
        local search trajectory

        Returns first trajectory if no crash-free solution is found.
        """

        # Determines order of actions to try based on deviation from 90 degree multiples
        possible_actions = ['right', 'left', 'neutral']
        new_angles = [self.calculate_new_angle(self.red.view, possible_actions[i]) for i in range(3)]
        deviations = [min(new_angles[i] % 90, abs((new_angles[i] % 90) - 90)) for i in range(3)]
        actions_sorted = sorted(range(3), key=lambda x: deviations[x])

        basic_trajectories = [[i] + ([2] * (num_basic_steps - 1)) for i in actions_sorted]
        trajectories = basic_trajectories + [i[::-1] for i in itertools.product([0,2,1], repeat=num_steps)]
        for input_sequence in trajectories:
            if self.simulate_steps(input_sequence) == 0:
                return input_sequence
        print "No solution found"
        return trajectories[0]

    def simulate_steps(self, input_sequence):
        """
        Simulates the game for "steps" number of steps from the current trajectory.
        Returns whether there is a collision or the car goes off the track.
        Cannot display graphics of the simulated game.
        """
        simulated_game = RaceGame(graphics=False, input_red=copy.deepcopy(self.red), input_dummy_cars=copy.deepcopy(self.dummy_cars))
        simulated_game.control_car(input_sequence=input_sequence, driving_agent=False)
        return simulated_game.red.timesHit + simulated_game.red.carsHit
