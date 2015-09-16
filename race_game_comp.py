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
import track_comp as track
import time
import matplotlib as plt
import itertools

from Agents.DAgger import Dagger
from Agents.Soteria import Soteria

from numpy import linalg as LA

class RaceGame:
    def __init__(self,agent = None, MAX_LAPS=100, graphics=False, input_red=None, input_track=None, turn_angle=5, initial_training=True,seed=1):

        self.graphics = graphics
        self.turn_angle = turn_angle
        self.agent = agent
        self.initial_training = initial_training
        self.cost = []
        self.queries = []
        self.sup_incr = []
        self.unc_incr = []
        self.qur_incr = []
        self.laps =0 
        track.Track.seed = seed
        self.x = 8
        self.y = 30
        os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" %(self.x,self.y)
        self.screen_size = (400,400)
        #self.screen_size = (2000,1040)
        if self.graphics:
            self.agent = agent
            self.screen = pygame.display.set_mode(self.screen_size)#,pygame.FULLSCREEN)
            #pygame.display.toggle_fullscreen()
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

        

        self.cars_hit = []
        self.timeOffTrack = []
        self.timeHit = []
        self.dummy_cars = []

        self.past_action = 0

        if input_red:
            self.red = input_red
        if input_track:
            self.Track = input_track

        if self.graphics:
            print "Loading graphics"
            self.Track = track.Track()
            self.Track.Load()
            theta, pos = self.Track.returnStart()
            self.red.Load('car_images',360,pos,theta)
            car.Static_Sprite.initialize_images(self.red.NF, self.red.path)


            
            # print [d_car.id for d_car in self.dummy_cars]
            # print "Done"

        self.inbox = self.trap.collidepoint(self.red.xc,self.red.yc)
        self.lap = 0

        self.first_frame = True
        self.retrain_net = False
        self.robot_only = False
        self.frames = 0
        self.iters = 0
       

    def run_frame(self):
        # Update screen
        
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
            self.clock.tick(24)
            self.screen.fill((0,0,0))
            self.screen.blit(self.visible_track,(self.car.xs-self.red.xc,self.car.ys-self.red.yc))
            self.Track.Draw(self.screen,(self.red.xc-self.car.xs,self.red.yc-self.car.ys))
            self.red.Draw(car.xs,car.ys,self.screen)
            self.state = pygame.surfarray.array3d(self.screen)

        self.red.updateStats(self.Track,self.dummy_cars)

   


    def control_car_step(self, key_input=None):
        """
        Takes a control input and updates the environment.
        0 = "d", 1 = "a", 2 = others/none
        """
        # Control
        if key_input != None:
            key = {K_f:False, K_d:False, K_a:False}
            if key_input == 0:
                key[K_d] = True
            elif key_input == 1:
                key[K_a] = True
        else:
            key = pygame.key.get_pressed()

        #exit 
        escape = pygame.key.get_pressed()

        if(escape[K_f]):
            self.running = False

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

        if(self.iters>700 and not self.initial_training):
            self.cars_hit.append(self.red.carsHit)
            self.iterations += 1
            self.timeOffTrack.append(self.red.timeOffTrack)
            self.timeHit.append(self.red.timesHit)
            self.cost.append(self.red.timeOffTrack)
            self.queries.append(self.agent.human_input)
            self.sup_incr.append(self.agent.sup_incorrect)
            self.unc_incr.append(self.agent.unc_incorrect)
            self.qur_incr.append(self.agent.qur_incorrect)
            self.red.reset()
           
            if(self.iterations == self.MAX_LAPS):
                self.running = False
            self.agent.updateModel()

            self.iters = 0

        if (self.initial_training or not self.graphics):
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
            self.agent.integrateObservation(self.state,a)

        # if self.graphics and self.initial_training and self.iters > 700:
        #     self.timeHit.append(self.red.timesHit)
        #     self.cost.append(self.red.timeOffTrack)
        #     self.laps +=1
        #     self.queries.append(self.laps*700)
        #     self.red.reset()


        if self.graphics and self.iters == 1200:
            #self.running = False
            if self.initial_training:

                self.agent.newModel()
                self.initial_training = False
                self.agent.initialTraining = False
                self.iters = 0
                self.red.reset()

        if not self.Track.IsOnTrack(self.red):
            self.red.wobble = 0
        else:
            self.red.wobble = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
           
        #print "coordinates", self.red.xc, self.red.yc

    def control_car(self, input_sequence=None, driving_agent=False, step_size=1):
        """
        Controls car using given input sequence.
        Calculates input sequence if none is given and driving_agent is true.
        """
        #original_crashes = self.red.carsHit + self.red.timesHit + self.red.timeOffTrack
        if input_sequence:
            for action in input_sequence:
                self.run_frame()
                self.control_car_step(action)
        elif driving_agent:
            input_sequence = self.driving_agent(step_size=step_size)
            print "action", input_sequence[0], input_sequence
            for action in input_sequence[0:step_size]:
                self.past_action = action
                self.run_frame()
                self.control_car_step(action)
            # print ("self.red.xc", "self.red.yc"), (self.red.xc, self.red.yc)
            # print "self.red.view", self.red.view
            # print "desiredRectangle", self.Track.desired_rectangle_angle(self.red)
        else:
            self.run_frame()
            self.control_car_step()
        # new_crashes = self.red.carsHit + self.red.timesHit + self.red.timeOffTrack
        # if new_crashes > original_crashes and self.graphics:
        #     print "CRASHED", new_crashes

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


    def driving_agent(self, step_size=1, num_basic_steps=7, num_search_steps=3):
        """
        Determines whether to steer left or right
        based on local trajectory simulation.
        Returns a trajectory that avoids crashing.
        Prioritizes actions that straighten the car.

        Heuristic: Going straight > one turn, then go straight >
        local search trajectory

        If no crash-free solution is found, prioritizes turning
        if about to run off the track, otherwise returns last
        crash-free solution.

        step_size: Number of steps to use in basic trajectory.
        num_basic_steps: Number of no-action steps to extend basic trajectory with.
        num_search_steps: Number of steps to use in search if basic trajectories crash.
        """

        # Determines order of actions to try based on deviation from 90 degree multiples
        desired_angle = self.Track.desired_rectangle_angle(self.red)
        possible_actions = ['right', 'left', 'neutral']
        new_angles = [self.calculate_new_angle(self.red.view, possible_actions[i]) for i in range(3)]
        deviations = [min(abs(desired_angle + 360 - new_angles[i]), abs(desired_angle - new_angles[i])) for i in range(3)]
        basic_actions_sorted = sorted(range(3), key=lambda x: deviations[x])
        actions_sorted = basic_actions_sorted[:]

        # Favors turning toward center of track
        if basic_actions_sorted[0] == 2:
            new_positions = [self.red.get_new_pos(i) for i in new_angles[0:2]]
            new_distances = [self.Track.distance_from_current_rectangle(self.red, new_positions[i]) for i in range(2)]
            basic_actions_sorted[1:3] = sorted(range(2), key=lambda x: -new_distances[x])

        # Basic Trajectories
        basic_trajectories = [[i] + ([2] * num_basic_steps) for i in basic_actions_sorted]
        for input_sequence in basic_trajectories:
            if self.simulate_steps(input_sequence) == 0:
                return input_sequence

        # Search
        for i in range(2, num_search_steps + 1):
            trajectories = [list(j[::-1]) + ([2] * (num_search_steps - i)) for j in itertools.product(actions_sorted, repeat=i)]
            for input_sequence in trajectories:
                if self.simulate_steps(list(input_sequence)) == 0:
                    return [min(input_sequence)] + input_sequence

        #new_positions = [self.red.get_new_pos(i) for i in new_angles[0:3]]

        print "Hit"
        # Crash

        return [basic_actions_sorted[0]]


    def simulate_steps(self, input_sequence):
        """
        Simulates the game for "steps" number of steps from the current trajectory.
        Returns whether there is a collision or the car goes off the track.
        Cannot display graphics of the simulated game.
        """

        simulated_game = RaceGame(graphics=False, input_red=copy.deepcopy(self.red), input_track = copy.deepcopy(self.Track))
        simulated_game.red.timesHit = 0
        simulated_game.red.carsHit = 0
        simulated_game.red.pastId = 0

        simulated_game.control_car(input_sequence=input_sequence, driving_agent=False)

        return simulated_game.red.timesHit + simulated_game.red.carsHit

  
