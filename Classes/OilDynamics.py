import IPython
import dummy_car
import math
import pdb
import learner
import copy

import car
import pickle
import numpy as np
import track_elipse as track
import time
import matplotlib as plt
import itertools

from Classes.Supervisor import Supervisor 

from numpy import linalg as LA

BLACK = (0,0,0)
RED = (255,0,0)
GREEN = (34,139,34)

class OilDynamics:

	interval = 40
	def __init__(self,car_dyn):
		self.car_dyn = car_dyn  
		self.t = self.interval
		self.low = True

		self.low_x = -900.0
		self.high_x = -400.0 

		self.low_y = 200.0
		self.high_y = 500.0
		self.help = 0.0
		self.ref_speed = 7.0



	def inOil(self,state):

		if(state[0] > self.low_x and state[0] < self.high_x):
			if(state[1] > self.low_y and state[1] < self.high_y):
				return True

		return False

	def pumpedBrakes(self,state):

		if(self.t < self.interval): 
			self.t += 1 
		else: 
			self.t = 0
			if(self.low): 
				self.ref += 1 
				self.low = False
			else:
				self.ref -= 1
				self.low = True

		self.help = -(self.ref-state[3])
		print "REFERNCE ", self.ref 
		print "STATE ", state[3]
		return np.abs(self.ref - state[3])

	def dynamics(self,state,control,turn_on):
		if(not self.inOil(state) or not turn_on):
			self.ref = self.ref_speed
			self.help = 0
			self.out = True
		else: 
			if(self.out):
				self.out = False
				state[3] = self.ref_speed

			val = self.pumpedBrakes(state)
			if(val > 0.0):
				print "VALUE ",val
				control[1] = control[1]+.25#np.random.normal(val*0.1,val*0.01)
			

		state = self.car_dyn(state,control)
		if(self.inOil(state) and turn_on):
			if(state[3] > self.ref_speed+1): 
				state[3] = self.ref_speed+1
			elif(state[3] < self.ref_speed-1): 
				state[3] = self.ref_speed-1

		# if(not self.inOil(state)):
		# 	# if(state[3] > 6): 
		# 	# 	state[3] = 6
		if(state[3] < 4): 
			state[3] = 4

		return state


