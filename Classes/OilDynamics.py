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

	interval = 5
	def __init__(self,car_dyn):
		self.car_dyn = car_dyn  
		self.t = 0
		self.low = True

		self.low_x = -900.0
		self.high_x = -400.0 

		self.low_y = 200.0
		self.high_y = 500.0



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
			else:
				self.ref -= 1

		return np.abs(self.ref - state[3])

	def dynamics(self,state):
		if(not self.inOil(state)):
			self.ref = state[3]
			return state
		else: 
			val = self.pumpedBrakes(state)

			control = np.zeros(2)

			if(val> 0):
				control[1] = np.random.normal(2,val*1)
			else: 
				control[1] = 2

			state = self.car_dyn(state,control)
			return state

