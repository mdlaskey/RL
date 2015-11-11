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

from Classes.Supervisor import Supervisor 
from Agents.DAgger import Dagger
from Agents.Soteria import Soteria

from scipy.stats import norm

from numpy import linalg as LA

BLACK = (0,0,0)
RED = (255,0,0)
GREEN = (34,139,34)

class SteeringWheel:

	def __init__(self,offset): 
		self.center = np.zeros(2)+offset
		self.radius = 50
		self.mouse = pygame.mouse
		self.st_ang = 0.0
		self.prev_st_ang = 0.0
		self.contained = False
		self.pre_gc = np.zeros(2)


	def getSteeringAngle(self): 
		for event in pygame.event.get():
			pass 
		
		if(self.mouse.get_pressed()[0]): 
			pos = self.mouse.get_pos()
		
			pos_np = np.array([pos[0],pos[1]])
			if(self.checkContains(pos)):
				vec = pos_np - self.center
				vec = vec/LA.norm(vec)
				self.prev_st_ang = self.st_ang
				self.st_ang = np.arctan2(vec[1],vec[0])
				#self.st_ang = (self.st_ang+math.pi)

		return self.st_ang - self.prev_st_ang


	def checkContains(self,pos):

		dist = LA.norm(self.center-pos)
	
		if(dist < self.radius or self.contained): 
			self.contained = True
			return True
		else: 
			return False 

	def drawCorrections(self,surface,x,y,g_c):
		#alpha = 1.0
		#g_c = alpha*g_c + (1-alpha)*self.pre_gc
		#self.pre_gc = g_c

		#Draw Steering Angle Correction
		if(np.abs(g_c[1]) > 1e-8):
			g_c[1] = math.pi/5*np.sign(g_c[1])

		rot_r = np.array([[np.cos(g_c[1]),-np.sin(g_c[1])],
			[np.sin(g_c[1]), np.cos(g_c[1])]])
		rot = np.array([[np.cos(self.st_ang),-np.sin(self.st_ang)],
			[np.sin(self.st_ang), np.cos(self.st_ang)]])

		unit = np.array([1,0])

		cor_pos = np.dot(rot_r,np.dot(rot,unit))*self.radius+self.center

		cor = (int(cor_pos[0] - x)+200, int(cor_pos[1] - y)+200)

		pointls = (self.cent,cor,self.end)

		pygame.draw.polygon(surface,GREEN,pointls)

		#Draw Speed Correction 

		



		

	def drawSteering(self,surface,x,y): 

		#draw steering angle
		self.cent = (int(self.center[0] - x)+200, int(self.center[1] - y)+200)
		pygame.draw.circle(surface,BLACK,self.cent,int(self.radius))

		#draw current steering angle 

		#get rot matrix from steering angle 
		rot = np.array([[np.cos(self.st_ang),-np.sin(self.st_ang)],
			[np.sin(self.st_ang), np.cos(self.st_ang)]])

		unit = np.array([1,0])

		end_pos = np.dot(rot,unit)*self.radius+self.center

		self.end = (int(end_pos[0] - x)+200, int(end_pos[1] - y)+200)

		pygame.draw.line(surface,RED,self.cent,self.end)




