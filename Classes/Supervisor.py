import numpy as np
import numpy.linalg as la
import math
import IPython
from Classes.vehicle import Vehicle 
from Tools.lqr import LQR 
from Tools.MotionPlanning.car_model import OptimizeTraj

"""
Continous dynamics car, contains cost function for elipse track
"""

class Supervisor(): 
	T = 50.0 
	dt = 1


	def __init__(self):
		Q = np.eye(4)
		R = np.eye(2)*1e-2
		self.TOpt = OptimizeTraj()
		self.car = Vehicle()
		self.lqr = LQR()
		[ref_x,ref_u] = self.genElipse()
		self.elip = ref_x
		#[self.ref_x,self.ref_u] = self.TOpt.trajOpt(ref_x,ref_u) 
		[self.ref_x,self.ref_u] = self.TOpt.loadTraj()
		#self.testTraj()
		self.init_state =  np.array([-395.36474681,  499.29913709,    3.3113592 ,    5        ])
		#self.init_state = self.ref_x[:,0]
		[As,Bs] = self.linearizeTraj(self.ref_x,self.ref_u)
		self.Ks = self.lqr.ltv_lqr(As,Bs,Q,R,self.T)

		
		#self.testController()

	def testTraj(self): 
		out_states = np.zeros(self.ref_x.shape)
		out_states[:,0] = self.ref_x[:,0]

		plt.scatter(self.ref_x[0,:],self.ref_x[1,:])

		for i in range(1,int(self.T)): 
			out_states[:,i] = self.car.dynamics(out_states[:,i-1], self.ref_u[:,i-1])



		plt.scatter(out_states[0,:],out_states[1,:],color='r')
		self.ref_x = out_states
		plt.show()

	def getPos(self,t):
		return self.ref_x[:,t]
		
	def genElipse(self):
		a = 925
		b = 550
		T = 2000
		self.elipT = T
		ref_states = np.zeros([4,T])
		ref_controls = np.zeros([2,T-1])
		
		#Set velocity to angle/T 
		ref_states[3,:] = 1.12*math.pi*np.sqrt(a**2+b**2)/float(T)

		
		for i in range(T):
			frac = i/float(T) 
			ref_states[0,i] = a*np.cos(frac*2*math.pi)
			ref_states[1,i] = b*np.sin(frac*2*math.pi)

			tan = np.array([-a*np.sin(frac*2*math.pi),b*np.cos(frac*2*math.pi)])
			tan = tan/la.norm(tan)
			unit_x = np.array([0,1])
			cos_a = np.dot(tan,unit_x)

			ref_states[2,i] = np.arccos(cos_a)

		for i in range(0,int(T-1)):
			ref_controls[1,i] = np.abs(ref_states[2,i] - ref_states[2,i-1])

		return ref_states,ref_controls

	def getInitialState(self):
		pos = np.array([self.init_state[0],self.init_state[1]])
		theta = self.init_state[2] 
		return theta,pos,self.init_state

	def linearizeTraj(self,ref_x,ref_u): 

		A_mats = [] 
		B_mats = [] 

		for i in range(int(self.T-1)):
			
			[A,B,c] = self.lqr.linearize_dynamics(self.car.dynamics,ref_x[:,i], ref_u[:,i], self.dt)
			
			A_mats.append(A)
			B_mats.append(B)

		return A_mats,B_mats

	def listPointsRef(self): 
		points = []

		for i in range(self.elipT):
			point = (self.elip[0,i],self.elip[1,i])
			points.append(point)

		return points

	def getControl(self,state,t):
		
		return np.dot(self.Ks[int(self.T-t-3)],(state-self.ref_x[:,t]))+self.ref_u[:,t]
		#return np.dot(self.Ks[t],(state-self.ref_x[:,t]))+self.ref_u[:,t]

	def getCost(self,state): 
		mn_val = np.inf
		for i in range(self.elip.shape[1]):
			val = la.norm(self.elip[0:2,i]- state[0:2])
			if(val < mn_val):
				mn_val = val
		return mn_val
		

	def testController(self):
		out_states = np.zeros(self.ref_x.shape)
		out_states[:,0] = self.ref_x[:,0]

		for i in range(1,int(self.T)):
			u = self.getControl(out_states[:,i-1],i-1) 
			out_states[:,i] = self.car.dynamics(out_states[:,i-1], u)

		plt.scatter(out_states[0,:],out_states[1,:])
		plt.show()
		IPython.embed()

if __name__ == '__main__':

	sup = Supervisor()
	IPython.embed()
	