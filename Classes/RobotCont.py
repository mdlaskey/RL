import numpy as np
import numpy.linalg as la
import numdifftools as nd
import math
import IPython
from Classes.vehicle import Vehicle 
from Tools.lqr import LQR 
from sklearn import linear_model
from Tools.MotionPlanning.car_model import OptimizeTraj
from sklearn.kernel_ridge import KernelRidge
import matplotlib.pyplot as plt
"""
Continous dynamics car, contains cost function for elipse track
"""

class RobotCont(): 
	T = 6
	dt = 1
	gamma = 0.05



	def calQ(self,Costs,States,Controls):

		Q = []
		X_U = []
		for i in range(len(Costs)):
			costs = Costs[i]
			states = States[i]
			controls = Controls[i]

			prev_sum = 0.0
			for t in range(states.shape[0]): 
				exp_r = 0.0
				for i in range(t,states.shape[0]):
					exp_r += costs[i]*self.gamma**(i-t)
				Q.append(exp_r)
				x_u = np.zeros(8)
				x_u[0:4] = states[t]
				x_u[4:6] = controls[t]
				X_U.append(x_u)

		
		self.Q = KernelRidge(alpha = 0.01, kernel = 'poly',degree = 2)
		
		self.Q.fit(X_U,Q)

	def evalQ(self,state,control): 
		x_u = np.zeros(8)
		x_u[0:4] = state
		x_u[4:6] = control 


		return self.Q.predict(x_u)


	def finit_dif(self,f, u_ref, my_eps=1e-3):
		"""
		Linearizes dynamics according to 
		given inputs using finite difference.
		Step size is given by my_eps.
		"""
		#IPython.embed()
		A = np.zeros([2])
		for i in range(2):
			u_dx = np.zeros(u_ref.shape[0])+u_ref
			u_dx[i] = u_dx[i] + my_eps
		
			z = (f(u_ref) - f(u_dx)) / my_eps
			A[i] = z
		return A
		

	def calGrad(self,state,control): 
		print "STATE",state
		print "CONTROl", control
		func = lambda u: self.evalQ(state,u)
		grad = nd.Gradient(func,step = 5e-1)
		#grad = self.finit_dif(func,control)
		
		return 1e5*grad(control)[0]


