import numpy as np
import numpy.linalg as la
import numdifftools as nd
import math
import IPython
import GPy
from Classes.vehicle import Vehicle 

from sklearn import preprocessing  
from Tools.lqr import LQR 
from sklearn import linear_model
from sklearn.kernel_ridge import KernelRidge
import numpy as np
from sklearn.gaussian_process import GaussianProcess

"""
Continous dynamics car, contains cost function for elipse track
"""

class RobotCont(): 
	T = 6
	dt = 1
	gamma = 0.05
	steps = 2

	def calQ(self,Costs,States,Controls):

		Qvals = []
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
				Qvals.append(exp_r)
				print 'Q ', exp_r
				x_u = np.zeros(6)
				x_u[0:4] = states[t]
				x_u[4:6] = controls[t]
				print 'X ', x_u
				X_U.append(x_u)

		X_U = np.asarray(X_U)
		print "XU SHAPE ", X_U.shape
		Qvals = np.asarray(Qvals)
		Qvals.shape += (1,)
		self.scaler_s = preprocessing.StandardScaler().fit(X_U)
		X_U = self.scaler_s.transform(X_U)
		
		IPython.embed()
		kernel = GPy.kern.Poly(input_dim=6, variance=1., order=3)
		self.Q = GPy.models.GPRegression(X_U,Qvals,kernel)
		self.Q.optimize()
		#self.Q = KernelRidge(alpha=1.0,kernel = 'poly',degree = 3)
		
		#self.Q.fit(X_U,Qvals)
		#IPython.embed()

	def updateQ(self,costs,states,controls):

		Qvals = []
		X_U = []
	
		prev_sum = 0.0
		for t in range(states.shape[0]): 
			exp_r = 0.0
			for i in range(t,states.shape[0]):
				exp_r += costs[i]*self.gamma**(i-t)
			Qvals.append(exp_r)
			print 'Q ', exp_r
			x_u = np.zeros(6)
			x_u[0:4] = states[t,:]
			x_u[4:6] = controls[t,:]
			print 'X ', x_u
			X_U.append(x_u)

		X_U = np.asarray(X_U)
		print "XU SHAPE ", X_U.shape
		Qvals = np.asarray(Qvals)
		Qvals.shape += (1,)
		self.scaler_s = preprocessing.StandardScaler().fit(X_U)
		X_U = self.scaler_s.transform(X_U)
		
		IPython.embed()
		kernel = GPy.kern.Poly(input_dim=6, variance=1., order=3)
		self.Q = GPy.models.GPRegression(X_U,Qvals,kernel)
		self.Q.optimize()
		#self.Q = KernelRidge(alpha=1.0,kernel = 'poly',degree = 3)
		
		#self.Q.fit(X_U,Qvals)
		#IPython.embed()

	
	def evalQVar(self,state,control):
		x_u = np.zeros([1,6])
		x_u[:,0:4] = state
		x_u[:,4:6] = control 
		x_u = self.scaler_s.transform(x_u)
		
		return self.Q.predict(x_u)[1][0,0]

	def evalQ(self,state,control): 
		x_u = np.zeros([1,6])
		x_u[:,0:4] = state
		x_u[:,4:6] = control 
		x_u = self.scaler_s.transform(x_u)

		return self.Q.predict(x_u)[0][0,0]


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
		func0 = lambda u: self.evalQ(state,u)
		func1 = lambda u: self.evalQVar(state,u)
		grad = nd.Gradient(func0,step = 1e-3)
		hess = nd.Hessian(func1,step=1e-3)
		
		grad = grad(control)
		hess = hess(control)
		grad = np.dot(hess,grad)
		#grad = np.dot(la.inv(hess+np.eye(2)*1e-5),grad)
		grad = grad/la.norm(grad)
		#grad = self.finit_dif(func,control)
		
		return grad

	def getControlGrad(self,state,control = np.zeros(2)): 
		grad = self.calGrad(state,control)
		alpha = 1
		v_old = self.evalQ(state,control)
		step = alpha*grad+control 
		val = self.evalQ(state,step)
	
		while(val < v_old):
			alpha = 0.9*alpha
			step = alpha*grad+control 
			val = self.evalQ(state,step)
		return step 

	def getControlOpt(self,state,control = None):
		if(control == None):
			c_old = np.zeros(2)
			control = self.getControlGrad(state)
		
		for i in range(self.steps):
			c_old = control
			control = self.getControlGrad(state,control)
			
		return control

	def getFeedback(self,state,control):

		control = self.getControlGrad(state,control)
		if(self.evalQVar(state,control) > self.epsilon):
			return np.sign(control[0])
		else: 
			return 0 

	def batchFeedBack(self):

		fdBack = []
		for x in range(1020,1400,20):
			for y in range(20,165,20):
				for v in range(4,6):
					state = np.array([x,y,0.0,v])
					control = self.getControlOpt(state).tolist()
					print control 
					fdBack.append([state.tolist(),control])

		return fdBack




