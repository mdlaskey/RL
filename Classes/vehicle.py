import numpy as np
import numpy.linalg as la
import IPython

"""
Continous dynamics car, contains cost function for elipse track
"""


class Vehicle(): 

	def dynamics(self,x, u, dt=1):
		"""
		Computes the dynamics of the car for the state x, which is [x,y, theta, velocity]
		Control input is acceleration and steering angle 

		"""
		W = 20
	
		a_n = 0.0
		ang_n = 0.0 
		A = np.array([dt*x[3]*np.cos(x[2]),
		             dt*x[3]*np.sin(x[2]),
		             dt*x[3]*np.tan(u[1]+ang_n)/W,
		             dt*u[0]])
		
		return x+A

	