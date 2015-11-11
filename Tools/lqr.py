import numpy as np
import numpy.linalg as la
import IPython




class LQR(): 

	def lqr_infinite_horizon_solution(self,A, B, Q, R, tol=1e-4):
		"""
		Iteratively computes optimal LQR control,
		until difference between iterations is below
		tolerance threshold.
		"""
		P = 0
		K_current = np.zeros(B.shape)
		K_new = np.ones(B.shape) * 10000
		while la.norm(K_new, K_current) > tol:
			K = - la.inv(R + B.T * P * B) * (B.T) * P * A
			P = Q + K.T*R*K + (A + B*K).T * P * (A + B*K)
			K_current = K_new
			K_new = K
		return K, P


	def ltv_lqr(self,As,Bs,Q,R,T):

		Ks = []

		P = np.zeros(As[0].shape)
		T = int(T)
		for i in range(2,T):
			K = np.dot(- la.inv(R + np.dot(np.dot(Bs[T-i].T, P), Bs[T-i])), np.dot(np.dot(Bs[T-i].T, P), As[T-i]))
			P = Q + np.dot(np.dot(K.T,R),K) + np.dot(np.dot(As[T-i] + np.dot(Bs[T-i],K).T ,P) , (As[T-i] + np.dot(Bs[T-i],K)))
			Ks.append(K)

		return Ks



	def linearize_dynamics(self,f, x_ref, u_ref, dt, my_eps=1e-2):
		"""
		Linearizes dynamics according to 
		given inputs using finite difference.
		Step size is given by my_eps.
		"""
		#IPython.embed()
		A = np.zeros([x_ref.shape[0], x_ref.shape[0]])
		B = np.zeros([x_ref.shape[0], u_ref.shape[0]])
		for i in range(x_ref.shape[0]):
			x_dx = np.zeros(x_ref.shape[0])+x_ref
			x_dx[i] = x_dx[i] + my_eps
		
			z = (f(x_dx, u_ref, dt) - f(x_ref, u_ref, dt)) / my_eps
			A[:,i] = z
		for i in range(u_ref.shape[0]):
			u_dx = np.zeros(u_ref.shape[0])+u_ref
			u_dx[i] = u_dx[i] + my_eps
			z = (f(x_ref, u_dx, dt) - f(x_ref, u_ref, dt)) / my_eps
			B[:,i] = z
		c = 0
		return A, B, c