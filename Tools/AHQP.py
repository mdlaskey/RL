import math
import random
import numpy as np
import IPython
import cPickle as pickle 
from numpy import linalg as LA
from sklearn import svm 
from sklearn import preprocessing  
from sklearn import linear_model
from sklearn import metrics 
from scipy.sparse import csr_matrix
from scipy.sparse import vstack
from scipy.spatial.distance import pdist,squareform
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import linear_kernel
from cvxopt import matrix, solvers

import IPython


class AHQP():

	B = 1000
	nu_g = 0.92
	nu_b = 0.1

	def assembleKernel(self,data,labels):
		self.m = data.shape[0]
		self.data = data
		self.gamma = 1e-2
		self.labels = labels
		self.G = squareform(pdist(data))

		self.G = np.exp((self.G**2)*-self.gamma)
		labels = np.ravel(labels)
		W = np.diag(labels)
		
		self.K = np.dot(W.T,np.dot(self.G,W)) 
		self.K = self.K

	def assembleKernelSparse(self,data,labels):
		self.m = data.shape[0]

		gamma = 1.0/float(self.m)		

		self.K = euclidean_distances(data,data)
		
		self.K = np.exp((self.K**2)*-gamma)

		

	def solveQP(self,dim):
		P = matrix(self.K)
		q = matrix(np.zeros((self.m,1)))

		h = np.zeros((self.m*2,1))
		h[0:self.m] = 1.0/(self.nu_g*self.m)
		# h[0.85*self.m:self.m] = 1.0/(self.nu_b*self.m)

		G = np.eye(self.m)
		G = np.vstack((G,np.eye(self.m)*-1))
	
		A = np.zeros((1,self.m))+1
		b = np.zeros((1,1)) +1

		h = matrix(h)
		G = matrix(G)

		A = matrix(A)
		b = matrix(b)



		sol = solvers.qp(P,q,G,h,A,b)
		self.weights = np.array(sol['x'])
		# plt.figure(3)
		# plt.plot(weights)
		# plt.show()
		
		self.caculateRho()
		return self.weights
	def caculateRho(self):
		mw = 0.0
		for i in range(self.data.shape[0]):
			if(mw < self.weights[i] and self.labels[i,0] == 1.0):
				mw = self.weights[i]
				maxSup = i 

		IPython.embed()
		
		self.rho = np.sum(np.ravel(self.labels*self.weights) * self.G[:,maxSup])

	def predict(self,x):
		k = np.zeros((self.m,1))
	
		k =  euclidean_distances(self.data,x)

		k = np.exp((k**2)*-self.gamma)

		# k = self.G[:,self.G.shape[0]-1]
		# k = k[0:self.G.shape[0]-1]

		ans = np.sign(np.sum(np.ravel(self.labels*self.weights)*np.ravel(k)) - self.rho)

		if(ans == 0.0):
			return 1
		else: 
			return ans
	

	def rbf(self,x_0,x_1):
		ed = euclidean_distances(x_0,x_1)
		return np.exp(-ed**2*self.gamma)



