import math
import random
import numpy as np
import IPython
import cPickle as pickle 
from numpy import linalg as LA
from sklearn.kernel_ridge import KernelRidge
from sklearn import svm 
from sklearn import preprocessing  
from sklearn import linear_model
from sklearn import metrics 
from scipy.sparse import csr_matrix
from scipy.sparse import vstack
from scipy.spatial.distance import pdist,squareform
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import linear_kernel
from sklearn.cross_validation import train_test_split
from cvxopt import matrix, solvers
from AHQP import AHQP 
from sklearn import svm
import matplotlib.pyplot as plt

import IPython


def getScore(ahqp_g,ahqp_b,state,label):

	incorrect_p = 0.0
	incorrect_n = 0.0  
	incorrect = 0.0 
	for i in range(state.shape[0]):
		if(ahqp_b.predict(state[i,:]) == 1.0):
			val =  -1.0
		else: 
		    val = ahqp_g.predict(state[i,:])

		if(label[i] != val and label[i] == -1.0):
			incorrect_n += 1.0
			incorrect += 1.0

		if(label[i] != val and label[i] == 1.0):
			incorrect_p += 1.0
			incorrect += 1.0
	
	print "N: ",incorrect_n,"P: ",incorrect_p

	return incorrect/state.shape[0]


THRESH = 0.6
data =  np.load('test.p')


states = data[0][1:1800,:]
actions = data[1][1:1800,:]



states_test = data[0][1801:2700,:]
actions_test = data[1][1801:2700,:]

clf = KernelRidge(alpha=1.0)

clf.fit(states,actions)

actions_pred = clf.predict(states)
bad_state = np.zeros(actions_pred.shape[0])
for i in range(actions_pred.shape[0]):
    print LA.norm(actions_pred[i,:] - actions[i,:])
    if(LA.norm(actions_pred[i,:] - actions[i,:])> THRESH):
        bad_state[i] = 1




scaler = preprocessing.StandardScaler().fit(states)
states_proc = scaler.transform(states)

good_labels = bad_state == 0.0         
states_g = states_proc[good_labels,:] 

bad_labels = bad_state == 1.0 
states_b = states_proc[bad_labels,:] 

#Get labels on training
labels = np.zeros(actions_test.shape[0])
prediction = clf.predict(states_test)
for i in range(prediction.shape[0]):

	if(LA.norm(actions_test[i,:] - prediction[i,:]) < THRESH):
		labels[i] = 1
	else: 
		labels[i] = -1


sigmas = [5.0]
nus = [1e-1]

# sigmas = [0.1]
# nus = [1e-3]

score_mat = np.zeros((len(nus),len(sigmas)))

for i in range(len(nus)):
	for j in range(len(sigmas)):

		ahqp_solver_g = AHQP(sigmas[i],nus[j])
		ahqp_solver_b = AHQP(1e-2,nus[j])


		ahqp_solver_g.assembleKernel(states_g, np.zeros(states_g.shape[0])+1.0)
		ahqp_solver_b.assembleKernel(states_b, np.zeros(states_b.shape[0])+1.0)

		ahqp_solver_g.solveQP()
		ahqp_solver_b.solveQP()



		score_mat[i,j] = getScore(ahqp_solver_g,ahqp_solver_b,scaler.transform(states_test),labels)
		

print score_mat
IPython.embed()