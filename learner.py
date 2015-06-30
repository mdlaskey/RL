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


class Learner():

	verbose = True
	option_1 = False 
	gamma = 1e-3
	gamma_clf = 1e-3
	first_time = True 
	iter_  = 1

	def Load(self,gamma = 1e-3):
		self.States = pickle.load(open('states.p','rb'))
		self.Actions = pickle.load(open('actions.p','rb'))
		self.Weights = np.zeros(self.Actions.shape)+1
		self.gamma = gamma 
		self.trainModel(self.States,self.Actions)
		
	def clearModel(self):
		self.States = pickle.load(open('states.p','rb'))
		self.Actions = pickle.load(open('actions.p','rb')) 
		self.Weights = np.zeros(self.Actions.shape)+1



	def trainModel(self,States,Action):
		self.clf = svm.LinearSVC()
		self.novel = svm.OneClassSVM()
	
		print States.shape
		print Action.shape
	
		Action = np.ravel(Action)
		
		
		self.clf.class_weight = 'auto'

		self.novel.gamma = self.gamma

		self.clf.C = 1e-2
		

		self.clf.fit(States,Action)
		#SVM parameters computed via cross validation
	
		
		self.novel.nu = 1e-3
		self.novel.kernel = 'rbf'
		self.novel.verbose = False
		self.novel.shrinking = False
		self.novel.max_iter = 3000
		

		#self.novel.fit(self.supStates)
		print self.novel.gamma
		
		if(self.verbose):
			self.debugPolicy(States,Action)
	

	def getScoreNovel(self,States):
		num_samples = States.shape[0]
		avg = 0
		for i in range(num_samples):
			ans = self.novel.predict(States[i,:])
			if(ans == -1): 
				ans = 0
			avg = avg+ans/num_samples

		return avg

	def debugPolicy(self,States,Action):
		prediction = self.clf.predict(States)
		classes = dict()

		for i in range(self.getNumData()):
			if(Action[i] not in classes):
				value = np.zeros(3)
				classes.update({Action[i]:value})
			classes[Action[i]][0] += 1
			if(Action[i] != prediction[i]):
				classes[Action[i]][1] += 1

			classes[Action[i]][2] = classes[Action[i]][1]/classes[Action[i]][0] 
		for d in classes:
			print d, classes[d]

		self.precision = self.clf.score(States,Action)

	def getPrecision(self):
		return self.precision

 	def getAction(self,state):
 		state = csr_matrix(state)
		return self.clf.predict(state)

	def askForHelp(self,state):
		
		if(isinstance(state,csr_matrix)):
			state = state.todense()

		return 1 
		#return self.novel.predict(state)

	def getNumData(self): 
		return self.Actions.shape[0]

	def newModel(self,states,actions):
		states = csr_matrix(states)

		self.States = states
		self.supStates = states.todense() 
		self.Actions = actions
		self.Weights = np.zeros(actions.shape)+1
		self.trainModel(self.States,self.Actions)

	def updateModel(self,new_states,new_actions,weights):
		print "UPDATING MODEL"

		#self.States = new_states
		#self.Actions = new_actions
		new_states = csr_matrix(new_states)
		
		self.States = vstack((self.States,new_states))
		self.supStates = np.vstack((self.supStates,new_states.todense()))
		self.Actions = np.vstack((self.Actions,new_actions))
		self.Weights = np.vstack((self.Weights,weights))
		self.trainModel(self.States,self.Actions)

	def listToMat(self,States):

		matStates = np.zeros((len(States),States[0].shape[0]))

		for i in range(len(matStates)):
			matStates[i,:] = States[i]

		return matStates

	def saveModel(self):
		pickle.dump(self.States,open('states.p','wb'))
		pickle.dump(self.Actions,open('actions.p','wb'))

