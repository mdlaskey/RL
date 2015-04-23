import math
import random
import numpy as np
import IPython
import pickle 
from numpy import linalg as LA
from sklearn import svm 
from sklearn import preprocessing 


class Learner():

	def Load(self):
		self.States = pickle.load(open('states.p','rb'))
		self.Actions = pickle.load(open('actions.p','rb'))
		self.trainModel(self.States,self.Actions)


	def trainModel(self,States,Action):
		self.clf = svm.SVC()
		self.novel = svm.OneClassSVM()
		self.scaler = preprocessing.StandardScaler().fit(States)
		States = self.scaler.transform(States)
		Action = np.ravel(Action)
		self.novel.nu = 0.05
		self.clf.fit(States,Action)
		self.novel.fit(States,Action)
		#IPython.embed()

 	def getAction(self,state):
 		state = self.scaler.transform(state)
		return self.clf.predict(state)

	def askForHelp(self,state):
		#IPython.embed()
		#if(abs(state[1]) > 80  and abs(state[2]) == 0):
			#IPython.embed() 
			#return -1
		#else: 
			#return 1
		state = self.scaler.transform(state)
		return self.novel.predict(state)

	def updateModel(self,new_states,new_actions):
		print "UPDATING MODEL"
		self.States = np.vstack((self.States,new_states))
		self.Actions = np.vstack((self.Actions,new_actions))
		self.trainModel(self.States,self.Actions)

	def saveModel(self):
		pickle.dump(self.States,open('states.p','wb'))
		pickle.dump(self.Actions,open('actions.p','wb'))