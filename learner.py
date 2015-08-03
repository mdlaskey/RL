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
from Tools.AHQP import AHQP

import sys

# Make sure that caffe is on the python path:
sys.path.append('/home/wesley/caffe/python')

import caffe
import os

import random

# OpenCV
import cv2

class Learner():

	verbose = True
	option_1 = False
	gamma = 0.01
	gamma_clf = 1e-3
	first_time = True
	iter_  = 1
	use_AHQP = True 
	# Neural net implementation
	neural = False

	# Assumes current directory is "RL/"
	NET_SUBDIR = os.getcwd() + '/net/'
	SOLVER_FILE = os.path.join(NET_SUBDIR, 'net_solver.prototxt')
	SOLVER_FILE_FT = os.path.join(NET_SUBDIR, 'net_solver_ft.prototxt')
	MODEL_FILE = os.path.join(NET_SUBDIR, 'net_model.prototxt')
	TRAINED_MODEL = os.path.join(os.getcwd(), '_iter_1000.caffemodel')

	def __init__(self,sigma=1.0):
		self.ahqp_solver_g = AHQP()
		self.ahqp_solver_b = AHQP(sigma = 200,nu=0.001)

	def Load(self,gamma = 1e-2, retrain_net=False):
		self.sup_states = pickle.load(open('states.p','rb'))
		self.trainSupport()
		if self.neural:
			if retrain_net:
				self.trainModel(retrain_net=retrain_net)

		else:
			self.Actions = pickle.load(open('actions.p','rb'))
			self.Weights = np.zeros(self.Actions.shape)+1
			
			self.trainModel(self.States,self.Actions)


	def clearModel(self):
		self.States = pickle.load(open('states.p','rb'))
		self.Actions = pickle.load(open('actions.p','rb'))
		self.Weights = np.zeros(self.Actions.shape)+1

	def split_training_test(self, States, Action):
		"""
		Splits the states/action pairs into
		training/test sets of 80/20 percent.
		"""
		total_size = len(States)
		train_size = int(total_size * 0.8)

		train_indices = random.sample([i for i in range(total_size)], train_size)
		test_indices = [i for i in range(total_size) if i not in train_indices]

		train_states = np.array([np.array(States[i]).astype(np.float32) for i in train_indices])
		train_actions = np.array([Action[i] for i in train_indices]).astype(np.float32)
		test_states = np.array([np.array(States[i]).astype(np.float32) for i in test_indices])
		test_actions = np.array([Action[i] for i in test_indices]).astype(np.float32)

		return train_states, train_actions, test_states, test_actions

	def output_images(self, States, Action):
		"""
		Downsamples the states twice.
		Outputs the given states/actions into
		image files for neural net training in Caffe.
		"""
		downsampled_states = [cv2.pyrDown((cv2.pyrDown(img))) for img in States]

		train_states, train_actions, test_states, test_actions = \
			self.split_training_test(downsampled_states, Action)

		self.sup_states = train_states

		# train/test.txt should be a list of image files / actions to be read
		with open(os.path.join(self.NET_SUBDIR, 'train.txt'), 'w') as f:
			for i in range(len(train_states)):
				train_filename = self.NET_SUBDIR + 'train_images/' + 'train_img_{0}.png'.format(i)
				cv2.imwrite(train_filename, train_states[i])
				f.write(train_filename + " " + str(int(train_actions[i])) + '\n')

		with open(os.path.join(self.NET_SUBDIR, 'test.txt'), 'w') as f:
			for i in range(len(test_states)):
				test_filename = self.NET_SUBDIR + 'test_images/' + 'test_img_{0}.png'.format(i)
				cv2.imwrite(test_filename, test_states[i])
				f.write(test_filename + " " + str(int(test_actions[i])) + '\n')



	def trainModel(self, States=None, Action=None, fineTune = False, retrain_net=False):
		"""
		Trains model on given states and actions.
		Uses neural net or SVM based on global
		settings.
		"""
		
		States, Action = States[1:], Action[1:]
		print "States.shape"
		print States.shape
		print "Action.shape"
		print Action.shape

		Action = np.ravel(Action)

		# Original SVC implementation
		self.clf = svm.LinearSVC()
		#self.clf.class_weight = 'auto' 
		print "GAMMA",self.gamma
		self.clf.C = self.gamma
	
		# self.scaler = preprocessing.StandardScaler().fit(States[:,:,0])
		# States = self.scaler.transform(States[:,:,0])
		
		self.clf.fit(States[:,:,0], Action)

		if (self.verbose or self.use_AHQP):
			self.debugPolicy(States[:,:,0], Action)
		if (self.use_AHQP):
			
			good_labels = self.labels == 1.0
			bad_labels = self.labels == -1.0


			States_g = States[good_labels,:]
			labels = np.zeros(States_g.shape[0])+1.0
			self.ahqp_solver_g.assembleKernel(States_g, labels)

			States_b = States[bad_labels,:]
			labels = np.zeros(States_b.shape[0])+1.0
			self.ahqp_solver_b.assembleKernel(States_b,labels)


			self.ahqp_solver_b.solveQP()
			self.ahqp_solver_g.solveQP()

			
	def getScoreNovel(self,States):
		num_samples = States.shape[0]
		avg = 0
		for i in range(num_samples):
			ans = self.novel.predict(States[i,:])
			if(ans == -1):
				ans = 0
			avg = avg+ans/num_samples

		return avg

	def debugPolicy(self, States, Action):
	
		prediction = self.clf.predict(States)
		classes = dict()
		self.labels = np.zeros(Action.shape)
		for i in range(Action.shape[0]):
			if (Action[i] not in classes):
				value = np.zeros(3)
				classes.update({Action[i]: value})
			classes[Action[i]][0] += 1
			if (Action[i] != prediction[i]):
				classes[Action[i]][1] += 1
				self.labels[i] = -1.0
			else:
				self.labels[i] = 1.0
			classes[Action[i]][2] = classes[Action[i]][1] / classes[Action[i]][0]
		for d in classes:
			print d, classes[d]

		self.precision = self.clf.score(States, Action)


	def getPrecision(self):
		return self.precision

	def processState(self):
		net = caffe.Net (self.MODEL_FILE,self.TRAINED_MODEL,caffe.TEST)


		sup_states_t = np.zeros((len(self.sup_states),40))

		for i in range(len(self.sup_states)):
			# Caffe takes in 4D array inputs.
			data4D = np.zeros([1,3,125,125])

			# Fill in last 3 dimensions

			data4D[0,0,:,:] = self.sup_states[i,:,:,0]
			data4D[0,1,:,:] = self.sup_states[i,:,:,1]
			data4D[0,2,:,:] = self.sup_states[i,:,:,2]

			net.blobs['data'].data[...] = data4D
			net.forward(start='conv1',end='fc1')
			sup_states_t[i,:] = net.blobs['fc1'].data

		self.sup_states = sup_states_t

	def trainSupport(self):

		if(self.sup_states.shape[1] != 40):
			self.processState()

		self.scaler = preprocessing.StandardScaler().fit(self.sup_states)
		self.sup_states = self.scaler.transform(self.sup_states)
		self.novel = svm.OneClassSVM()


		self.novel.gamma = self.gamma.clf # Gamma
		self.novel.nu = 1e-3
		self.novel.kernel = 'rbf'
		self.novel.verbose = True

		self.novel.max_iter = 3000

		self.novel.fit(self.sup_states)


		self.saveModel()


 	def getAction(self, state):
		"""
		Returns a prediction given the input state.
		Uses neural net or SVM based on global
		settings.
		"""
		if self.neural:
			net = caffe.Net (self.MODEL_FILE,self.TRAINED_MODEL,caffe.TEST)

			# Caffe takes in 4D array inputs.
			data4D = np.zeros([1,3,125,125])

			# Fill in last 3 dimensions

			img = cv2.pyrDown((cv2.pyrDown(state)))

			data4D[0,0,:,:] = img[:,:,0]
			data4D[0,1,:,:] = img[:,:,1]
			data4D[0,2,:,:] = img[:,:,2]
			#cv2.imshow('img',img)
			# Forward call creates a dictionary corresponding to the layers

			pred_dict = net.forward_all(data=data4D)
			# 'prob' layer contains actions and their respective probabilities
			prediction = pred_dict['prob'].argmax()
			print pred_dict

			return [prediction]
		else:

			img = cv2.pyrDown((cv2.pyrDown(state)))

			img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
			state = np.reshape(img,(img.shape[0]*img.shape[1],1))
			# winSize = (32,32)
			# blockSize = (16,16)
			# blockStride = (8,8)
			# cellSize = (8,8)
			# nbins = 9
			# derivAperture = 1
			# winSigma = 4.
			# histogramNormType = 0
			# L2HysThreshold = 2.0000000000000001e-01
			# gammaCorrection = 0
			# nlevels = 64
			# hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
			#                 histogramNormType,L2HysThreshold,gammaCorrection,nlevels)

			# state = hog.compute(img)
			
			# state = self.scaler.transform(state.T)
			return self.clf.predict(state.T)

	def askForHelp(self,state):
	
		state = self.scaler.transform(state)
		if(self.ahqp_solver_b.predict(state) == 1.0):
			return -1.0
		else:
			return self.ahqp_solver_g.predict(state)

	def getNumData(self):
		return self.Action.shape[0]

	def newModel(self,states,actions):
		states = csr_matrix(states)

		self.States = states
		self.supStates = states.todense()
		self.Actions = actions
		self.Weights = np.zeros(actions.shape)+1
		self.trainModel(self.States,self.Actions)

	def updateModel(self,new_states,new_actions,weights):
		print "UPDATING MODEL"

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
		pickle.dump(self.sup_states,open('states.p','wb'))


