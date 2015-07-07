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

import sys

# Make sure that caffe is on the python path:
sys.path.append('/home/wesley/caffe/python')

import caffe
import os
import h5py
import shutil
import tempfile

import random
import matplotlib.pyplot as plt

# OpenCV
import cv2

class Learner():

	verbose = True
	option_1 = False 
	gamma = 1e-3
	gamma_clf = 1e-3
	first_time = True 
	iter_  = 1

	# Neural net implementation
	neural = True

	# Assumes current directory is "RL/"
	NET_SUBDIR = os.getcwd() + '/net/'
	SOLVER_FILE = os.path.join(NET_SUBDIR, 'net_solver.prototxt')
	MODEL_FILE = os.path.join(NET_SUBDIR, 'net_model.prototxt')
	TRAINED_MODEL = os.path.join(os.getcwd(), '_iter_1000.caffemodel')



	def Load(self,gamma = 1e-3):
		if self.neural:
			caffe.set_mode_cpu()
			self.net = caffe.Classifier(model_file=self.MODEL_FILE, pretrained_file=self.TRAINED_MODEL)
		else:
			self.States = pickle.load(open('states.p','rb'))
			self.Actions = pickle.load(open('actions.p','rb'))
			self.Weights = np.zeros(self.Actions.shape)+1
			self.gamma = gamma
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

		downsampled_states = [cv2.pyrDown((cv2.pyrDown(img))).transpose([1,0,2]) for img in States]
		channel_swapped = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in downsampled_states]
		Action = [int(act) for act in Action]
		train_states, train_actions, test_states, test_actions = \
			self.split_training_test(channel_swapped, Action)
		# train/test.txt should be a list of image files / actions to be read
		with open(os.path.join(self.NET_SUBDIR, 'train.txt'), 'w') as f:
			for i in range(len(train_states)):
				train_filename = self.NET_SUBDIR + 'train_images/' + 'train_img_{0}.png'.format(i)
				cv2.imwrite(train_filename, train_states[i])
				f.write(train_filename + " " + str(train_actions[i]) + '\n')

		with open(os.path.join(self.NET_SUBDIR, 'test.txt'), 'w') as f:
			for i in range(len(test_states)):
				test_filename = self.NET_SUBDIR + 'test_images/' + 'test_img_{0}.png'.format(i)
				cv2.imwrite(test_filename, test_states[i])
				f.write(test_filename + " " + str(test_actions[i]) + '\n')



	def trainModel(self, States=None, Action=None):
		"""
		Trains model on given states and actions.
		Uses neural net or SVM based on global
		settings.
		If no states/actions are passed in, retrains net
		based on preprocessed images.
		"""
		if States != None and Action != None:
			print "States.shape"
			print States.shape
			print "Action.shape"
			print Action.shape
			Action = np.ravel(Action)

		if self.neural:
			if States != None and Action != None:
				# Output images if passing in new states/actions
				self.output_images(States, Action)

			# Change to "caffe.set_mode_gpu() for GPU mode"
			caffe.set_mode_cpu()
			solver = caffe.get_solver(self.SOLVER_FILE)
			solver.solve()
		else:
			# Original SVC implementation
			self.clf = svm.LinearSVC()
			self.clf.class_weight = 'auto'
			self.clf.C = 1e-2
			self.clf.fit(States, Action)

		"""
		# Original novel implementation
		self.novel = svm.OneClassSVM()

		self.novel.gamma = self.gamma
		self.novel.nu = 1e-3
		self.novel.kernel = 'rbf'
		self.novel.verbose = False
		self.novel.shrinking = False
		self.novel.max_iter = 3000

		self.novel.fit(self.supStates)

		if (self.verbose):
			self.debugPolicy(States, Action)
		"""
	

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

 	def getAction(self, state):
		"""
		Returns a prediction given the input state.
		Uses neural net or SVM based on global
		settings.
		"""
		if self.neural:
			downsampled = cv2.pyrDown((cv2.pyrDown(state)).transpose([1,0,2]))

			pred_matrix = self.net.predict([downsampled])  # predict takes any number of images, and formats them for the Caffe net automatically
			prediction = pred_matrix[0].argmax()

			"""
			input_image = plt.imread('/home/wesley/Desktop/RL/net/train_images/train_img_0.png')
			input_pred = self.net.predict([input_image])
			"""
			"""
			caffe_in = np.zeros([1,3,125,125], dtype=np.float32)
			caffe_in[0] = input_image.transpose([2,0,1])
			out = self.net.forward_all(data=caffe_in)
			pred_matrix2 = out[self.net.outputs[0]]
			"""

			print "pred_matrix: {0}".format(pred_matrix)
			print "Prediction: {0}".format(prediction)
			return [prediction]
		else:
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

