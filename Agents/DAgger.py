import numpy as np
import IPython
__author__ = "Sergey Karakovskiy, sergey at idsia fullstop ch"
__date__ = "$May 1, 2009 2:46:34 AM$"

from learner import Learner 
import cv2
from scipy.sparse import csr_matrix
from scipy.sparse import vstack

class Dagger():
    """ In fact the Python twin of the
        corresponding Java ForwardAgent.
    """
         
    def __init__(self):
        """Constructor"""
        self.initialTraining = True
        self.States = []
        self.Actions = []
        self.learner = Learner()
        self.human_input = 0.0

    def getName(self):
        return 'Dagger'

    def loadModel(self):
        self.learner.Load()

    def askForHelp(self,img):
        return 1

    def getAction(self,img):
        """ Possible analysis of current observation and sending an action back
        """
      
        action = self.learner.getAction(img)
        self.actionTaken = action
        return action

    def integrateObservation(self, img,action):
        """This method stores the observation inside the agent"""
        if(not self.initialTraining):
            self.human_input += 1.0
        if (self.initialTraining or (self.actionTaken[0] != action[0])):
            img = cv2.pyrDown((cv2.pyrDown(img)))
            winSize = (32,32)
            blockSize = (16,16)
            blockStride = (8,8)
            cellSize = (8,8)
            nbins = 9
            derivAperture = 1
            winSigma = 4.
            histogramNormType = 0
            L2HysThreshold = 2.0000000000000001e-01
            gammaCorrection = 0
            nlevels = 64
            hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                            histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
            state = hog.compute(img)
            self.States.append(state)
            self.Actions.append(action)  
            #self.printLevelScene()


    def updateModel(self):

        States = np.array(self.States)
        Actions = np.array(self.Actions)
        self.learner.trainModel(States,Actions)
        

    def getDataAdded(self):
        return self.dataAdded

    def newModel(self):
        states = np.array(self.States)
        actions = np.array(self.Actions) 
        self.learner.trainModel(states,actions)

    def getNumData(self): 
        return self.learner.getNumData()
 
    def reset(self):
        self.States = []
        self.Actions = []

    