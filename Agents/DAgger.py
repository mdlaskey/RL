import numpy as np
import IPython
__author__ = "Sergey Karakovskiy, sergey at idsia fullstop ch"
__date__ = "$May 1, 2009 2:46:34 AM$"

from learner import Learner 
from scipy.sparse import csr_matrix
from scipy.sparse import vstack

class Dagger():
    """ In fact the Python twin of the
        corresponding Java ForwardAgent.
    """
         
    def __init__(self,initialTraining):
        """Constructor"""
        self.initialTraining = initialTraining
        self.States = []
        self.Actions = []
        self.learner = Learner()
     
        
    def askForHelp(self,img):
        return 1
    def loadModel(self):
        self.learner.Load()

    def getAction(self,img):
        """ Possible analysis of current observation and sending an action back
        """
      
        action = self.learner.getAction(img)

        return action

    def integrateObservation(self, img,action):
        """This method stores the observation inside the agent"""
        self.States.append(img)
        self.Actions.append(action)  
            #self.printLevelScene()


    def updateModel(self):
     
        States = np.array(self.States)
        Actions = np.array(self.Actions)
        self.learner.trainModel(States,Actions,fineTune = True)
       

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

    