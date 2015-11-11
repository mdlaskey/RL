from race_game_asst import RaceGame
from Agents.DAgger import Dagger
from Agents.SHEATH import SHEATH
from Classes.RobotCont import RobotCont
import pygame
import IPython
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
import car
import dummy_car
import learner
import copy

if __name__ == '__main__':

    gammas = [1e-2]
    ROUNDS = 3
    rc = RobotCont()
    sigma_results = []
   
    sigmas = [300,250,150, 50,1]
    # for sigma in sigmas:
    results = []

    cost = []
    states = []
    controls = []

    for seed in range(ROUNDS):

        incorr = [] 
        names = [] 
        for i in range(5):
            if(seed == 0):
                race_game = RaceGame()
            else: 
                race_game = RaceGame(roboCoach = rc)
            while race_game.running:
                race_game.control_car()
            
            #IPython.embed()
            cost.append(race_game.cost)
            states.append(race_game.states)
           
            controls.append(race_game.controls)

        rc.calQ(cost,states,controls)

        # plt.plot(race_game.cost)
        # plt.show()
        # sigma_results.append(results)
    

    pickle.dump(results,open('results_expert.p','wb'))
    IPython.embed()
    plt.legend(names,loc='upper left')
    plt.show()
    plt.savefig("results_DAgger_auto.eps")
   
