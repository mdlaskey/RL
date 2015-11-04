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
    ROUNDS = 10
    rc = RobotCont()
    sigma_results = []
   
    sigmas = [300,250,150, 50,1]
    # for sigma in sigmas:
    results = []

    oil_cost = []
    oil_states = []
    oil_prob = []
    oil_controls = []

    for seed in range(ROUNDS):

        incorr = [] 
        names = [] 
       
        race_game = RaceGame()
        while race_game.running:
            race_game.control_car()
        

        oil_cost.append(race_game.cost_oil)
        oil_states.append(race_game.states_oil)
        oil_prob.append(race_game.probs_oil)
        oil_controls.append(race_game.controls_oil)


        # plt.plot(race_game.cost)
        # plt.show()
        # sigma_results.append(results)
    

    rc.calQ(oil_cost,oil_states,oil_controls)
    grad = rc.calGrad(oil_states,oil_controls,oil_prob)

    pickle.dump(results,open('results_expert.p','wb'))
    IPython.embed()
    plt.legend(names,loc='upper left')
    plt.show()
    plt.savefig("results_DAgger_auto.eps")
   
