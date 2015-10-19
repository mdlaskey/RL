from race_game_asst import RaceGame
from Agents.DAgger import Dagger
from Agents.SHEATH import SHEATH
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
    ROUNDS = 20
   
    sigma_results = []
   
    sigmas = [300,250,150, 50,1]
    # for sigma in sigmas:
    results = []
    for seed in range(ROUNDS):

        incorr = [] 
        names = [] 
        sup_incr = np.zeros(ROUNDS)
        unc_incr = np.zeros(ROUNDS)
        qur_incr = np.zeros(ROUNDS)
        race_game = RaceGame()
        while race_game.running:
            race_game.control_car()
    

        values = [race_game.queries,race_game.cost]
        results.append(values)
        names.append(str(seed))
        plt.plot(values[0],values[1])
        sigma_results.append(results)
       
    pickle.dump(results,open('results_expert.p','wb'))
    IPython.embed()
    plt.legend(names,loc='upper left')
    plt.show()
    plt.savefig("results_DAgger_auto.eps")
   
