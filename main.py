
from race_game_comp import RaceGame

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
        race_game = RaceGame(agent = SHEATH(sigma = 150),graphics=True, MAX_LAPS=1,seed=seed)
        while race_game.running:
            race_game.control_car(input_sequence=None, driving_agent=True)
        # sup_incr[seed] = race_game.sup_incr[0]
        # unc_incr[seed] = race_game.unc_incr[0]
        # qur_incr[seed] = race_game.qur_incr[0]
        unc_incr[seed] = race_game.unc_incr[0]

        values = [race_game.queries,race_game.cost]#,race_game.sup_incr[0],race_game.unc_incr[0]]
        results.append(values)
        names.append(str(seed))
        plt.plot(values[0],values[1])
        sigma_results.append(results)
       
    pickle.dump(results,open('results_expert.p','wb'))
    IPython.embed()
    plt.legend(names,loc='upper left')
    plt.show()
    plt.savefig("results_DAgger_auto.eps")
   
