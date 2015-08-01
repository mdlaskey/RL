from race_game import RaceGame
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
    #gammas = [5e-3, 1e-3, 5e-2, 1e-2, 5e-1]
    gammas = [1e-2]
    results = []
    names = [] 
    for seed in range(10):
        race_game = RaceGame(agent = Dagger(),graphics=True, MAX_LAPS=10,seed=seed)
        while race_game.running:
            race_game.control_car(input_sequence=None, driving_agent=True)
        
        values = [race_game.queries,race_game.cost]
        results.append(values)
        names.append(str(seed))
        plt.plot(values[0],values[1])

    pickle.dump(results,open('results_DAGGER.p','wb'))
    plt.legend(names,loc='upper left')
    plt.show()
    plt.savefig("results_DAgger_auto.eps")
   
IPython.embed()
