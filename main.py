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
    gammas = [1e-3]
    results = []
    for gamma in gammas:
        learner.Learner.gamma = gamma
        learner.Learner.gamma_clf = gamma
        race_game = RaceGame(agent = Dagger(),graphics=True, MAX_LAPS=10)
        while race_game.running:
            race_game.control_car(input_sequence=None, driving_agent=True)
        assert race_game.agent.learner.gamma == gamma
        values = [copy.deepcopy(race_game.queries),copy.deepcopy(race_game.cost)]
        results.append(values)
        plt.plot(values[0],values[1])
        plt.savefig("results_gamma_{0}.png".format(str(gamma)))
        plt.close()
    pickle.dump(results,open('results.p','wb'))
IPython.embed()
