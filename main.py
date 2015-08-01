from race_game import RaceGame
from Agents.DAgger import Dagger
#from Agents.SHEATH import SHEATH
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
    results = []
    laps = 2
    race_game = RaceGame(agent = Dagger(),graphics=True, MAX_LAPS=laps)
    while race_game.running:
        race_game.control_car(input_sequence=None, driving_agent=True)
    values = [copy.deepcopy(race_game.queries),copy.deepcopy(race_game.cost)]
    results.append(values)
    plt.plot(values[0],values[1])
    plt.savefig("results_{0}_laps.png".format(str(laps)))
    plt.close()
    pickle.dump(results,open('results.p','wb'))
IPython.embed()
