from race_game import RaceGame
from Agents.DAgger import Dagger
from Agents.SHEATH import SHEATH
import pygame
import IPython
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt

rounds = 2
results = []

#results = pickle.load(open('results.p','rb'))
race_game = RaceGame(agent = Dagger(),graphics=True, MAX_LAPS=15)
while race_game.running:
	race_game.control_car(input_sequence=None, driving_agent=True)
	
values = [race_game.queries,race_game.cost]

results.append(values)
plt.plot(values[0],values[1])


# # race_game = RaceGame(agent = Dagger(),graphics=True, MAX_LAPS=8)
# # while race_game.running:
# # 	race_game.control_car(input_sequence=None, driving_agent=True)
	
# # values = [race_game.queries,race_game.cost]

results.append(values)
plt.plot(values[0],values[1])
pickle.dump(results,open('results.p','wb'))
IPython.embed()


plt.show()