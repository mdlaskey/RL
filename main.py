from race_game import RaceGame
from Agents.DAgger import Dagger
import pygame
import IPython
import matplotlib.pyplot as plt

race_game = RaceGame(agent = Dagger(),graphics=True, MAX_LAPS=5)
while race_game.running:
    race_game.control_car(input_sequence=None, driving_agent=True)


plt.plot(race_game.queries,race_game.cost)
IPython.embed()


plt.show()