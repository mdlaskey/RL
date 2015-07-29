from race_game import RaceGame
from Agents.DAgger import Dagger
import pygame
import IPython
import matplotlib.pyplot as plt
import car
import dummy_car
import learner
import copy

if __name__ == '__main__':
    #gammas = [5e-3, 1e-3, 5e-2, 1e-2, 5e-1]
    gammas = [1e-3]
    race_game_queries = []
    race_game_cost = []
    for gamma in gammas:
        learner.Learner.gamma = gamma
        learner.Learner.gamma_clf = gamma
        race_game = RaceGame(agent = Dagger(),graphics=True, MAX_LAPS=10)
        while race_game.running:
            race_game.control_car(input_sequence=None, driving_agent=True)
        assert race_game.agent.learner.gamma == gamma
        queries = copy.deepcopy(race_game.queries)
        cost = copy.deepcopy(race_game.cost)
        race_game_queries.append(queries)
        race_game_cost.append(cost)
        results = plt.plot(queries,cost)
        plt.savefig("results_gamma_{0}.png".format(str(gamma)))
        plt.close()
    IPython.embed()
