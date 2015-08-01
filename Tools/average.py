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



results = pickle.load(open('results_DAGGER.p','rb'))

avg_cost = np.zeros(len(results[0][1]))
avg_queries = np.zeros(len(results[0][1]))

IPython.embed()

for t in range(len(results[0][1])):
	total_cost = 0.0
	total_queries = 0.0 
	for k in range(len(results)):
		total_queries += results[k][0][t] 
		total_cost += results[k][1][t]
	
	avg_queries[t] = total_queries/float(len(results[0][1]))
	avg_cost[t] = total_cost/float(len(results[0][1]))

plt.plot(avg_queries,avg_cost)
plt.show()