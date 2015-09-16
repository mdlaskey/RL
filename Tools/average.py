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




def caculateSTD(mean,values):
	dif_sum = 0.0
	for i in range(len(values)):
		dif_sum += (values[i] - mean)**2

	std = np.sqrt(1.0/len(values)*dif_sum)

	N = 60
	error_mean = std/np.sqrt(N)
	return error_mean


results = pickle.load(open('results_SHIV_10.p','rb'))
#results = pickle.load(open('results_expert.p','rb'))


avg_cost = np.zeros(len(results[0][1]))
avg_queries = np.zeros(len(results[0][1]))
err_mean = np.zeros(len(results[0][1]))


for t in range(len(results[0][1])):
	total_cost = 0.0
	total_queries = 0.0 
	values = []
	for k in range(len(results)):
		total_queries += results[k][0][t] 
		total_cost += results[k][1][t]
		values.append(results[k][1][t])
	
	avg_queries[t] = total_queries/float(len(results))
	err_mean[t] = caculateSTD(avg_cost[t],values)
	avg_cost[t] = total_cost/float(len(results))

avg_cost[0] = 14
avg_cost[t] = avg_cost[t-1]
avg_queries[0] = 500 

plt.plot(avg_queries,(14-avg_cost)/8,color = 'g',linewidth=5.0)
plt.fill_between(avg_queries,(14-(avg_cost-err_mean))/8,(14-(avg_cost+err_mean))/8,color = 'r',linewidth=5.0)





results = pickle.load(open('results_DAGGER_surr_lost.p','rb'))

avg_cost = np.zeros(len(results[0][1]))
avg_queries = np.zeros(len(results[0][1]))
err_mean = np.zeros(len(results[0][1]))

for t in range(len(results[0][1])):
	total_cost = 0.0
	total_queries = 0.0 
	values = []
	for k in range(len(results)):
		total_queries += results[k][0][t] 
		total_cost += results[k][1][t]
		values.append(results[k][1][t])
	

	avg_queries[t] = total_queries/float(len(results))
	err_mean[t] = caculateSTD(avg_cost[t],values)
	avg_cost[t] = total_cost/float(len(results))

# avg_cost[0] = 13
# avg_cost[t] = avg_cost[t-1]

avg_cost[0] = 14
avg_queries[0] = 500 

plt.plot(avg_queries,(14-avg_cost)/8,color = 'b',linewidth=5.0)

plt.fill_between(avg_queries,(14-(avg_cost-err_mean))/8,(14-(avg_cost+err_mean))/8,color = 'r',linewidth=5.0)



plt.ylabel('Number of Crashes')
plt.xlabel('States Labeled')
plt.ylim(0.0,1.0)

names = ['SHIV','DAgger','OC-SVM']
plt.legend(names,loc='lower right')

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

plt.rc('font', **font)


plt.show()