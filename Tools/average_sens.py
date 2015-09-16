from race_game import RaceGame
from Agents.DAgger import Dagger
from Agents.SHEATH import SHEATH

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


results = pickle.load(open('results_Sens_10.p','rb'))


for result in results:

	avg_cost = np.zeros(len(result[0][1]))
	avg_queries = np.zeros(len(result[0][1]))
	err_mean = np.zeros(len(result[0][1]))


	for t in range(len(result[0][1])):
		total_cost = 0.0
		total_queries = 0.0 
		values = []
		for k in range(len(result)):
			total_queries += result[k][0][t] 
			total_cost += result[k][1][t]
			values.append(result[k][1][t])

		avg_queries[t] = total_queries/float(len(result))
		err_mean[t] = caculateSTD(avg_cost[t],values)
		avg_cost[t] = total_cost/float(len(result))

	print "LABELS", avg_queries
	print "COST",avg_cost

	plt.plot(avg_queries,avg_cost,linewidth=5.0)
#plt.fill_between(avg_queries,avg_cost-err_mean,avg_cost+err_mean,color = 'r',linewidth=5.0)


plt.ylabel('Number of Crashes')
plt.xlabel('States Labeled')


names = ['300','250','150', '50','1']
plt.legend(names,loc='upper right')

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

plt.rc('font', **font)


plt.show()