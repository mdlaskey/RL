#!/usr/bin/env python
# a stacked bar plot with errorbars
import numpy as np
import matplotlib.pyplot as plt


N = 5
riskMeans   = (59,45, 40, 21, 15)
riskStd     = (2,4,  4, 1, 2)
ind = np.arange(N)    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence

p1 = plt.bar(ind, riskMeans,   width, color='r', yerr=riskStd)


plt.ylabel('Prediction Error')
plt.xticks(ind+width/2., ('Confidence Measure','MMD', 'Query by Committee', 'One Class SVM', 'R.O.C. SVM') )
plt.yticks(np.arange(0,70,15))

plt.show()
