print(__doc__)

# Author: Virgile Fritsch <virgile.fritsch@inria.fr>
# License: BSD 3 clause

import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn.datasets import load_boston
import cPickle as pickle 
import IPython
# Get data
Samples = np.random.randn(2,100)

X1 = Samples.T

# Define "classifiers" to be used
classifiers = {
    "OCSVM": OneClassSVM(nu= 0.9,gamma = 1/100,verbose = True)}
colors = ['r']
legend1 = {}
legend2 = {}


# Learn a frontier for outlier detection with several classifiers
#xx1, yy1 = np.meshgrid(np.linspace(1250,0, 500), np.linspace(1250, 0, 500))
xx1, yy1 = np.meshgrid(np.linspace(-5,5), np.linspace(-5, 5))
IPython.embed()
for i, (clf_name, clf) in enumerate(classifiers.items()):
    plt.figure(1)

    clf.fit(X1)
    Z1 = clf.decision_function(np.c_[xx1.ravel(), yy1.ravel()])
    Z1 = Z1.reshape(xx1.shape)
    legend1[clf_name] = plt.contour(
          xx1, yy1, Z1, levels=[0], linewidths=5, colors=colors[i])


legend1_values_list = list( legend1.values() )
legend1_keys_list = list( legend1.keys() )

# Plot the results (= shape of the data points cloud)
plt.figure(1)  # two clusters
plt.scatter(X1[:, 0], X1[:, 1], color='black')
plt.xlim((xx1.min(), xx1.max()))
plt.ylim((yy1.min(), yy1.max()))
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 30}

matplotlib.rc('font', **font)

plt.ylabel("Y Position")
plt.xlabel("X Position")



plt.show()