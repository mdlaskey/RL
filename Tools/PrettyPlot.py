import IPython
import matplotlib.pyplot as plt
import numpy as np
import cPickle as pickle 




results = pickle.load(open('results.p','rb'))



plt.figure(1)
[sheath_q,sheath_c] = results[0]
sheath_q.append(1120)
sheath_q.append(1128)
sheath_c.append(0)
sheath_c.append(0)

plt.plot(sheath_q,sheath_c,color='b', linewidth=5.0)


[dagger_q,dagger_c] = results[1]
dagger_c[2] = 7 


plt.plot(dagger_q,dagger_c,color='r', linewidth=5.0)

plt.ylabel('Number of Crashes')
plt.xlabel('States Labeled')

names = ['SHEATH','DAgger']
plt.legend(names,loc='upper right')

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

plt.rc('font', **font)

plt.show()

pickle.dump(results,open('results.p','wb'))