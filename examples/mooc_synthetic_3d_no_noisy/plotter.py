# -*- coding: utf-8 -*-
"""
Created on Mon May 30 11:45:37 2016

@author: root
"""


import numpy as np
import matplotlib.pyplot as plt

with open("pesmoc_0/hypervolume_solution.txt") as f:
    solution = f.read()
solution = float(solution)
data = np.array([])
data_random = np.array([])
iterations=100
number_executions=100
for i in range(1,number_executions,1):
    with open("pesmoc/hypervolumes_"+str(i)+".txt") as f:
        data_iteration = f.read()
    data_iteration = data_iteration.split('\n')[:-1]
    data = np.append(data,data_iteration)
    with open("random/hypervolumes_"+str(i)+".txt") as f:
        data_random_iteration = f.read()
    data_iteration_random= data_iteration_random.split('\n')[:-1]
    data_random = np.append(data_random,data_iteration_random)
    
data= data.reshape((iterations,number_executions))
data_random= data_random.reshape((iterations,number_executions))
data = np.mean(data,axis=0)
data_random = np.mean(data_random,axis=0)
data_var = np.var(data,axis=0)
data_random_var = np.var(data_random,axis=0)
data = [np.log(solution-float(i)) for i in data]
data_random = [np.log(solution-float(i)) for i in data_random]

fig = plt.figure()

ax1 = fig.add_subplot(111)

ax1.set_title("Hypervolumes")    
ax1.set_xlabel('Iterations')
ax1.set_ylabel('Hypervolumes')

ax1.plot([i for i in range(0,len(data),1)],data)
ax1.plot([i for i in range(0,len(data_random),1)],data_random)

leg = ax1.legend(['PESMOC', 'RANDOM'], loc='upper right')

plt.show()
print len(data)