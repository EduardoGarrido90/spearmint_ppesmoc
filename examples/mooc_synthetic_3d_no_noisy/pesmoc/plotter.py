# -*- coding: utf-8 -*-
"""
Created on Mon May 30 11:45:37 2016

@author: root
"""


import numpy as np
import matplotlib.pyplot as plt

with open("hypervolume_solution.txt") as f:
    solution = f.read()
with open("hypervolumes.txt") as f:
    data = f.read()
    
print solution
solution = float(solution)
data = data.split('\n')[:-1]
print data
data = [np.log(solution-float(i)) for i in data]

fig = plt.figure()

ax1 = fig.add_subplot(111)

ax1.set_title("Hypervolumes")    
ax1.set_xlabel('Iterations')
ax1.set_ylabel('Hypervolumes')

ax1.plot([i for i in range(0,len(data),1)],data, c='r', label='the data')

leg = ax1.legend()

plt.show()
print len(data)