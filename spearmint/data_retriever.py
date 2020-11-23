# -*- coding: utf-8 -*-
"""
Created on Tue May 31 11:24:08 2016

@author: root
"""

import numpy as np
import os.path
import json

iterations=100
number_executions=100
experiments = [{"name":"mooc_synthetic_3d_noisy"},{"name":"mooc_synthetic_3d_no_noisy"},{"name":"mooc_synthetic_2d_noisy"},
               {"name":"mooc_synthetic_2d_no_noisy"},{"name":"mooc_synthetic_6d_noisy"},{"name":"mooc_synthetic_6d_no_noisy"}]

#Finding ideal solution and iterations done.
for s in experiments:    
    s["name"] = "examples/" + s["name"]
    for i in range(1,number_executions,1):
        if(os.path.isfile(s["name"]+"_"+str(i)+"/pesmoc/hypervolume_solution.txt")):
            with open(s["name"]+"_"+str(i)+"/pesmoc/hypervolume_solution.txt") as f:
                s["hyp_solution"] = float(f.read().split("\n")[0])
            break;
        if(os.path.isfile(s["name"]+"_"+str(i)+"/random/hypervolume_solution.txt")):
            with open(s["name"]+"_"+str(i)+"/random/hypervolume_solution.txt") as f:            
                s["hyp_solution"] = float(f.read().split("\n")[0])
            break;

#Retrieving results and computing statistics.
for s in experiments:
    data = np.array([])
    hits = 0
    data_random = np.array([])
    hits_random = 0
    for i in range(1,number_executions,1):
        if(os.path.isfile(s["name"]+"_"+str(i)+"/pesmoc/hypervolumes.txt")):
            with open(s["name"]+"_"+str(i)+"/pesmoc/hypervolumes.txt") as f:
                data_iteration = f.read()
            data_iteration = data_iteration.split('\n')[:-1]
            data_iteration = np.array([float(i) for i in data_iteration])
            if len(data_iteration)==iterations:
                hits += 1
                data = np.append(data,data_iteration)
        if(os.path.isfile(s["name"]+"_"+str(i)+"/random/hypervolumes.txt")):            
            with open(s["name"]+"_"+str(i)+"/random/hypervolumes.txt") as f:
                data_iteration_random = f.read()
            data_iteration_random = data_iteration_random.split('\n')[:-1]
            data_iteration_random = np.array([float(i) for i in data_iteration_random])
            if len(data_iteration_random)==iterations:
                hits_random += 1
                data_random = np.append(data_random,data_iteration_random)    
    data= data.reshape((iterations,hits))
    data_random= data_random.reshape((iterations,hits_random))
    s["data_mean"] = np.mean(data,axis=0).tolist()
    s["data_random_mean"] = np.mean(data_random,axis=0).tolist()
    s["data_var"] = np.var(data,axis=0).tolist()
    s["data_random_var"] = np.var(data_random,axis=0).tolist()
    s["data_median"] = np.median(data,axis=0).tolist()
    s["data_random_median"] = np.median(data_random,axis=0).tolist()
    if(len(s["data_mean"])>0):
        with open(s["name"]+"_results.txt","w") as out:
            json.dump(s,out)