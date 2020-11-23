#Bibeta.
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from scipy.stats import norm
from scipy.stats import uniform
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from scipy.signal import argrelextrema

def fixed_objective_function(x, a=25):
	target = np.sin(a*x)
        t_max = np.max(target)
        t_min = np.min(target)	
        return (target-t_min)/(t_max-t_min)

def b_warp_fixed_objective_function(x, times=3):
        y = fixed_objective_function(x)
	fixed = y
	for i in range(times):
		x, y = apply_beta_warp(x, y)

	####################################
	#Making things complicated for beta.
	y = fixed_objective_function(norm.cdf(x,0,1))
	x = norm.cdf(x,0,1)
	####################################

	return x, y, fixed

def apply_beta_warp(x, y):
	a, b = uniform(), uniform()
        extrema_max_obj = argrelextrema(y, np.greater)[0].shape[0]
        extrema_min_obj = argrelextrema(y, np.less)[0].shape[0]
        discovered = False
        counter = 0
        while not discovered:
                a_param = a.rvs()
                b_param = b.rvs()
		warped_input = beta.cdf(x, a_param, b_param)
                warped_objective = fixed_objective_function(beta.cdf(x, a_param, b_param))
                e_w_max = argrelextrema(warped_objective, np.greater)[0].shape[0]
                e_w_min = argrelextrema(warped_objective, np.less)[0].shape[0]
                counter+=1
                print counter
                if e_w_max == extrema_max_obj and e_w_min == extrema_min_obj:
                	discovered = True
			print "Warping with a = " + str(a_param) + " and b = " + str(b_param)
        return warped_input, warped_objective
	
def plot_1D_function(x, y, y_name='y'):
	ax = plt.subplot(111)
	ax.plot(x, y, y_name)
	plt.legend(loc='best')
	plt.show()

def return_best_beta_for_dewarp(input_space, objective, y, warped_space, wrt_orig=True, criterion='distance'):
	a_grid = np.linspace(0.1,10,100)
	b_grid = np.linspace(0.1,10,100)
	if wrt_orig==True:
		if criterion == 'distance':
			min_distance = 1e10
		else:	
			return None
		#extrema_max_obj = argrelextrema(objective, np.greater)[0].shape[0]
                #extrema_min_obj = argrelextrema(objective, np.less)[0].shape[0]
		counter = 0
		a_min_value = -1
		b_min_value = -1
		for a_value in a_grid:
        		for b_value in b_grid:
                		wo = fixed_objective_function(beta.cdf(warped_space, a_value, b_value))
                		#extrema_max_wo = argrelextrema(wo, np.greater)[0].shape[0]
                		#extrema_min_wo = argrelextrema(wo, np.less)[0].shape[0]
				if criterion == 'distance':
                			distance = np.sum(np.abs(objective-wo))
				else:
					return None
		                #if distance < min_distance and extrema_max_wo == extrema_max_obj and extrema_min_wo == extrema_min_obj:
				if distance < min_distance:
                		        min_distance = distance
		                        a_min_value = a_value
		                        b_min_value = b_value
                		counter+=1
		                print counter
	else:
		return None #Not done.
	return a_min_value, b_min_value	, min_distance

def return_best_beta_mixture_for_dewarp(input_space, objective, y, warped_space, wrt_orig=True, criterion='distance'):
        a_1_grid = np.linspace(0.1,10,10)
        b_1_grid = np.linspace(0.1,10,10)
        a_2_grid = np.linspace(0.1,10,10)
        b_2_grid = np.linspace(0.1,10,10)
        pi_grid = np.linspace(0,1,10)
        if wrt_orig==True:
                if criterion == 'distance':
                        min_distance = 1e10
                else:
                        return None
                counter = 0
                a_1_min_value = -1
                b_1_min_value = -1
		a_2_min_value = -1
                b_2_min_value = -1
		pi_min_value = -1
                for a_1_value in a_1_grid:
                        for b_1_value in b_1_grid:
                        	for a_2_value in a_2_grid:
                        		for b_2_value in b_2_grid:
			                        for pi_value in pi_grid:
                                			wo = fixed_objective_function(pi_value*beta.cdf(warped_space, a_1_value, b_1_value)+\
								(1.0-pi_value)*beta.cdf(warped_space, a_2_value, b_2_value))
                                			if criterion == 'distance':
			                                        distance = np.sum(np.abs(objective-wo))
                        			        else:
			                                        return None
                        			        #if distance < min_distance and extrema_max_wo == extrema_max_obj and extrema_min_wo == extrema_min_obj:
			                                if distance < min_distance:
                        			                min_distance = distance
			                                        a_1_min_value = a_1_value
			                                        b_1_min_value = b_1_value
                        			                a_2_min_value = a_2_value
			                                        b_2_min_value = b_2_value
			                                        pi_min_value = pi_value
                        			        counter+=1
			                                print counter
        else:
                return None #Not done.
        return a_1_min_value, b_1_min_value , a_2_min_value, b_2_min_value, pi_min_value, min_distance

input_space = np.linspace( 0, 1, 1000 )
warped_space, warped_objective, objective = b_warp_fixed_objective_function( input_space, times=2 )
plot_1D_function( input_space, warped_objective )
a_dewarp, b_dewarp, distance = return_best_beta_for_dewarp(input_space, objective, warped_objective, warped_space)
a_1_dewarp, b_1_dewarp, a_2_dewarp, b_2_dewarp, pi_dewarp, distance = \
	return_best_beta_mixture_for_dewarp(input_space, objective, warped_objective, warped_space)
print "Dewarping with a = " + str(a_dewarp) + " and b = " + str(b_dewarp) 

ax = plt.subplot(111)
ax.plot(input_space, fixed_objective_function(beta.cdf(warped_space, a_dewarp, b_dewarp)), label='Dewarped Beta Objective')
ax.plot(input_space, fixed_objective_function(pi_dewarp*beta.cdf(warped_space, a_1_dewarp, b_1_dewarp))+ \
	(1-pi_dewarp)*beta.cdf(warped_space, a_2_dewarp, b_2_dewarp), label='Dewarped Bibeta Objective')
ax.plot(input_space, warped_objective, label="Warped Objective")
ax.plot(input_space, objective, label="Objective")
plt.legend(loc='best')
plt.show()

print distance
