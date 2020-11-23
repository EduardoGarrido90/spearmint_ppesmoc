import numpy as np
import matplotlib.pyplot as plt
from gb2library import Gb1
from scipy.stats import beta
from scipy.stats import beta
from scipy.stats import norm
from scipy.stats import uniform
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from scipy.signal import argrelextrema

#Grid de betas para deswarpear una beta warpeada.
def fixed_objective_function(x):
        target = np.sin(x*50)
        t_max = np.max(target)
        t_min = np.min(target)
        return (target-t_min)/(t_max-t_min)

input_space = np.linspace(0, 1, 1000)

a, b = uniform(), uniform()
#objective = sample_objective_function(input_space)
objective = fixed_objective_function(input_space)

#The stationary function must have the same number of extremas in order to be valid.
#The other solutions are not feasible.
extrema_max_obj = argrelextrema(objective, np.greater)[0].shape[0]
extrema_min_obj = argrelextrema(objective, np.less)[0].shape[0]

#Beta params
discovered = False
counter = 0
a_warped = -1
b_warped = -1
while not discovered:
        a_param = a.rvs()
        b_param = b.rvs()
        beta_fun = beta.cdf(input_space, a_param, b_param)
        warped_objective = fixed_objective_function(beta.cdf(input_space, a_param, b_param))
        e_w_max = argrelextrema(warped_objective, np.greater)[0].shape[0]
        e_w_min = argrelextrema(warped_objective, np.less)[0].shape[0]
        counter+=1
        print counter
        if e_w_max == extrema_max_obj and e_w_min == extrema_min_obj:
                discovered = True
                a_warped = a_param
                b_warped = b_param

beta_warped_space = beta.cdf(input_space, a_warped, b_warped)
final_space = norm.cdf(beta_warped_space, 0, 1)

#Now we transform this space with a normal cdf.
#Ahora sabes que los parametros a y b son los que warpean. Ya puedes hacer el grid para deswarpear.
print 'Starting grid search to look for the best beta params to squash the input params in order to get a stationary function'
print 'The obj function is to maximize KPSS s.t. having the same number of local minimas and maximas'
a_grid = np.linspace(0.1,10,200)
b_grid = np.linspace(0.1,10,200)

a_dewarp = -1
b_dewarp = -1
min_distance = 100000
counter = 0

for a_value in a_grid:
        for b_value in b_grid:
		wo = fixed_objective_function(beta.cdf(final_space,a_value,b_value))
                distance = np.sum(np.abs(objective-wo))
		if distance < min_distance:
        		min_distance = distance
	                a_dewarp = a_value
                	b_dewarp = b_value
		counter+=1
             	print counter

#Plotting.
ax = plt.subplot(111)
wo = fixed_objective_function(beta.cdf(final_space,a_dewarp,b_dewarp))
beta_warp = beta.cdf(input_space, a_warped, b_warped)
beta_dewarp = beta.cdf(input_space,a_dewarp,b_dewarp)
norm_cdf = norm.cdf(input_space, 0, 1)
warped_objective = fixed_objective_function(norm.cdf(beta.cdf(input_space, a_warped, b_warped),0,1))
ax.plot(input_space, objective, label='Objective')
ax.plot(input_space, warped_objective, label='Warped Objective')
ax.plot(input_space, wo, label='Dewarped Objective')
ax.plot(input_space, beta_warp, label='Beta Warp')
ax.plot(input_space, norm_cdf, label='Norm std Warp')
ax.plot(input_space, beta_dewarp, label='Beta Dewarp')
plt.legend(loc='best')
plt.show()
print 'Final distance is ' + str(distance)
