###Carla Johnston
###Last updated October 14, 2013

import numpy as np
import gb2library as lib
import matplotlib.pyplot as plt
from scipy import optimize as opt 

###This example covers how to used the distributions objects Gb2


##...........................................GB2..............................................
#Referring to the README doc the distribution paratemeters are 3 = a, 5000 = b, .5 = p, 10 = q
#a, p and q are shape parameters which affect tail density and skewness. b is a scale parameter

distGb2 = lib.Gb2(3, 5000, .5, 10)
#listing our distribution parameters
distGb2.a
distGb2.b
distGb2.p
distGb2.q

#first four mean centered moments
print distGb2.mean(), "Mean"
print distGb2.std(), "Standard Deviation"
print distGb2.skew(), "Skewness"
print distGb2.kurt(), "Kurtosis"

#Third moment (not mean centered)
print distGb2.mom(3), "Third moment"

# x_range = np.linspace(.1, 10000, 500)
# pdf_points = distGb2.pdf(x_range)
# cdf_points = distGb2.cdf(x_range)
# plt.plot(x_range, pdf_points)
# plt.title('Gb2 Pdf')
# plt.show()
# plt.plot(x_range, cdf_points)
# plt.title('Gb2 Cdf')
# plt.show()


#......................how to perform MLE to fit distribution to data................

sal = np.loadtxt("income_data.csv", delimiter = ",")
#Restricting our data to a reasonable range
sal = sal[sal>0] 
sal = sal[sal < 80000] 

#Since we are trying to estimate the paratemeters, just initate the Gb2 object with zeros
newdistGb2 = lib.Gb2(0, 0, 0, 0) 

gb2loglike = lambda x: newdistGb2.loglike(sal, x, sign = -1)
int_guess = [1, 500000, 1, 1000]

#In order to restrict the optimization procedure to estimate only 
#positive values of a, b, p, q, the loglikelihood method takes in the natural logs 
#of our actual parameters and then exponentiates them within the method function. 
Gb2_logparam = opt.minimize(gb2loglike, np.log(int_guess), method = 'Powell', tol = 1e-5, options = ({'maxiter': 50000, 'maxfev' : 50000}))
print np.exp(Gb2_logparam.x), "Estimated parameters, a, b, p, q"

#reset your parameters
Gb2_param = list(np.exp(Gb2_logparam.x)) 
newdistGb2 = lib.Gb2(*Gb2_param)
