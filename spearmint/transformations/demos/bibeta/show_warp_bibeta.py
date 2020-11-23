#Bibeta in action.
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from scipy.stats import randint

def plot_1D_function(x, y, y_name='y'):
        ax = plt.subplot(111)
        ax.plot(x, y, label=y_name)
        plt.legend(loc='best')
	plt.title(y_name)
        plt.show()

def fixed_objective_function(x, a=25):
        target = np.sin(a*x)
        t_max = np.max(target)
        t_min = np.min(target)
        return (target-t_min)/(t_max-t_min)

a_1 = np.linspace(0,10,100)
a_2 = np.linspace(0,10,100)
b_1 = np.linspace(0,10,100)
b_2 = np.linspace(0,10,100)
pi = np.linspace(0,1,10)
input_space = np.linspace(0,1,1000)

pi_rvs = randint.rvs(0,10)
a_1_rvs = randint.rvs(0,100)
a_2_rvs = randint.rvs(0,100)
b_1_rvs = randint.rvs(0,100)
b_2_rvs = randint.rvs(0,100)

a = a_1[a_1_rvs]
b = b_1[b_1_rvs]
a1 = a_1[a_1_rvs]
a2 = a_2[a_2_rvs]
b1 = b_1[b_1_rvs]
b2 = b_2[b_2_rvs]
p = pi[pi_rvs]

beta_cdf = beta.cdf(input_space,a,b)
bibeta_cdf = p*beta.cdf(input_space,a1,b1) + (1-p)*beta.cdf(input_space,a2,b2)

plot_1D_function(input_space, input_space, 'Input Space')
plot_1D_function(input_space, beta_cdf, 'Beta cdf')
plot_1D_function(input_space, bibeta_cdf, 'Bibeta cdf')
plot_1D_function(input_space, fixed_objective_function(input_space), 'Objective')
plot_1D_function(input_space, fixed_objective_function(beta_cdf), 'Warped Objective Beta')
plot_1D_function(input_space, fixed_objective_function(bibeta_cdf), 'Warped Objective Bibeta')
