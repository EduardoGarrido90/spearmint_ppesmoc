#Bibeta in action.
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from scipy.stats import randint

def plot_1D_function(x, y, y_name='y'):
        ax = plt.subplot(111)
        ax.plot(x, y, y_name)
        plt.legend(loc='best')
        plt.show()

a_1 = np.linspace(0,10,100)
a_2 = np.linspace(0,10,100)
b_1 = np.linspace(0,10,100)
b_2 = np.linspace(0,10,100)
pi = np.linspace(0,1,10)
input_space = np.linspace(0,1,1000)
for i in range(5):
	pi_rvs = randint.rvs(0,10)
	a_1_rvs = randint.rvs(0,100)
	a_2_rvs = randint.rvs(0,100)
	b_1_rvs = randint.rvs(0,100)
	b_2_rvs = randint.rvs(0,100)
	bibeta_example_pdf = pi[pi_rvs]*beta.pdf(input_space,a_1[a_1_rvs],b_1[b_1_rvs]) + (1-pi[pi_rvs])*beta.pdf(input_space,a_2[a_2_rvs],b_2[b_2_rvs])
	bibeta_example_cdf = pi[pi_rvs]*beta.cdf(input_space,a_1[a_1_rvs],b_1[b_1_rvs]) + (1-pi[pi_rvs])*beta.cdf(input_space,a_2[a_2_rvs],b_2[b_2_rvs])
	ax = plt.subplot(111)
        ax.plot(input_space, pi[pi_rvs]*beta.pdf(input_space,a_1[a_1_rvs],b_1[b_1_rvs]), label="1 comp pdf")
        ax.plot(input_space, (1-pi[pi_rvs])*beta.pdf(input_space,a_2[a_2_rvs],b_2[b_2_rvs]), label="2 comp pdf")
        ax.plot(input_space, pi[pi_rvs]*beta.pdf(input_space,a_1[a_1_rvs],b_1[b_1_rvs]) + (1-pi[pi_rvs])*beta.pdf(input_space,a_2[a_2_rvs],b_2[b_2_rvs]), label="Mix pdf")
	plt.legend(loc='best')
        plt.show()
	ax = plt.subplot(111)
	ax.plot(input_space, pi[pi_rvs]*beta.cdf(input_space,a_1[a_1_rvs],b_1[b_1_rvs]), label="1 comp cdf")
	ax.plot(input_space, (1-pi[pi_rvs])*beta.cdf(input_space,a_2[a_2_rvs],b_2[b_2_rvs]), label="2 comp cdf")
	ax.plot(input_space, pi[pi_rvs]*beta.cdf(input_space,a_1[a_1_rvs],b_1[b_1_rvs]) + (1-pi[pi_rvs])*beta.cdf(input_space,a_2[a_2_rvs],b_2[b_2_rvs]), label="Mix cdf")
	plt.legend(loc='best')
        plt.show()

bibeta_example_pdf = pi[pi_rvs]*beta.pdf(input_space,a_1[a_1_rvs],b_1[b_1_rvs]) + (1-pi[pi_rvs])*beta.pdf(input_space,a_2[a_2_rvs],b_2[b_2_rvs])
bibeta_example_cdf = pi[pi_rvs]*beta.cdf(input_space,a_1[a_1_rvs],b_1[b_1_rvs]) + (1-pi[pi_rvs])*beta.cdf(input_space,a_2[a_2_rvs],b_2[b_2_rvs])

