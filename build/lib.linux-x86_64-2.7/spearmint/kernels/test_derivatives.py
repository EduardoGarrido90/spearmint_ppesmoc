import numpy as np
import kernel_utils

SQRT_3 = np.sqrt(3.0)
SQRT_5 = np.sqrt(5.0)
EPSILON = 1e-5

def matern52_r(r):
	cov = (1.0 + SQRT_5*r + (5.0/3.0)*np.power(r,2)) * np.exp(-SQRT_5*r)
	return cov

def matern52_r2(r2):
	cov = (1.0 + np.sqrt(5*r2) + (5.0/3.0)*r2) * np.exp(-np.sqrt(5*r2))
	return cov

################
# OK
def single(r2):
	return (1.0 + np.sqrt(5*r2) + (5.0/3.0)*r2)

def dev_single(r2):
	return ((5.0/3.0) + SQRT_5 / (2.0*np.sqrt(r2)))
################

################
# OK
def single2(r2):
	return np.exp(-np.sqrt(5.0)*np.sqrt(r2))

def dev_single2(r2):
	return np.exp(-np.sqrt(5.0)*np.sqrt(r2))*(-np.sqrt(5.0)/(2.0*np.sqrt(r2)))
################

def exponential_component(r2):
	return dev_single2(r2)

################
# OK
def partial_edu_r2(r2):
	return (1.0 + np.sqrt(5*r2) + (5.0/3.0)*r2) * exponential_component(r2) + ((5.0/3.0) + SQRT_5 / (2.0*np.sqrt(r2))) * np.exp(-np.sqrt(5*r2))
################

################
# OK
def partial_edu_r3(r2):
        return (1.0 + np.sqrt(5*r2) + (5.0/3.0)*r2) * exponential_component(r2) + ((5.0/3.0) + SQRT_5 / (2.0*np.sqrt(r2))) * np.exp(-np.sqrt(5*r2))
################

#Puesto el signo, resulta ser que es equivalente.
def partial_spearmint(r):
	return (5.0/6.0)*np.exp(-SQRT_5*r)*(1 + SQRT_5*r)

def partial_edu(r):
	return ((10*np.power(r,3)+6*SQRT_5*np.power(r,2)+3*SQRT_5+16*r)/(6.0*r))*np.exp(-SQRT_5*r)

def partial_edu_2(r):
        return (1+SQRT_5*r+5.0/3.0*np.power(r,2)+SQRT_5/(2*r)+5/3.0)*np.exp(-SQRT_5*r)

def partial_edu_3(r):
	return (1.0 + SQRT_5*r + (5.0/3.0)*np.power(r,2)) * np.exp(-SQRT_5*r) + (SQRT_5/(2.0*r)+(5.0/3.0)) * np.exp(-SQRT_5*r)

def matern52(x, x_prime, ls):
	r2 = np.abs(kernel_utils.dist2(ls, x, x_prime))
	r   = np.sqrt(r2)
	cov = (1.0 + SQRT_5*r + (5.0/3.0)*r2) * np.exp(-SQRT_5*r)
	return cov

def dev_x_prime_edu(x, x_prime, ls):
	r2 = np.abs(kernel_utils.dist2(ls, x, x_prime))
        r       = np.sqrt(r2)
	grad_r2 = partial_edu_r2(r2)
	#Minus is added because grad_dist2 returns the gradient wrt x, which is the gradient of -1*x_prime.
	return - grad_r2[:,:,np.newaxis] * kernel_utils.grad_dist2(ls, x, x_prime)

def dev_x_prime_matern52(x, x_prime, ls):
	r2 = np.abs(kernel_utils.dist2(ls, x, x_prime))
	r       = np.sqrt(r2)
	grad_r2 = (5.0/6.0)*np.exp(-SQRT_5*r)*(1 + SQRT_5*r)
	return grad_r2[:,:,np.newaxis] * kernel_utils.grad_dist2(ls, x, x_prime)

def dev_x_matern52(x, x_prime, ls):
	return - dev_x_prime_matern52(x, x_prime, ls)

def check_partial_derivative_x_prime(x, x_prime, ls):
	partial_derivatives = np.zeros([x_prime.shape[1]])
	for i in range(x_prime.shape[1]):
		x_prime_plus_h = np.copy(x_prime)
		x_prime_plus_h[0][i] += EPSILON
		partial_derivatives[i] = (matern52(x, x_prime_plus_h, ls) - matern52(x, x_prime, ls)) / EPSILON
	return partial_derivatives

def check_partial_derivative_x(x, x_prime, ls):
        partial_derivatives = np.zeros([x.shape[1]])
        for i in range(x.shape[1]):
                x_plus_h = np.copy(x)
                x_plus_h[0][i] += EPSILON
                partial_derivatives[i] = (matern52(x_plus_h, x_prime, ls) - matern52(x, x_prime, ls)) / EPSILON
        return partial_derivatives

def check_partial_derivative_r_matern(r):
        return (matern52_r(r+EPSILON) - matern52_r(r)) / EPSILON

def check_partial_derivative_r2_matern(r):
        return (matern52_r2(r+EPSILON) - matern52_r2(r)) / EPSILON

def check_partial_derivative_single(r):
        return (single(r+EPSILON) - single(r)) / EPSILON

def check_partial_derivative_single2(r):
        return (single2(r+EPSILON) - single2(r)) / EPSILON

#First result is the kernel wrt x and then wrt x_prime.
def compute_gradient_k_x_xprime(x, x_prime, ls):
	return np.array([dev_x_matern52(x, x_prime, ls), dev_x_prime_matern52(x, x_prime, ls)])

x1 = np.array([[4.0,1.8,2.0]])
x2 = np.array([[1.9,1.4,3.0]])
#xx1 = np.array([[4.0,1.8,2.0], [23.2,11.8,22.0]])
#xx2 = np.array([[1.9,1.4,3.0], [11.9, 19.4, 31.0]]) 
l=1.25
print("Partial derivative of x_prime approximation: " + str(check_partial_derivative_x_prime(x1, x2, l)))
print("Partial derivative of x_prime: " + str(dev_x_prime_matern52(x1, x2, l)))
print("Partial derivative of x_prime edu: " + str(dev_x_prime_edu(x1, x2, l)))
#print("Partial derivative of x approximation: " + str(check_partial_derivative_x(x1, x2, l)))
#print("Partial derivative of x: " + str(dev_x_matern52(x1, x2, l)))
#print("Gradient of k(x,x_prime): " + str(compute_gradient_k_x_xprime(xx1, xx2, l)))

'''
r=3.0
print("Matern_r, result: " + str(matern52_r(r)))
print("Matern_r2, result: " + str(matern52_r2(np.power(r,2))))
print("Partial spearmint, result: " + str(partial_spearmint(r)))
print("Partial edu, result: " + str(partial_edu_3(r)))
print("Derivative matern_r, result: " + str(check_partial_derivative_r_matern(r)))
'''
'''
r2=3.0
print("Partial edu, result: " + str(partial_edu_r2(r2)))
print("Partial spearmint, result: " + str(-partial_spearmint(np.sqrt(r2))))
print("Derivative matern_r, result: " + str(check_partial_derivative_r2_matern(r2)))
'''
'''
print("Single, result: " + str(dev_single(r2)))
print("Single derivative, result: " + str(check_partial_derivative_single(r2)))
print("Single2, result: " + str(dev_single2(r2)))
print("Single2 derivative, result: " + str(check_partial_derivative_single2(r2)))
'''
