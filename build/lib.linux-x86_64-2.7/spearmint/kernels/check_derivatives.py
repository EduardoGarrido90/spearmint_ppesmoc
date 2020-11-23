import numpy as np
import kernel_utils

SQRT_3 = np.sqrt(3.0)
SQRT_5 = np.sqrt(5.0)
EPSILON = 1e-12

def matern52(x, x_prime, ls):
	r2 = np.abs(kernel_utils.dist2(ls, x, x_prime))
	r   = np.sqrt(r2)
	cov = (1.0 + SQRT_5*r + (5.0/3.0)*r2) * np.exp(-SQRT_5*r)
	return cov

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

#First result is the kernel wrt x and then wrt x_prime.
def compute_gradient_k_x_xprime(x, x_prime, ls):
	return np.array([dev_x_matern52(x, x_prime, ls), dev_x_prime_matern52(x, x_prime, ls)])

x1 = np.array([[4.0,1.8,2.0]])
x2 = np.array([[1.9,1.4,3.0]])
xx1 = np.array([[4.0,1.8,2.0], [23.2,11.8,22.0]])
xx2 = np.array([[1.9,1.4,3.0], [11.9, 19.4, 31.0]]) 
l=1.25
print("Partial derivative of x_prime approximation: " + str(check_partial_derivative_x_prime(x1, x2, l)))
print("Partial derivative of x_prime: " + str(dev_x_prime_matern52(x1, x2, l)))
print("Partial derivative of x approximation: " + str(check_partial_derivative_x(x1, x2, l)))
print("Partial derivative of x: " + str(dev_x_matern52(x1, x2, l)))
print("Gradient of k(x,x_prime): " + str(compute_gradient_k_x_xprime(xx1, xx2, l)))
