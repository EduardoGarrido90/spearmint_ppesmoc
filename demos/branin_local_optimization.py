import scipy.optimize as spo
import numpy as np
import math


def evaluate(job_id, params):

    x = params['X']
    y = params['Y']

    print 'Evaluating at (%f, %f)' % (x, y)

    # if x < 0 or x > 5.0 or y > 5.0:
    #     return np.nan
    # Feasible region: x in [0,5] and y in [0,5]

    obj1 = float(np.square(y - (5.1/(4*np.square(math.pi)))*np.square(x) + (5/math.pi)*x- 6) + 10*(1-(1./(8*math.pi)))*np.cos(x) + 10)

    obj2 = -obj1

    return {
        "branin_1"       : obj1,
        "branin_2"       : obj2,
    }

def branin(X):
	
	x = X[0]
	y = X[1]

	obj1 = float(np.square(y - (5.1/(4*np.square(math.pi)))*np.square(x) + (5/math.pi)*x- 6) + 10*(1-(1./(8*math.pi)))*np.cos(x) + 10)

	return obj1

TOLERANCE = 1e-4
def test_optimization_algorithm(x_best, y_best, f):
	best_sol = f(x_best)
	print "Best solution is..."
	print best_sol
	print "Near solution..."
	sol_1 = f(x_best-TOLERANCE)
	print sol_1
	print "Near solution..."
	sol_2 = f(x_best+TOLERANCE)
	print sol_2
	print "Testing if the optimization algorithm has done a good work..."
	assert best_sol < sol_1 and best_sol < sol_2
	print "TEST OK" 

if __name__ == '__main__':
	print "Branin Local Optimization"
	def f(X):
		return branin(X)			
	
	def f_x(x):
		return branin(np.array([x,7.5]))

	init_point = np.array([2.5,7.5])
	print "Init point result:"
	print f(init_point)
	print "Full Optimization result"
	x_opt, y_opt, info = spo.fmin_l_bfgs_b(f, init_point, bounds=np.array([[-5,10], [0,15]]), disp=1, approx_grad=True)
	print y_opt
	test_optimization_algorithm(x_opt, y_opt, f)
	print "Partial Optimization result"
	x_opt, y_opt, info = spo.fmin_l_bfgs_b(f_x, init_point[0], bounds=np.array([[-5,10]]), disp=1, approx_grad=True)
	print y_opt
	test_optimization_algorithm(x_opt, y_opt, f_x)
