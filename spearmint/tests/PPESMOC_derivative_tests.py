import sys
import numpy as np
from spearmint.acquisition_functions.parallel_predictive_entropy_search_multiobjective_constraints import PPESMOC 
EPSILON = 1e-08
NUM_DIMS = 2

#Shape inv_Bxx: n*n, dev_Bxx_xij: n*n.
def test_dev_Bxx_inv(inv_Bxx, dev_Bxx_xij):
                return

#Just some toy examples to test the derivative definitions and test
#of the test methods.
def test1_generic(value, tt_func):
	return (tt_func(value+EPSILON) - tt_func(value)) / EPSILON

def test2_generic(value, tt_func):
	return (tt_func(value+EPSILON) - tt_func(value-EPSILON)) / (2*EPSILON)

def test2_dev_x_2(x):
	return (x_2(x+EPSILON) - x_2(x-EPSILON)) / (2*EPSILON)

def test1_dev_x_2(x):
	return (x_2(x+EPSILON) - x_2(x)) / EPSILON

def dev_x_2(x):
	return x*2.0

def x_2(x):
	return np.power(x,2.0)

def test1_dev_x2_2(x):
	y = np.copy(x)
	y[0] += EPSILON
	return (x2_2(y) - x2_2(x)) / EPSILON

def dev_x2_2(x):
	return 2*x[0]*np.power(x[1],2)

def x2_2(x):
	return np.power(x[0],2) * np.power(x[1],2)

#X test points are be given and X test points that have X_ij+e point and its matrices will also be given.
#Two sets of covariances matrices are returned.
#TODO More mock data remain to be put!
#This is OK, but it would be nice if this is generated automatically from PPESMOC having the X points.
#mocotoy_2 task data of iteration 1 with 3 batch points of mocotoy_ppesmoc experiment.
def mock1_covariances_matrices():
	covariances_matrices = dict()
	covariances_matrices["n_test"] = 3
	covariances_matrices["n_obs"] = 3
	covariances_matrices["n_pset"] = 2
	covariances_matrices["dev_kbt_xij"] = np.array([np.array([np.array([2.25129382e+00, -2.04184593e+00]), \
		np.array([-1.19234869e+00, 1.33272810e+00]), np.array([7.13283497e-05, -1.12842549e-03])]), \
		np.array([np.array([ 2.19348238e+00, 2.17469712e+00]), np.array([-7.23251281e-01, 4.16985614e+00]), \
		np.array([2.46738651e-04, -3.65842883e-03])])])
	covariances_matrices["dev_kot_xij"] = np.array([np.array([np.array([1.07336765e-03, 1.13447279e-02]), \
		np.array([1.30178998e-05, 5.79379436e-03]), np.array([1.29920132e+00, -2.41532494e+00])]), \
		np.array([np.array([3.61702273e-02, 1.06115933e-01]), np.array([1.71770637e-02, 9.92146436e-02]), \
		np.array([6.94183948e-02, -1.68096844e-01])]), np.array([np.array([1.02833971e+00, 1.21329511e+00]), \
		np.array([-1.89071801e+00, 2.91389766e+00]), np.array([6.87662206e-05, -2.65274968e-03])])])
	covariances_matrices["dev_ktb_xij"] = np.array([np.array([np.array([2.25129382e+00,  -2.04184593e+00]), \
		np.array([2.19348238e+00,   2.17469712e+00])]), np.array([np.array([-1.19234869e+00,   1.33272810e+00]), \
		np.array([-7.23251281e-01,   4.16985614e+00])]), np.array([np.array([7.13283497e-05,  -1.12842549e-03]), \
		np.array([2.46738651e-04,  -3.65842883e-03])])])
	covariances_matrices["dev_kto_xij"] = np.array([np.array([np.array([1.07336765e-03,   1.13447279e-02]), \
		np.array([3.61702273e-02,   1.06115933e-01]), np.array([1.02833971e+00,   1.21329511e+00])]), \
		np.array([np.array([1.30178998e-05,   5.79379436e-03]), np.array([1.71770637e-02,   9.92146436e-02]), \
		np.array([-1.89071801e+00,   2.91389766e+00])]), np.array([np.array([1.29920132e+00,  -2.41532494e+00]), \
		np.array([6.94183948e-02,  -1.68096844e-01]), np.array([6.87662206e-05,  -2.65274968e-03])])])
	covariances_matrices["dev_ktt_xij"] = np.array([np.array([np.array([0.00000000e+00,   0.00000000e+00]), \
		np.array([-2.18691731e+00,   2.10027688e+00]), np.array([1.93957336e-05,  -2.16261397e-03])]), \
		np.array([np.array([2.18691731e+00,  -2.10027688e+00]),np.array([0.00000000e+00,   0.00000000e+00]), \
		np.array([6.76355477e-05,  -8.44252711e-04])]), np.array([np.array([-1.93957336e-05,   2.16261397e-03]), \
		np.array([-6.76355477e-05,   8.44252711e-04]),np.array([0.00000000e+00,   0.00000000e+00])])])
	covariances_matrices["inv_noisied_Koo"] = np.array([np.array([3.33333439e+05,  -1.42293674e-02,   7.38945082e-05]), \
		np.array([-1.42293674e-02,   3.33333439e+05,  -1.41534250e-03]), \
		np.array([7.38945082e-05,  -1.41534250e-03,   3.33333437e+05])])	
	covariances_matrices["Kbo"] = np.array([np.array([-1.56551574e-10,  -1.29393824e-09,   7.81671833e-07]), \
		np.array([-1.00259525e-09,   1.31635730e-08,   8.68797285e-07])])
	covariances_matrices["Kob"] = np.array([np.array([-1.56551574e-10,  -1.00259525e-09]), \
		np.array([-1.29393824e-09,   1.31635730e-08]), \
		np.array([7.81671833e-07,   8.68797285e-07])])
	covariances_matrices["Kbo_inv_Koo"] = np.array([np.array([ -4.69655258e-16,  -3.88180357e-15,   2.34501477e-12]), \
		np.array([ -3.00778370e-15,   3.94907174e-14,   2.60639105e-12])])
	covariances_matrices["Bxx"] = np.array([np.array([0.41733008,  0.08369502]), np.array([0.08369502,  0.26252377])])
	covariances_matrices["inv_Bxx"] = np.array([np.array([2.55985366, -0.81610514]), np.array([-0.81610514,  4.06936078])])

	#An EPSILON of 1e-08 is added to test point 1 dimension 1. (Starting with index 0).
	#These are the resultant matrices.
	covariances_matrices_e = dict()
	covariances_matrices_e["n_test"] = 3
        covariances_matrices_e["n_obs"] = 3
        covariances_matrices_e["n_pset"] = 2

	#These structures are not affected by the variation of EPSILON.
	covariances_matrices_e["Kob"] = np.copy(covariances_matrices["Kob"])
	covariances_matrices_e["Kbo"] = np.copy(covariances_matrices["Kbo"])
	covariances_matrices_e["Bxx"] = np.copy(covariances_matrices["Bxx"])
	covariances_matrices_e["inv_Bxx"] = np.copy(covariances_matrices["inv_Bxx"])
	covariances_matrices_e["inv_noisied_Koo"] = np.copy(covariances_matrices["inv_noisied_Koo"])

	#Now, these are affected.
	covariances_matrices_e["dev_kbt_xij"] = np.array([np.array([np.array([  2.25129382e+00,  -2.04184593e+00]), \
        	np.array([ -1.19234872e+00,   1.33272746e+00]), np.array([  7.13283497e-05,  -1.12842549e-03])]), \
       		np.array([np.array([  2.19348238e+00,   2.17469712e+00]), np.array([ -7.23251345e-01,   4.16985614e+00]), \
        	np.array([  2.46738651e-04,  -3.65842883e-03])])])
	covariances_matrices_e["dev_kot_xij"] = np.array([np.array([np.array([  1.07336765e-03,   1.13447279e-02]), \
        	np.array([  1.30179015e-05,   5.79379507e-03]), np.array([  1.29920132e+00,  -2.41532494e+00])]), \
		np.array([np.array([  3.61702273e-02,   1.06115933e-01]), np.array([  1.71770659e-02,   9.92146541e-02]), \
        	np.array([  6.94183948e-02,  -1.68096844e-01])]), np.array([np.array([  1.02833971e+00,   1.21329511e+00]), \
        	np.array([ -1.89071813e+00,   2.91389749e+00]), np.array([  6.87662206e-05,  -2.65274968e-03])])])
	covariances_matrices_e["dev_ktb_xij"] = np.array([np.array([np.array([  2.25129382e+00,  -2.04184593e+00]), \
        	np.array([  2.19348238e+00,   2.17469712e+00])]), np.array([np.array([ -1.19234872e+00,   1.33272746e+00]), \
        	np.array([ -7.23251345e-01,   4.16985614e+00])]), np.array([np.array([  7.13283497e-05,  -1.12842549e-03]), \
        	np.array([  2.46738651e-04,  -3.65842883e-03])])])
	covariances_matrices_e["dev_kto_xij"] = np.array([np.array([np.array([  1.07336765e-03,   1.13447279e-02]), \
        	np.array([  3.61702273e-02,   1.06115933e-01]), np.array([  1.02833971e+00,   1.21329511e+00])]), \
		np.array([np.array([  1.30179015e-05,   5.79379507e-03]), np.array([  1.71770659e-02,   9.92146541e-02]), \
        	np.array([ -1.89071813e+00,   2.91389749e+00])]), np.array([np.array([  1.29920132e+00,  -2.41532494e+00]), \
        	np.array([  6.94183948e-02,  -1.68096844e-01]), np.array([  6.87662206e-05,  -2.65274968e-03])])])
	covariances_matrices_e["dev_ktt_xij"] = np.array([np.array([np.array([  0.00000000e+00,   0.00000000e+00]), \
        	np.array([ -2.18691741e+00,   2.10027667e+00]), np.array([  1.93957336e-05,  -2.16261397e-03])]), \
		np.array([np.array([  2.18691741e+00,  -2.10027667e+00]), np.array([  0.00000000e+00,   0.00000000e+00]), \
        	np.array([  6.76355568e-05,  -8.44252816e-04])]), np.array([np.array([ -1.93957336e-05,   2.16261397e-03]), \
        	np.array([ -6.76355568e-05,   8.44252816e-04]), np.array([  0.00000000e+00,   0.00000000e+00])])])
	covariances_matrices_e["Kbo_inv_Koo"] = np.copy(covariances_matrices["Kbo_inv_Koo"])

	return covariances_matrices, covariances_matrices_e
		
if __name__ == "__main__":
	
	#Toy tests of derivatives in order to see how to build the PPESMOC scenario.
	print "hi"
	value = 5.0
	print x_2(value)
	print dev_x_2(value)
	print test1_dev_x_2(value)
	print test2_dev_x_2(value)
	print test1_generic(value, x_2)
	print test2_generic(value, x_2)
	value = np.array([5.0,3.0])
	print x2_2(value)
	print dev_x2_2(value)
	print test1_dev_x2_2(value) #Everything is OK.

	# --- PPESMOC test battery ---
	print "Initiating PPESMOC tests"
	ppesmoc =  PPESMOC(NUM_DIMS)
	ppesmoc.mock_test_message()
	covariances_matrices, covariances_matrices_e = mock1_covariances_matrices() #e (1e-08) is added to tp=1, dp=1.
	tp = 1
	dp = 1

	#Now you can test any function of PPESMOC given that mock data is available and verify the derivatives.
	#Trivial one:
	assert np.array_equal(ppesmoc.dev_Bxx_inv(covariances_matrices, None, 1, 0), np.zeros((2, 2))), "dev_Bxx_inv KO"
	
	#Derivative definition: (tt_func(value+EPSILON) - tt_func(value)) / EPSILON
	Bxy = np.array([np.array([ -1.46195221e-02,   4.90261519e-01,   1.01204751e-05]), \
		np.array([ -9.38842275e-02,   1.09186334e-01,   2.74541039e-04])])
	Bxy_e = np.array([np.array([ -1.46195221e-02,   4.90261510e-01,   1.01204751e-05]), \
       		np.array([ -9.38842275e-02,   1.09186350e-01,   2.74541039e-04])])

	print "Bxy"
	print Bxy
	print "Bxy_e"
	print Bxy_e	
	#How am I supposed to validate this derivative?. Not sure, I have some ideas.
	#Maybe I have to compute the matrix without the derivatives of the K matrices.
	#This is supposed to give Bxy, put above, because it is a function of X_test.
	#Then, I have to compute the expression of below without the dev, it would be
	#the construction of the B block Bxy.
	dev_Bxy_xij_value = (Bxy_e - Bxy) / EPSILON
	print "Derivative definition value"
	print dev_Bxy_xij_value
	print "Computed Derivative value"
	print ppesmoc.dev_Bxy_xij(covariances_matrices_e, None, tp, dp)
	print "End"
