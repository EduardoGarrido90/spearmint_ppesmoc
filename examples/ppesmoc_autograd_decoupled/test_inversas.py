import scipy.linalg   as spla
import autograd.numpy as np

#Generating a PSD matrix.
matrixSize = 10 
A = np.random.rand(matrixSize, matrixSize)
B = np.dot(A,A.transpose())
print 'initial matrix\n'
print B 
print '\n' 

#Generating the inverse with numpy.
print 'numpy' + '\n'
inv_b_n = np.linalg.inv(B)
print inv_b_n
print '\n' 

#Generating the inverse with scipy (cholesky decomp. proc.)
print 'scipy' + '\n'
int_inv_b = spla.cholesky(B)
inv_b_s = spla.cho_solve((int_inv_b, False), np.eye(int_inv_b.shape[0]))
print inv_b_s 
print '\n' 

print 'diff' + '\n'
print np.abs(inv_b_n - inv_b_s)
