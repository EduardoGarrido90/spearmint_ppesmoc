from autograd import grad
from autograd import numpy

def test(x):
    return np.sqrt(x)**np.power(2*x)

dev_test = grad(test)
print(dev_test(2))
