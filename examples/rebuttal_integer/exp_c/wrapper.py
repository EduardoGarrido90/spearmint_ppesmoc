import numpy as np
from sklearn.externals import joblib
import copy
NUM_EXP = 1

def main(job_id, params):
        params = copy.deepcopy(params)
        np.random.seed(NUM_EXP)

	#The integer value lies in the range [1,5].
	#The integer values lie in the range [0,5].
        a_integer_variable = np.linspace(0, 1, 6)
        b_integer_variable = np.linspace(0, 1, 6)
        c_integer_variable = np.linspace(0, 1, 6)
        d_integer_variable = np.linspace(0, 1, 6)
        e_integer_variable = np.linspace(0, 1, 6)
        f_integer_variable = np.linspace(0, 1, 6)

        a_value = int(np.argmin(np.absolute(params['a'][ 0 ] - a_integer_variable))) 
        b_value = int(np.argmin(np.absolute(params['b'][ 0 ] - a_integer_variable))) 
        c_value = int(np.argmin(np.absolute(params['c'][ 0 ] - a_integer_variable))) 
        d_value = int(np.argmin(np.absolute(params['d'][ 0 ] - a_integer_variable))) 
        e_value = int(np.argmin(np.absolute(params['e'][ 0 ] - a_integer_variable))) 
        f_value = int(np.argmin(np.absolute(params['f'][ 0 ] - a_integer_variable))) 
	
	result = 1.0 / np.exp((a_value + 0.2*b_value - 0.1*c_value) / d_value) + 5.0*np.sin(e_value) - 0.2*f_value

        return {'score' : result}

if __name__ == "__main__":
        print(main(1, { u"a": np.array([0.2]), u"b": np.array([0.2]), u"c": np.array([0.8]), u"d": np.array([0.2]),
			u"e": np.array([0.2]), u"f": np.array([0.2])}))
