import numpy as np
from sklearn.externals import joblib
import copy
NUM_EXP = 1

def main(job_id, params):
        params = copy.deepcopy(params)
        np.random.seed(NUM_EXP)

        #Parsing parameters.
	cv_b = ["f", "g", "h", "i", "j"]
        b_value = cv_b[np.argmax(params['b_cat'])] #Cat value hold the selected value of the transformation.

	#The integer value lies in the range [1,5].
        iv_c = np.linspace(0, 1, 4)
        c_value = int(np.argmin(np.absolute(params['c_int'][ 0 ] - iv_c))) + 1 #Int value hold the selected value of the transformation
	
	d_value = float(params['d_real'])

	result = d_value

        return {'score' : result}

if __name__ == "__main__":
       print(main(1, { u"a_cat": np.array([0.8,0.2,0.3,0.41,0.1]), u"b_cat": np.array([0.4,0.2,0.3,0.41,0.1]), u"c_int": np.array([0.2]), u"d_real" : np.array([30]) }))
