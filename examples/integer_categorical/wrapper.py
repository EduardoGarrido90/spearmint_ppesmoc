import numpy as np
from sklearn.externals import joblib
import copy
NUM_EXP = 1

def main(job_id, params):
        params = copy.deepcopy(params)
        np.random.seed(NUM_EXP)

        #Parsing parameters.
        categorical_values = ["red", "blue", "green"]
        cat_value = categorical_values[np.argmax(params['a_categorical_variable'])] #Cat value hold the selected value of the transformation.
	
	#The integer value lies in the range [1,5].
        b_integer_variable = np.linspace(0, 1, 4)
        int_value = int(np.argmin(np.absolute(params['b_integer_variable'][ 0 ] - b_integer_variable))) + 1 #Int value hold the selected value of the transformation
	
	#The real variable lies in the range [0,50].	
	real_value = float(params['c_real_variable'])

	result = 0.0
	if cat_value == 'red':
		result = real_value * int_value
	elif cat_value == 'blue':
		result = real_value - int_value
	else:
		result = real_value + int_value

        return {'score' : result}

if __name__ == "__main__":
        print(main(1, { u"a_categorical_variable": np.array([0.4,0.2,0.3]), u"b_integer_variable": np.array([0.2]), u"c_real_variable" : np.array([30]) }))
