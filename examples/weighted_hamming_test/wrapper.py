import numpy as np
from sklearn.externals import joblib
import copy
NUM_EXP = 1

def main(job_id, params):
        params = copy.deepcopy(params)
        np.random.seed(NUM_EXP)

        #Parsing parameters.
        c_a = ['a', 'b', 'c']
        c_b = ['d', 'e', 'f', 'g']
        c_c = ['h', 'i', 'j', 'k', 'l']
        a_value = c_a[np.argmax(params['a'])] #Cat value hold the selected value of the transformation.
        b_value = c_b[np.argmax(params['b'])] #Cat value hold the selected value of the transformation.
        c_value = c_c[np.argmax(params['c'])] #Cat value hold the selected value of the transformation.

	result = np.sin(ord(a_value)+ ord(b_value)) + np.cos(ord(c_value))
        return {'score' : result}

if __name__ == "__main__":
        print(main(1, { u"a": np.array([0.4,0.2,0.3]), u"b": np.array([0.4,0.2,0.3, 0.6]), u"c": np.array([0.4,0.2,0.3, 0.6, 0.9]) }))
