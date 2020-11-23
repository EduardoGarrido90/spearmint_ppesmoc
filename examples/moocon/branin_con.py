import math
import numpy as np

def evaluate(job_id, params):

    x = params['X']
    y = params['Y']

    print 'Evaluating at (%f, %f)' % (x, y)

    # if x < 0 or x > 5.0 or y > 5.0:
    #     return np.nan
    # Feasible region: x in [0,5] and y in [0,5]

    obj1 = float(np.square(y - (5.1/(4*np.square(math.pi)))*np.square(x) + (5/math.pi)*x- 6) + 10*(1-(1./(8*math.pi)))*np.cos(x) + 10)
 
    obj2 = -obj1

#    c1 = 1.5 - x - 2.0*y - 0.5*np.sin(2*np.pi*(x**2 - 2.0*x))
#    c2 = x**2 + y**2 - 1.5
#    c1 = -c1
#    c2 = -c2
#    c1 = x - (-5.0)
#    c2 = y - 5.0
    c1 = 0.5
    c2 = 0.5
    
    return {
        "branin_1"       : obj1, 
        "branin_2"       : obj2, 
        "c1"             : c1,
        "c2"             : c2
    }

def main(job_id, params):
    try:
        return evaluate(job_id, params)
    except Exception as ex:
        print ex
        print 'An error occurred in branin_con.py'
        return np.nan
