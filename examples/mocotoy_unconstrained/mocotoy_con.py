import numpy as np

def evaluate(job_id, params):

    x = params['X']
    y = params['Y']

    print 'Evaluating at (%f, %f)' % (x, y)

#    obj1 = float(-np.power(x-10.0,2.0)-np.power(y-15.0,2.0))
    obj1 = x*y
    obj2 = -y*x
#    obj2 = float(-np.power(x+5.0,2.0)-np.power(y,2.0))

#    c1 = float(np.power(y-(5.1/(4.0*np.power(np.pi,2.0)))*np.power(x,2.0)+(5.0/np.pi)*x-6.0,2.0)\
#	+10.0*(1-1/(8.0*np.pi))*np.cos(x)+9.0)
    
#    c1 = x
#    c2 = y
    
    return {
        "mocotoy_1"       : obj1, 
        "mocotoy_2"       : obj2
  #      "c1"	   : c1,
 #       "c2"	   : c2,
    }

def main(job_id, params):
    try:
        return evaluate(job_id, params)
    except Exception as ex:
        print ex
        print 'An error occurred in mocotoy_con.py'
        return np.nan
