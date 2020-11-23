import numpy as np

def evaluate(job_id, params):

    x = params['X']
    y = params['Y']

    print 'Evaluating at (%f, %f)' % (x, y)

    obj1 = float(-np.power(x-10.0,2.0)-np.power(y-15.0,2.0))
    obj2 = float(-np.power(x+5.0,2.0)-np.power(y,2.0))

    c1 = float(np.power(y-(5.1/(4.0*np.power(np.pi,2.0)))*np.power(x,2.0)+(5.0/np.pi)*x-6.0,2.0)+10.0*(1-1/(8.0*np.pi))*np.cos(x)+9.0)

    return {
        "o1"       : obj1, 
        "o2"       : obj2, 
        "c1"	   : c1 * - 1.0
    }

def main(job_id, params):
    try:
        return evaluate(job_id, params)
    except Exception as ex:
        print ex
        print 'An error occurred in mocotoy_con.py'
        return np.nan

if __name__ == "__main__":
	main(0, {u'X': np.array([ 5.0 ]), u'Y': np.array([ 2.8 ])})
