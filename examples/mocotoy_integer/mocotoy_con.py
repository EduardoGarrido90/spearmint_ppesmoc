import numpy as np

def evaluate(job_id, params):

    #Z is just noise for testing.

    x = params['X']
    y = params['Y']
    z = params['Z']

    print 'Evaluating at (%f, %f, %f)' % (x, y, z)

#    obj1 = float(-np.power(x-10.0,2.0)-np.power(y-15.0,2.0))
    obj1 = x*y+z/1000.0
    obj2 = -y*x+z/1000.0
#    obj2 = float(-np.power(x+5.0,2.0)-np.power(y,2.0))

#    c1 = float(np.power(y-(5.1/(4.0*np.power(np.pi,2.0)))*np.power(x,2.0)+(5.0/np.pi)*x-6.0,2.0)\
#	+10.0*(1-1/(8.0*np.pi))*np.cos(x)+9.0)
    
    c1 = x + np.cos(z)/1000.0
    c2 = y - np.sin(z)/1000.0

    print 'c1: ' + str(c1)
    print 'c2: ' + str(c2)
    print 'obj1: ' + str(obj1)
    print 'obj2: ' + str(obj2)

    return {
        "mocotoy_1"       : obj1, 
        "mocotoy_2"       : obj2, 
        "c1"	   : c1,
        "c2"	   : c2
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
