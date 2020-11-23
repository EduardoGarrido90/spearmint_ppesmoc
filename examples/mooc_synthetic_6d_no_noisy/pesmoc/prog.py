import math
import numpy as np
from synthetic_problem import  Synthetic_problem

problem = None
NUM_EXP = 2

def main(job_id, params):

	global problem

	if problem is None:
		problem = Synthetic_problem(NUM_EXP)

	print("Point where the functions are evaluated in input space:")
        print(params)
        return problem.f(params)

