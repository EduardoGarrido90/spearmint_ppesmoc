import math
import numpy as np
from synthetic_problem import  Synthetic_problem

problem = None
NUM_EXP = 2

def main(job_id, params):

	global problem

	if problem is None:
		problem = Synthetic_problem(NUM_EXP)

        return problem.f(params)

