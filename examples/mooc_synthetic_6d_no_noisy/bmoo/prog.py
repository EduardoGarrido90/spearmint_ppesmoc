import math
import numpy as np
from synthetic_problem import  Synthetic_problem

problem = None
NUM_EXP = 1

def main(job_id, params):

	global problem

	if problem is None:
		problem = Synthetic_problem(NUM_EXP)

        values = problem.f(params)
	values["c1"] *= -1.0
	values["c2"] *= -1.0
	return values

