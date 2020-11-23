import os
import sys
import importlib
import imp
import pdb
import numpy             as np
import numpy.random      as npr
import numpy.linalg      as npla
import matplotlib        as mpl
mpl.use('Agg')

from spearmint.visualizations         import plots_2d
from spearmint.utils.parsing          import parse_config_file
from spearmint.utils.parsing          import parse_tasks_from_jobs
from spearmint.utils.parsing          import get_objectives_and_constraints
from spearmint.utils.parsing          import DEFAULT_TASK_NAME
from spearmint.utils.database.mongodb import MongoDB
from spearmint.tasks.input_space      import InputSpace
from spearmint.tasks.input_space      import paramify_no_types
from spearmint.main                   import load_jobs
from spearmint.utils.moop             import MOOP_basis_functions
from spearmint.utils.moop             import average_min_distance
import os
import sys
from spearmint.grids                 import sobol_grid
import scipy.optimize as spo

def main(expt_dir):

	os.chdir(expt_dir)
	sys.path.append(expt_dir)

	options         = parse_config_file(expt_dir, 'config.json')
	experiment_name = options["experiment-name"]

#	main_file = options['main_file']
	main_file = 'wrapper'
	if main_file[-3:] == '.py':
		main_file = main_file[:-3]
	module  = __import__(main_file)

	input_space     = InputSpace(options["variables"])
	chooser_module  = importlib.import_module('spearmint.choosers.' + options['chooser'])
	chooser         = chooser_module.init(input_space, options)
	db              = MongoDB(database_address=options['database']['address'])
	jobs            = load_jobs(db, experiment_name)
	hypers          = db.load(experiment_name, 'hypers')
	objective       = parse_tasks_from_jobs(jobs, experiment_name, options, input_space).values()[0]

	def create_fun(task):
		def fun(params, gradient = False):

			if len(params.shape) > 1 and params.shape[ 1 ] > 1:

				values = np.zeros(params.shape[ 0 ])
				params_orig = params

				for i in range(params_orig.shape[ 0 ]):
					param = params[ i, : ]
					param = param.flatten()
					param = input_space.from_unit(np.array([ param ])).flatten()
					
					values[ i ] = module.main(0, paramify_no_types(input_space.paramify(param)))

			else:
				return module.main(0, paramify_no_types(input_space.paramify(params)))

			return values

		return fun

	fun = create_fun(objective)

	# We iterate through each recommendation made

	i = 0
	more_recommendations = True
	while more_recommendations:

                recommendation = db.load(experiment_name, 'recommendations', {'id' : i + 1})

		if recommendation == None:
			more_recommendations = False
		else:

                        solution_om = input_space.vectorify(recommendation[ 'params_om' ])

			M=1
			vsom_acum = 0.0
			for j in range(M):
	                        vsom_acum += fun(solution_om, gradient = False)['score']
			values_solution_om = - vsom_acum/float(M)

                        with open('value_solution_om.txt', 'a') as f:
                                print >> f, "%lf" % (values_solution_om)

                        with open('params_om.txt','a') as f_handle:
                                np.savetxt(f_handle, np.array([solution_om]), delimiter = ' ', newline = '\n')

		i+=1

if __name__ == '__main__':
	main(*sys.argv[1:])

