from spearmint.models.gp import GP
from spearmint.acquisition_functions.predictive_entropy_search_multiobjective import sample_gp_with_random_features
from spearmint.utils.parsing          import parse_config_file
from spearmint.tasks.input_space      import InputSpace
from spearmint.tasks.input_space      import paramify_no_types
from spearmint.utils.parsing          import parse_tasks_from_jobs
import numpy as np

NUM_RANDOM_FEATURES = 1000

import numpy as np
import ghalton

class Synthetic_problem:

    def __init__(self, num_experiment):
	
	state = np.random.get_state()

	np.random.seed(num_experiment)

	options = parse_config_file('.', 'config.json')
	input_space = InputSpace(options["variables"])
	tasks = parse_tasks_from_jobs(None, options["experiment-name"], options, input_space)

	for key in tasks:
		tasks[ key ].options['likelihood'] = "NOISELESS"

        sequence_size = 1000
        sequencer = ghalton.Halton(input_space.num_dims)
        X = np.array(sequencer.get(sequence_size))

	self.models = dict()
	self.tasks = tasks
	self.input_space = input_space

	for key in tasks:
		self.models[ key ] = GP(input_space.num_dims, **tasks[ key ].options)
		self.models[ key ].params['ls'].set_value(np.ones(input_space.num_dims) * 0.25 * input_space.num_dims)

		params = dict()
		params['hypers'] = dict()

		for hp in self.models[ key ].params:
			params['hypers'][ hp ] = self.models[ key ].params[ hp ].value

		params['chain length'] = 0.0

		# We sample given the specified hyper-params and repeat to guarante negative and positive
		# values. This is done to have feasible constraints

		samples = self.models[ key ].sample_from_prior_given_hypers(X)

		while np.minimum(1.0 - np.mean(samples < 0), np.mean(samples < 0)) < 0.25:
			samples = self.models[ key ].sample_from_prior_given_hypers(X)

		self.models[ key ].fit(X, samples, hypers = params, fit_hypers = False)

#	def compute_function(gp):
#		def f(x):
#			return gp.predict(x)[ 0 ]
#		return f

#	self.funs = dict()

#	for key in self.models:
#		self.funs[ key ] = compute_function(self.models[ key ])

	self.funs = { key : sample_gp_with_random_features(self.models[ key ], NUM_RANDOM_FEATURES) for key in self.models }

	np.random.set_state(state)
	
    def f(self, x):

	values = np.zeros(len(x))

	i = 0
	for name in sorted(x.keys()):
		values[ i ] = x[ name ]
		i += 1

	if len(values.shape) <= 1:
		values = values.reshape((1, len(values)))

	evaluation = dict()

	for key in self.funs:
		evaluation[ key ] = self.funs[ key ](values, gradient = False)

        return evaluation

    def f_noisy(self, x):

	values = np.zeros(len(x))

	i = 0
	for name in sorted(x.keys()):
		values[ i ] = x[ name ]
		i +=1

	if len(values.shape) <= 1:
		values = values.reshape((1, len(values)))

	evaluation = dict()

	for key in self.funs:
		evaluation[ key ] = self.funs[ key ](values, gradient = False) + np.random.normal() * np.sqrt(1.0 / 100)

        return evaluation





