import matplotlib
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
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

		# We sample given the specified hyper-params

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
	for name in x:
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
	for name in x:
		values[ i ] = x[ name ]
		i +=1

	if len(values.shape) <= 1:
		values = values.reshape((1, len(values)))

	evaluation = dict()

	for key in self.funs:
		evaluation[ key ] = self.funs[ key ](values, gradient = False) + np.random.normal() * np.sqrt(1.0 / 100)

        return evaluation

    def plot(self):

	assert(self.input_space.num_dims == 2 or self.input_space.num_dims == 1)

        size = 50
        x = np.linspace(0, 1, size)
        y = np.linspace(0, 1, size)
        X, Y = np.meshgrid(x, y)

	if self.input_space.num_dims == 2:
		k = 0
		for key in self.models:

			Z = np.zeros((size, size))
			for i in range(size):
				for j in range(size):
					params = self.input_space.from_unit(np.array([ X[ i, j ], Y[ i, j ]])).flatten()
					Z[ i, j ] = self.f(paramify_no_types(self.input_space.paramify(params)))[ key ]
	
			plt.figure()
			im = plt.imshow(Z, interpolation = 'bilinear', origin = 'lower', cmap = cm.gray, extent = (0, 1, 0, 1))
			CS = plt.contour(X, Y, Z)
			plt.clabel(CS, inline = 1, fontsize = 10)
			plt.title(str(key))
			plt.show()
			k += 1
	else:
		k = 0
		for key in self.models:

			Z = np.zeros(size)
			for i in range(size):
				params = self.input_space.from_unit(np.array([ x[ i ]])).flatten()
				Z[ i ] = self.f(paramify_no_types(self.input_space.paramify(params)))[ key ]

			plt.figure()
			plt.plot(x, Z, color='red', marker='.', markersize=1)
			plt.title(str(key))
			plt.show()
			k += 1



