# -*- coding: utf-8 -*-
# Spearmint
#
# Academic and Non-Commercial Research Use Software License and Terms
# of Use
#
# Spearmint is a software package to perform Bayesian optimization
# according to specific algorithms (the “Software”).  The Software is
# designed to automatically run experiments (thus the code name
# 'spearmint') in a manner that iteratively adjusts a number of
# parameters so as to minimize some objective in as few runs as
# possible.
#
# The Software was developed by Ryan P. Adams, Michael Gelbart, and
# Jasper Snoek at Harvard University, Kevin Swersky at the
# University of Toronto (“Toronto”), and Hugo Larochelle at the
# Université de Sherbrooke (“Sherbrooke”), which assigned its rights
# in the Software to Socpra Sciences et Génie
# S.E.C. (“Socpra”). Pursuant to an inter-institutional agreement
# between the parties, it is distributed for free academic and
# non-commercial research use by the President and Fellows of Harvard
# College (“Harvard”).
#
# Using the Software indicates your agreement to be bound by the terms
# of this Software Use Agreement (“Agreement”). Absent your agreement
# to the terms below, you (the “End User”) have no rights to hold or
# use the Software whatsoever.
#
# Harvard agrees to grant hereunder the limited non-exclusive license
# to End User for the use of the Software in the performance of End
# User’s internal, non-commercial research and academic use at End
# User’s academic or not-for-profit research institution
# (“Institution”) on the following terms and conditions:
#
# 1.  NO REDISTRIBUTION. The Software remains the property Harvard,
# Toronto and Socpra, and except as set forth in Section 4, End User
# shall not publish, distribute, or otherwise transfer or make
# available the Software to any other party.
#
# 2.  NO COMMERCIAL USE. End User shall not use the Software for
# commercial purposes and any such use of the Software is expressly
# prohibited. This includes, but is not limited to, use of the
# Software in fee-for-service arrangements, core facilities or
# laboratories or to provide research services to (or in collaboration
# with) third parties for a fee, and in industry-sponsored
# collaborative research projects where any commercial rights are
# granted to the sponsor. If End User wishes to use the Software for
# commercial purposes or for any other restricted purpose, End User
# must execute a separate license agreement with Harvard.
#
# Requests for use of the Software for commercial purposes, please
# contact:
#
# Office of Technology Development
# Harvard University
# Smith Campus Center, Suite 727E
# 1350 Massachusetts Avenue
# Cambridge, MA 02138 USA
# Telephone: (617) 495-3067
# Facsimile: (617) 495-9568
# E-mail: otd@harvard.edu
#
# 3.  OWNERSHIP AND COPYRIGHT NOTICE. Harvard, Toronto and Socpra own
# all intellectual property in the Software. End User shall gain no
# ownership to the Software. End User shall not remove or delete and
# shall retain in the Software, in any modifications to Software and
# in any Derivative Works, the copyright, trademark, or other notices
# pertaining to Software as provided with the Software.
#
# 4.  DERIVATIVE WORKS. End User may create and use Derivative Works,
# as such term is defined under U.S. copyright laws, provided that any
# such Derivative Works shall be restricted to non-commercial,
# internal research and academic use at End User’s Institution. End
# User may distribute Derivative Works to other Institutions solely
# for the performance of non-commercial, internal research and
# academic use on terms substantially similar to this License and
# Terms of Use.
#
# 5.  FEEDBACK. In order to improve the Software, comments from End
# Users may be useful. End User agrees to provide Harvard with
# feedback on the End User’s use of the Software (e.g., any bugs in
# the Software, the user experience, etc.).  Harvard is permitted to
# use such information provided by End User in making changes and
# improvements to the Software without compensation or an accounting
# to End User.
#
# 6.  NON ASSERT. End User acknowledges that Harvard, Toronto and/or
# Sherbrooke or Socpra may develop modifications to the Software that
# may be based on the feedback provided by End User under Section 5
# above. Harvard, Toronto and Sherbrooke/Socpra shall not be
# restricted in any way by End User regarding their use of such
# information.  End User acknowledges the right of Harvard, Toronto
# and Sherbrooke/Socpra to prepare, publish, display, reproduce,
# transmit and or use modifications to the Software that may be
# substantially similar or functionally equivalent to End User’s
# modifications and/or improvements if any.  In the event that End
# User obtains patent protection for any modification or improvement
# to Software, End User agrees not to allege or enjoin infringement of
# End User’s patent against Harvard, Toronto or Sherbrooke or Socpra,
# or any of the researchers, medical or research staff, officers,
# directors and employees of those institutions.
#
# 7.  PUBLICATION & ATTRIBUTION. End User has the right to publish,
# present, or share results from the use of the Software.  In
# accordance with customary academic practice, End User will
# acknowledge Harvard, Toronto and Sherbrooke/Socpra as the providers
# of the Software and may cite the relevant reference(s) from the
# following list of publications:
#
# Practical Bayesian Optimization of Machine Learning Algorithms
# Jasper Snoek, Hugo Larochelle and Ryan Prescott Adams
# Neural Information Processing Systems, 2012
#
# Multi-Task Bayesian Optimization
# Kevin Swersky, Jasper Snoek and Ryan Prescott Adams
# Advances in Neural Information Processing Systems, 2013
#
# Input Warping for Bayesian Optimization of Non-stationary Functions
# Jasper Snoek, Kevin Swersky, Richard Zemel and Ryan Prescott Adams
# Preprint, arXiv:1402.0929, http://arxiv.org/abs/1402.0929, 2013
#
# Bayesian Optimization and Semiparametric Models with Applications to
# Assistive Technology Jasper Snoek, PhD Thesis, University of
# Toronto, 2013
#
# 8.  NO WARRANTIES. THE SOFTWARE IS PROVIDED "AS IS." TO THE FULLEST
# EXTENT PERMITTED BY LAW, HARVARD, TORONTO AND SHERBROOKE AND SOCPRA
# HEREBY DISCLAIM ALL WARRANTIES OF ANY KIND (EXPRESS, IMPLIED OR
# OTHERWISE) REGARDING THE SOFTWARE, INCLUDING BUT NOT LIMITED TO ANY
# IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE, OWNERSHIP, AND NON-INFRINGEMENT.  HARVARD, TORONTO AND
# SHERBROOKE AND SOCPRA MAKE NO WARRANTY ABOUT THE ACCURACY,
# RELIABILITY, COMPLETENESS, TIMELINESS, SUFFICIENCY OR QUALITY OF THE
# SOFTWARE.  HARVARD, TORONTO AND SHERBROOKE AND SOCPRA DO NOT WARRANT
# THAT THE SOFTWARE WILL OPERATE WITHOUT ERROR OR INTERRUPTION.
#
# 9.  LIMITATIONS OF LIABILITY AND REMEDIES. USE OF THE SOFTWARE IS AT
# END USER’S OWN RISK. IF END USER IS DISSATISFIED WITH THE SOFTWARE,
# ITS EXCLUSIVE REMEDY IS TO STOP USING IT.  IN NO EVENT SHALL
# HARVARD, TORONTO OR SHERBROOKE OR SOCPRA BE LIABLE TO END USER OR
# ITS INSTITUTION, IN CONTRACT, TORT OR OTHERWISE, FOR ANY DIRECT,
# INDIRECT, SPECIAL, INCIDENTAL, CONSEQUENTIAL, PUNITIVE OR OTHER
# DAMAGES OF ANY KIND WHATSOEVER ARISING OUT OF OR IN CONNECTION WITH
# THE SOFTWARE, EVEN IF HARVARD, TORONTO OR SHERBROOKE OR SOCPRA IS
# NEGLIGENT OR OTHERWISE AT FAULT, AND REGARDLESS OF WHETHER HARVARD,
# TORONTO OR SHERBROOKE OR SOCPRA IS ADVISED OF THE POSSIBILITY OF
# SUCH DAMAGES.
#
# 10. INDEMNIFICATION. To the extent permitted by law, End User shall
# indemnify, defend and hold harmless Harvard, Toronto and Sherbrooke
# and Socpra, their corporate affiliates, current or future directors,
# trustees, officers, faculty, medical and professional staff,
# employees, students and agents and their respective successors,
# heirs and assigns (the "Indemnitees"), against any liability,
# damage, loss or expense (including reasonable attorney's fees and
# expenses of litigation) incurred by or imposed upon the Indemnitees
# or any one of them in connection with any claims, suits, actions,
# demands or judgments arising from End User’s breach of this
# Agreement or its Institution’s use of the Software except to the
# extent caused by the gross negligence or willful misconduct of
# Harvard, Toronto or Sherbrooke or Socpra. This indemnification
# provision shall survive expiration or termination of this Agreement.
#
# 11. GOVERNING LAW. This Agreement shall be construed and governed by
# the laws of the Commonwealth of Massachusetts regardless of
# otherwise applicable choice of law standards.
#
# 12. NON-USE OF NAME.  Nothing in this License and Terms of Use shall
# be construed as granting End Users or their Institutions any rights
# or licenses to use any trademarks, service marks or logos associated
# with the Software.  You may not use the terms “Harvard” or
# “University of Toronto” or “Université de Sherbrooke” or “Socpra
# Sciences et Génie S.E.C.” (or a substantially similar term) in any
# way that is inconsistent with the permitted uses described
# herein. You agree not to use any name or emblem of Harvard, Toronto
# or Sherbrooke, or any of their subdivisions for any purpose, or to
# falsely suggest any relationship between End User (or its
# Institution) and Harvard, Toronto and/or Sherbrooke, or in any
# manner that would infringe or violate any of their rights.
#
# 13. End User represents and warrants that it has the legal authority
# to enter into this License and Terms of Use on behalf of itself and
# its Institution.


import numpy          as np
import numpy.random   as npr
import random
import scipy.optimize as spo
import multiprocessing
import importlib
import logging
import time

from collections import defaultdict

from spearmint.utils.grad_check      import check_grad
from spearmint.grids                 import sobol_grid
from spearmint.models.abstract_model import function_over_hypers
from spearmint.models.gp             import GP
from spearmint.utils.moop            import MOOP
from spearmint                       import models
from spearmint                       import acquisition_functions
from spearmint.acquisition_functions.constraints_helper_functions import constraint_confidence_over_hypers, total_constraint_confidence_over_hypers
from scipy.spatial.distance import cdist
from spearmint                       import optimizers

CHOOSER_OPTION_DEFAULTS  = {
    'grid_size'      : 20000,
    'grid_seed'      : 0,
    'check_grad'     : False,
    'parallel_opt'   : False,
    'batch_size'     : 1,
    'num_spray'      : 10,
    'moo_use_grid_only_to_solve_problem'      : False,
    'moo_grid_size_to_solve_problem'      : 20000,
    'iters_done_to_optimize_means_multiobjective' : 1,
    'pop_size_nsga2' : 100,
    'epochs_nsga2' : 100,
    'spray_std'      : 1e-4, # todo -- just set this to the tolerance?
    'grid_subset'    : 1,
    'optimize_best'  : True,
    'optimize_acq'   : True,
    'regenerate_grid': True,  # whether to pick a new sobol grid every time.. grid_seed is ignored if this is True
    'use_sobol_grid_for_mo_optimization' : True, # whether to use a sobol or uniform grid.
    'seed_for_uniform_grid' : 1, # If sobol is not used, select the seed for the grid.
    'optimizer'  : 'L_BFGS_B',
    'print_acquisition' : 'OFF',
    'print_frontier' : 'OFF',
    'ppesmoc_type' : 'VANILLA',
    'ppesmoc_num_batches': 10,
    'use_grid_for_optimizing_acquisition_batch_1' : False, #EGM: To be changed.
    'moo_const_C_value' : 0, # C value to add to zero when checking that a constraint is larger than 0
    'use_sobol_grid_for_mo_optimization' : True, # whether to use a sobol or uniform grid.
    'seed_for_uniform_grid' : 1, # If sobol is not used, select the seed for the grid.
    'log_route_acq_funs' : 'log_acq_fun.txt'
    }

DEFAULT_NUMDESIGN = 1  # setting to 0 will cause an error. i humbly suggest setting it to 1
NLOPT_METHOD_GRADIENT_BASED  = 'LD_MMA'
NLOPT_METHOD_DERIVATIVE_FREE = 'LN_BOBYQA'

try:
    import nlopt
except:
    nlopt_imported = False
else:
    nlopt_imported = True

def init(*args, **kwargs):
    return DefaultChooser(*args, **kwargs)

class DefaultChooser(object):
    def __init__(self, input_space, options):

        self.options = CHOOSER_OPTION_DEFAULTS.copy()
        self.options.update(options)

        if self.options['parallel_opt']:
            raise NotImplementedError("Parallel optimization of EI not implemented.")
        if self.options['grid_subset'] > 1:
            raise NotImplementedError("Optimizing with multiple initializers not implemented.")

        # load the acquisition function(s)
        # we allow them to have different acquisition functions in case there is non-competitive decoupling
        # self.acquisition_function_names = dict()
        # self.acquisition_functions = dict()
        # for task_name, task_opts in self.options["tasks"].iteritems():
        #     self.acquisition_functions[task_name] = {
        #         "name"  : task_opts["acquisition"],
        #         "class" : getattr(acquisition_functions, task_opts["acquisition"])
        #     }
        self.acquisition_function_name = self.options["acquisition"]
        self.acquisition_function = getattr(acquisition_functions, self.acquisition_function_name)

	self.optimizer_name = self.options["optimizer"]
        self.optimizer = getattr(optimizers, self.optimizer_name)
	self.optimizer = self.optimizer(False, self.options)
        self.stored_recommendation = None

        self.models      = {}
        self.duration_models = {}
        self.objective   = {}		# If it is single objective optimizaiton this attribute contains the only objective
        self.objectives   = defaultdict(dict)	# In multiobjective optimization we have several objectives
        self.constraints = defaultdict(dict)
        self.tasks       = None
        self.acq         = {} # for plotting convenience, not important

        if nlopt_imported:
            self.nlopt_method                 = getattr(nlopt, NLOPT_METHOD_GRADIENT_BASED)
            self.nlopt_method_derivative_free = getattr(nlopt, NLOPT_METHOD_DERIVATIVE_FREE)

        self.input_space = input_space
        self.num_dims    = input_space.num_dims

        if not self.options['regenerate_grid']:
            self.original_grid = sobol_grid.generate(self.num_dims, 
                grid_size=self.options['grid_size'], 
                grid_seed=self.options['grid_seed'])
        self.grid = None

        # The tolerance for NLOPT in optimizaing things. if the tolerance is specified
        # in the original units, used that
        # otherwise, use the tolerance specified in the [0,1 units]
        if options['tolerance'] is not None:
            self.tolerance = options['tolerance']
            # if the tolerance is a single number, repeat it over dimensions
            if not isinstance(self.tolerance, np.ndarray) or self.tolerance.size == 1:
                self.tolerance = self.tolerance + np.zeros(self.num_dims)
            # transform the tolerance to the unit space
            self.tolerance = input_space.rescale_to_unit(self.tolerance)
        else:
            self.tolerance = options['unit_tolerance']
            # in this case, since we don't need to rescale it, we don't bother repeating it over
            # dimensions (although we could), because NLOPT interprets it properly
        # NOTE: tolerance is ignored if NLOPT is not being used!

	self.fixed_discrete_dimensions = None
        self.real_dimensions = None
        self.hypers = None #Used for Parallel-like acquisition functions to create fantasies.


    def generate_grid(self, tasks=None):

        if self.tasks == None and tasks != None:
            self.tasks = tasks

        if self.options['regenerate_grid']:
            grid = sobol_grid.generate(self.num_dims,
            grid_size=self.options['grid_size'],
            grid_seed=npr.randint(0, self.options['grid_size']))
        else:
            grid = self.original_grid
        for task_name, task in self.tasks.iteritems():
            if task.has_valid_inputs():
                to_include = task.valid_normalized_data_dict(self.input_space)['inputs']
                for i in range(to_include.shape[ 0 ]):
                        if np.min(cdist(to_include[ i : (i + 1), : ], grid)) > 0:
                                grid = np.vstack((grid, to_include[ i, : ]))
            if task.has_pending():
                #if self.num_dims == 1:
                #    to_include = task.valid_normalized_data_dict(self.input_space)['pending'][0]
                #else:
                to_include = task.valid_normalized_data_dict(self.input_space)['pending']
                for i in range(to_include.shape[ 0 ]):
                        if np.min(cdist(to_include[ i : (i + 1), : ], grid)) > 0:
                                grid = np.vstack((grid, to_include[ i, : ]))
        self.grid = grid
        self.total_inputs = reduce(lambda x,y:x+y,map(lambda t: t._inputs.shape[0], self.tasks.values()), 0)


    def set_black_boxes(self, tasks):
        
        for task_name, task in tasks.iteritems():
            if task.type.lower() == 'objective':
                self.objective = task
                self.objectives[task_name] = task
            elif task.type.lower() == 'constraint':
                self.constraints[task_name] = task
            else:
                raise Exception('Unknown task type.')
    
    def fit(self, tasks, hypers=None):

        self.tasks = tasks
        new_hypers = dict()

        self.stored_recommendation = None
        # Reset these if you are refitting. The reason is that in suggest(), to save time,
        # it checks if there is anything stored here in case best() was already
        # called explicity. So, we need to make sure best is called again if needed!

        self.set_black_boxes(self.tasks)

	task_couplings = {task_name : self.tasks[task_name].options["group"] for task_name in self.tasks}

	if self.numConstraints() == 0 and len(self.objectives) > 1:
		if len(set(task_couplings.values())) > 1 and self.options["acquisition"] != "PESM":
			raise NotImplementedError("Multi-objective optimization with decoupled tasks and acquisition different from PESM.")

        # Create the grid of optimization initializers

        # A useful hack: add previously visited points to the grid (do this every time)
        self.generate_grid()

        hypers = hypers if hypers is not None else defaultdict(dict)

        # Find the total number of samples across tasks, and do not fit if less than DEFAULT_NUMDESIGN
        if self.total_inputs < DEFAULT_NUMDESIGN:
            return hypers

        for task_name, task in tasks.iteritems():

            inputs  = task.valid_normalized_inputs(self.input_space)
            values  = task.valid_normalized_values(self.input_space)
            pending = task.normalized_pending(self.input_space)
            # Always want to fit all tasks, even if there is no data
            # default_model = 'GP' if task.options['likelihood'].lower() in ['gaussian', 'noiseless'] else 'GPClassifier'
            # model_class_name = task.options.get('model', default_model)
            model_class_name = task.options['model']


            # Don't re-instantiate the model every time
		
            if task_name not in self.models:
                
                if (task.options["acquisition"] == "PESM" or task.options["acquisition"] == "PES" or task.options["acquisition"] == "PESMC" or task.options["acquisition"] == "PPESMOC") and task.options.get("fit_mean", True):
                    logging.debug("Warning: PES or PESM is used and fit_mean is on. Setting fit_mean to False!")
                    task.options["fit_mean"] = False
                    # raise Exception("If PES is used, then you must set fit_mean to 0")

                self.models[task_name] = getattr(models, model_class_name)(self.num_dims, **task.options)

                logging.debug('')
                logging.debug('Initialized %s for task %s' % (model_class_name, task_name))
                for opt_name, opt_val in self.models[task_name].options.iteritems():
                    logging.debug('  %-18s: %s' % (opt_name, opt_val))
                logging.debug('')

            # We only want to fit the model if there is new data
            # -- otherwise, the previous fit of the model is fine
            if np.array_equal(self.models[task_name]._inputs, inputs) and \
               np.array_equal(self.models[task_name]._values, values) and \
               not self.options["always_sample"]:


                # if there is also no pending, really do absolutely nothing.
                # if there is new pending, stick it in but don't fit the hypers
                if np.array_equal(self.models[task_name].pending, pending):
                    pass
                else:
                    logging.info('Adding pending samples to %s for %s task' % (model_class_name, task_name)) # debug only, I prefer info.
                    self.models[task_name].fit(inputs, values, pending=pending, fit_hypers=False)

                # hypers are unchanged
                new_hypers[task_name] = hypers[task_name] # ... .get(task_name, None)? 
            else:
                logging.info('Fitting %s to %d data for %s task...' % (model_class_name, len(values), task_name))
                new_hypers[task_name] = self.models[task_name].fit(
                    inputs,
                    values,
                    pending=pending,
                    hypers=hypers.get(task_name, None))

            if self.options['scale-duration']:
                # need to do the same here
                if task_name not in self.duration_models:
                    # BTW, it's a bit wasteful to have a separate GP here, since the inputs are the same 
                    # for the durations and the other stuff, and the expensive part is this matrix inversion
                    # but let's just not worry about that right now

		    # We remove all task transformations except for IgnoreDims

		    task_options_only_ignore_dims  = task.options.copy()
		
		    for trans in task_options_only_ignore_dims['transformations']:
	            	if not trans.keys()[ 0 ] == "IgnoreDims":
				task_options_only_ignore_dims['transformations'].remove(trans)

		    task_options_only_ignore_dims['likelihood'] = "gaussian"

                    self.duration_models[task_name] = GP(self.num_dims, **task_options_only_ignore_dims) # durations are noisy

                    logging.debug('')
                    logging.debug('Initialized duration GP for task %s' % task_name)

                if 'duration hypers' not in hypers:
                    hypers['duration hypers'] = defaultdict(dict)
                if 'duration hypers' not in new_hypers:
                    new_hypers['duration hypers'] = defaultdict(dict)

                # We only want to fit the model if there is new data
                # -- otherwise, the previous fit of the model is fine
                if np.array_equal(self.duration_models[task_name]._inputs, inputs) and \
                              np.array_equal(self.duration_models[task_name]._values, values) and \
                              not self.options["always_sample"]:
                    new_hypers['duration hypers'][task_name] = hypers['duration hypers'][task_name]
                else:
                    logging.info('Fitting GP to %d data for durations of %s task...' % (len(values), task_name))
                    # print hypers['duration hypers'].get(task_name, None)
                    # print task.durations
                    # print np.log(task.durations)
                    # print new_hypers['duration hypers']
                    new_hypers['duration hypers'][task_name] = self.duration_models[task_name].fit(
                        self.input_space.to_unit(task.inputs), # not just valid inputs -- all inputs 
                        np.log(task.durations), 
                        hypers=hypers['duration hypers'].get(task_name, None))
                    # print task.durations

        self.hypers = new_hypers
        return new_hypers

    # There are 3 things going on here
    # 1) all tasks (self.tasks)
    # 2) the tasks that we are choosing from (either a list, or the keys of a dict)
    # 3) the decoupling, if any, of the tasks we are choosing from (decoupling) .
    #    This is stored in the values of the dict task_names.
    # optim_start time is something wierd.
    # it says when we started the "thinking" step of BO for this iteration
    # it is used in conjunction with durations, to take into account the thinking time
    # so that it's really bits per second including this stuff.
    # only used in the multi-task case where "scale-duration" is turned on...
    def suggest(self, task_couplings, optim_start_time=None):

        grid = self.grid

        if grid is None:
            raise Exception("You must call fit() before calling suggest()")

        assert np.all(grid >= 0)
        assert np.all(grid <= 1)

        if not isinstance(task_couplings, dict):
            task_couplings = {task_name : 0 for task_name in task_couplings}

        task_names = task_couplings.keys()

	# We check for random collection of points
	if self.options['acquisition'] == 'RANDOM':
        	suggestion = np.random.uniform(size = (1, self.input_space.num_dims))
		suggestion = self.input_space.from_unit(suggestion).flatten()
		if self.optimizer_name != "One_Exchange_Neighbourhood":
	#		if not self.total_inputs < DEFAULT_NUMDESIGN:
	#			self.print_images_2d_random()
			return suggestion, task_names
		else:
			return self.optimizer.give_nearest_feasible_point(suggestion), task_names

	#This code is redundant with the previous code but it is easier to read by the programmer.
	if self.options['acquisition'] == 'parallel_RANDOM':
		batch_size = self.options["batch_size"]
		suggestion = np.random.uniform(size = (batch_size, self.input_space.num_dims))
                suggestion = self.input_space.from_unit(suggestion)
                if self.options['print_acquisition'] == 'ON' and self.input_space.num_dims == 1 and batch_size == 2:
                    self.print_random_contour()
                    self.options['print_acquisition'] == 'OFF'
                return suggestion, task_names

        # Indeed it does not make sense to compute the best() and all that if we
        # have absolutely no data. 
        # But I want to be careful here because of the problem I had before, that we
        # never observed the objective (kept getting NaNs) and so kept picking randomly
        # that is not good at all-- need to use the GPs to model the NaNs.
        # so, if I REALLY have no data, then I want to do this. But that means no data
        # from ANY of the tasks. 

        if self.total_inputs < DEFAULT_NUMDESIGN:
            design_index = npr.randint(0, grid.shape[0], self.options["batch_size"])
            #suggestion = self.input_space.from_unit(grid[design_index : (design_index + 1), :])
            if self.input_space.num_dims > 1:
	        suggestion = np.array([list(self.input_space.from_unit(grid[index])) for index in design_index])
            else:
	        suggestion = np.array([np.array([self.input_space.from_unit(grid[index]) for index in design_index])]).reshape((self.options["batch_size"], self.num_dims))
            logging.info("\nSuggestion:     ")
            if self.options["batch_size"] > 1:
                self.input_space.paramify_and_print(suggestion, left_indent=16)
            else:
                self.input_space.paramify_and_print(suggestion.flatten(), left_indent=16)
	
	        suggestion = suggestion.flatten()
            if len(set(task_couplings.values())) > 1: # if decoupled
                return suggestion, [random.choice(task_names)]
            else:  # if not decoupled. this is a bit of a hack but w/e
                return suggestion, task_names
	
	# If it is a multi-objective problem and there are not at least one observation per each task,
	# we recommend something random corresponding to the tasks without any observation.
        # CHECK TO PUT HERE CONSTRAINTS.

	if self.numConstraints() >= 0 and len(self.objectives) > 1:
		if not np.all(np.array(map(lambda t: t._inputs.shape[0], self.tasks.values())) >= 1):

			design_index = npr.randint(0, grid.shape[0], self.options["batch_size"])
			#suggestion = self.input_space.from_unit(grid[ design_index : (design_index + 1), :])
			suggestion = np.array([list(self.input_space.from_unit(grid[index])) for index in design_index])
			logging.info("\nSuggestion:     ")
                        if self.options["batch_size"] > 1:
                            self.input_space.paramify_and_print(suggestion, left_indent=16)
                        else:
			    self.input_space.paramify_and_print(suggestion.flatten(), left_indent=16)
	    		    suggestion = suggestion.flatten()

			choice = int(np.where(np.array(map(lambda t: t._inputs.shape[0], self.tasks.values())) < 1)[ 0 ][ 0 ])

                        
			if len(set(task_couplings.values())) > 1: # if decoupled
				return suggestion, [ task_names[ choice ] ]
			else:  # if not decoupled. this is a bit of a hack but w/e
				return suggestion, task_names
	
        # Make sure all tasks here have the same acquisition function
        # if len({self.acquisition_functions[t]["name"] for t in task_names}) > 1:
            # raise Exception("If you are getting a suggestion for more than 1 task, they must all have the same acquisition function")

        # Compute the current best if it hasn't already been computed by the caller
        if self.stored_recommendation is None and ("initial_point" not in self.options or self.options["initial_point"] == "OFF"):
		self.best() # sets self.stored_recommendation

        if "initial_point" not in self.options or self.options["initial_point"] == "OFF":
	    current_best_value    = self.stored_recommendation['model_model_value']
	    current_best_location = self.stored_recommendation['model_model_input']
        
       	    # Put the best into the normalized space
            current_best_location = self.input_space.to_unit(current_best_location)
        else:
            current_best_location = None
            current_best_value = None

	if len(self.objectives) == 1:

                if current_best_location is not None:
	            if current_best_location.ndim == 1:
  		        current_best_location = current_best_location[None]
		    if current_best_value is not None:
		        current_best_value = (current_best_value - self.objective.standardization_mean) / self.objective.standardization_variance

		    # Add some extra candidates around the best so far (a useful hack)

	            spray_points = npr.randn(self.options['num_spray'], self.num_dims)*self.options['spray_std'] + current_best_location
		    spray_points = np.minimum(np.maximum(spray_points,0.0),1.0) # make sure they are within the unit hypercube
        
		    # also add the current best
		    grid_plus_spray = np.vstack((grid, spray_points, current_best_location))
	else:
		# If it is a multiobjective problem we only add the seen points, plus the current best values of 
		# each objective with a spray of points

	        local_grid = sobol_grid.generate(self.num_dims, grid_size = 20000)

		# We add to the grid the extra points

		local_grid = np.vstack((local_grid, self.grid[ self.options['grid_size'] : self.grid.shape[ 0 ], : ]))

		grid_plus_spray = self.grid.copy()

		for i in range(len(self.models)):

			current_best_location = self.find_optimum_gp(self.models[ self.models.keys()[ i ] ], local_grid)
			
	        	spray_points = npr.randn(self.options['num_spray'], self.num_dims)*self.options['spray_std'] + current_best_location
			spray_points = np.minimum(np.maximum(spray_points,0.0),1.0) # make sure they are within the unit hypercube

			grid_plus_spray = np.vstack((grid_plus_spray, spray_points, current_best_location))

        # flip the data structure of task couplings

        task_groups = defaultdict(list)
        for task_name, group in task_couplings.iteritems():
            task_groups[group].append(task_name)

        # ok, now task_groups is a dict with keys being the group number and the
        # values being lists of tasks

        # Now we actually compute the acquisition function!
        # first we instantiate the class
        # i have chosen to instantiate it only once instead of for each task,
        # because in PESC there is no need to redo the x* sampling and EP for each task
        # since the difference only comes at then end. in fact, the acq could just return
        # the dict for tasks and the summing could happen out here, but that's ok
        # since not all acquisition functions might be able to do that, this seems
        # like a good compromise
        self.acq = self.acquisition_function(self.num_dims, grid=grid_plus_spray, input_space=self.input_space, opt = self.options)

        task_acqs = dict()
        for group, task_group in task_groups.iteritems():
            task_acqs[group] = self.compute_acquisition_function(self.acq,
                self.acquisition_function_name, grid_plus_spray, current_best_value, task_groups[group])

        # Now task_acqs is a dict, with keys being the arbitrary group index, and the values
        # being a dict with keys "location" and "value"

        if optim_start_time is not None:
            elapsed_thinking_time = time.time() - optim_start_time
            logging.debug("Elapsed thinking time: %f" % elapsed_thinking_time)

        # normalize things by the costs
        group_costs = dict()
        for task_name, group in task_couplings.iteritems():
            if self.options['scale-duration']:
                # scale the acquisition function by the expected duration of the task
                # i.e. set the cost to the expected duation
                # print '**'
                # print np.exp(self.duration_models[task_name].values)
                # print self.duration_models[task_name].predict(task_acqs[group]["location"][None])[0]
                # print self.duration_models[task_name].predict(task_acqs[group]["location"][None])[1]
                expected_duration = np.exp(self.duration_models[task_name].predict(task_acqs[group]["location"][None])[0]) # [0] to grab mean only
                logging.debug('Expected job duration for %s: %f' % (task_name, expected_duration))
                if optim_start_time is not None:
                    expected_duration += elapsed_thinking_time # take the job time + the bayes opt time
                    logging.debug('   Total expected duration: %f' % expected_duration)
                # we take the exp because we model the log durations. this prevents us
                # from ever predicting a negative duration...
                # print '%s: cost %f' % (task_name, group_costs[group])
                group_costs[group] = expected_duration
            else:
                group_costs[group] = self.tasks[task_name].options["cost"]

        # This is where tasks compete
        if len(task_groups.keys()) > 1: # if there is competitive decoupling, do this -- it would be fine anyway, but i don't want it to print stuff
            for group, best_acq in task_acqs.iteritems():
                best_acq["value"] /= group_costs[group]
                if group_costs[group] != 1:
                    logging.debug("Scaling best acq for %s by a %s factor of 1/%f, from %f to %f" % ( \
                            task_groups[group], 
                                "duration" if self.options['scale-duration'] else "cost",
                            group_costs[group],
                            best_acq["value"]*group_costs[group],
                            task_acqs[group]["value"]))
                else:
                    logging.debug("Best acq for %s: %f" % (task_groups[group], task_acqs[group]["value"]))

        # Now, we need to find the task with the max acq value
        max_acq_value = -np.inf
        best_group = None
        for group, best_acq in task_acqs.iteritems():
            if best_acq["value"] > max_acq_value:
                best_group = group
                max_acq_value = best_acq["value"]

        # Now we know which group to evaluate
        suggested_location = task_acqs[best_group]["location"]
        best_acq_value     = task_acqs[best_group]["value"]
        suggested_tasks    = task_groups[best_group]

        # Make sure we didn't do anything weird with the bounds
        suggested_location[suggested_location > 1] = 1.0
        suggested_location[suggested_location < 0] = 0.0

        suggested_location = self.input_space.from_unit(suggested_location)

        logging.info("\nSuggestion: task(s) %s at location" % ",".join(suggested_tasks))
        if self.options["batch_size"] > 1:
                self.input_space.paramify_and_print(suggested_location, left_indent=16)
        else:
                self.input_space.paramify_and_print(suggested_location.flatten(), left_indent=16)

	# This is to avoid failure in 1-d problems

	if len(suggested_location.shape) == 0:
		suggested_location = np.array([ suggested_location ]).reshape((1, 1))

        return suggested_location, suggested_tasks

        # TODO: probably better to return suggested group, not suggested tasks... whatever.


    ITERATIONS = 0
    def compute_acquisition_function(self, acq, acq_name, grid, current_best_value, task_names):

        logging.info("Computing %s for %s" % (acq_name, ', '.join(task_names)))

        # Compute the acquisition function on the grid
	if self.options['batch_size'] == 1:
		if self.options['use_grid_for_optimizing_acquisition_batch_1']:
	        	grid_acq = function_over_hypers(self.models.values(), acq.acquisition, 
		                                        self.objective_model_dict, self.constraint_models_dict,	
        		                                grid, current_best_value, compute_grad=False, tasks=task_names, tasks_values=self.tasks)

		        # if the acquisition function is the same everywhere (usually all zeros)
        		# then there is no point using it or optimizing it
		        #if np.max(grid_acq) == np.min(grid_acq):
		        #    logging.debug('Acquisition function is constant, suggesting random grid point')
		        #    design_index = np.random.randint(0, self.grid.shape[0])
		        #    suggestion = self.input_space.from_unit(self.grid[design_index])
		        #    logging.info("\nSuggestion:     ")
		        #    self.input_space.paramify_and_print(suggestion.flatten(), left_indent=16)
		        #    return suggestion, task_names[0]

	        	# # Find the points on the grid with highest EI
		        # best_grid_inds = np.argsort(grid_ei)[-self.options['grid_subset']:]
	        	# best_grid_pred = grid_plus_spray[best_grid_inds]

        		# The index and value of the top grid point
		        best_acq_ind = np.argmax(grid_acq)
	        	best_acq_location = grid[best_acq_ind]
		        best_grid_acq_value  = np.max(grid_acq)

        		# abc = total_constraint_confidence_over_hypers(self.constraint_models, best_acq_location[None], compute_grad=False)
		        # logging.info('***Constraint prob at best grid acq: %f' % abc)
		        # logging.info('***Total probs at grid 0: %f' % total_constraint_confidence_over_hypers(self.constraint_models, self.grid[0][None], compute_grad=False))
		        # logging.info('Grid len: %d' % grid_acq.size)
		        # logging.info('Best value: %s' % current_best_value)
		        # bbb = function_over_hypers(self.models.values(), acq.acquisition,
		        #                         self.objective_model_dict, self.constraint_models_dict,
		        #                         self.grid[0][None], current_best_value, compute_grad=False)
		        # logging.info('**acq at grid 0: %f' % bbb)
		        # ccc = function_over_hypers(self.models.values(), acq.acquisition,
		        #                         self.objective_model_dict, self.constraint_models_dict,
		        #                         best_acq_location[None], current_best_value, compute_grad=False)
		        # logging.info('***acq at best acq : %f' % ccc)

		        # logging.info("\nBest acq grid value:     ")
		        # self.input_space.paramify_and_print(self.input_space.from_unit(best_acq_location).flatten(), left_indent=16)
        
		else:

			best_acq_location = np.random.rand(self.input_space.num_dims)
			best_grid_acq_value = grid_acq = function_over_hypers(self.models.values(), acq.acquisition, 
		                                        self.objective_model_dict, self.constraint_models_dict,	
        		                                best_acq_location, current_best_value, compute_grad=False, tasks=task_names, tasks_values=self.tasks)

        if self.options['optimize_acq']:

            if self.options['check_grad']:
                check_grad(lambda x: function_over_hypers(self.models.values(), acq.acquisition, 
                    self.objective_model_dict, self.constraint_models_dict, x, current_best_value, compute_grad=True),
                           best_acq_location, verbose=True, tasks=task_names, tasks_values=self.tasks)

            if nlopt_imported:

                alg = self.nlopt_method if acq.has_gradients else self.nlopt_method_derivative_free
                opt = nlopt.opt(alg, self.num_dims)

                logging.info('Optimizing %s with NLopt, %s' % (acq_name, opt.get_algorithm_name()))
                
                opt.set_lower_bounds(0.0)
                opt.set_upper_bounds(1.0)

                # define the objective function
                def f(x, put_gradient_here):
                    if x.ndim == 1:
                        x = x[None,:]

                    if put_gradient_here.size > 0:
                        a, a_grad = function_over_hypers(self.models.values(), acq.acquisition, 
                                self.objective_model_dict, self.constraint_models_dict,
                                x, current_best_value, compute_grad=True, tasks=task_names, tasks_values=self.tasks)
                        put_gradient_here[:] = a_grad.flatten()
                    else:
                        a = function_over_hypers(self.models.values(), acq.acquisition,
                                self.objective_model_dict, self.constraint_models_dict,
                                x, current_best_value, compute_grad=False, tasks=task_names, tasks_values=self.tasks)

                    return float(a)

                opt.set_max_objective(f)
                opt.set_xtol_abs(self.tolerance)
                opt.set_maxeval(self.options['opt_acq_maxeval'])

                x_opt = opt.optimize(best_acq_location.copy())

                returncode = opt.last_optimize_result()
                # y_opt = opt.last_optimum_value()
                y_opt = f(x_opt, np.array([]))

                # overwrite the current best if optimization succeeded
                if (returncode > 0 or returncode==-4) and y_opt > best_grid_acq_value:

                    if returncode == 1:
                        print 'Normal termination'
                    elif returncode == 2:
                        print 'stopval reached'
                    elif returncode == 3:
                        print 'ftol rel or abs reached'
                    elif returncode == 4:
                        print 'xtol rel or abs reached'
                    elif returncode == 5:
                        print 'maxeval %d reached' % self.options['opt_acq_maxeval']
                    elif returncode == 6:
                        print 'max time reached'
                    elif returncode == -4:
                        print 'roundoff limited termination'

                    best_acq_location = x_opt
                    best_acq_value = y_opt
                else:
                    best_acq_value = best_grid_acq_value

            else: # use bfgs
                # see http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.optimize.fmin_l_bfgs_b.html
                logging.info('Optimizing %s with %s%s' % (acq_name, self.optimizer_name, '' if acq.has_gradients else ' (numerically estimating gradients)'))
                #acq.has_gradients = False ### DEBUG
                if acq.has_gradients:
		    if self.optimizer_name == "One_Exchange_Neighbourhood":
			cg = False
		    else:
			cg = True
                    def f(x):
                        if x.ndim == 1:
                            x = x[None,:]
			if not cg:
				a = function_over_hypers(self.models.values(), acq.acquisition,
                                            self.objective_model_dict, self.constraint_models_dict,
                                        x, current_best_value, compute_grad=cg, tasks=task_names, tasks_values=self.tasks)
                                return -a.flatten()
			else:
                        	a, a_grad = function_over_hypers(self.models.values(), acq.acquisition, 
                                	    self.objective_model_dict, self.constraint_models_dict,
                                    	x, current_best_value, compute_grad=cg, tasks=task_names, tasks_values=self.tasks)
                        	return (-a.flatten(), -a_grad.flatten())
                else:
                    def f(x):
                        if x.ndim == 1:
                            x = x[None,:]
                        a = function_over_hypers(self.models.values(), acq.acquisition, 
                                    self.objective_model_dict, self.constraint_models_dict,
                                    x, current_best_value, compute_grad=False, tasks=task_names, tasks_values=self.tasks)
                        return -a.flatten()

		if self.options['batch_size'] == 1:
                	bounds = [(0,1)]*self.num_dims
			self.optimizer.set_has_gradients(acq.has_gradients)
			best_all_space_location = best_acq_location.copy()
                    	best_all_space_value = f(best_all_space_location)
			if self.optimizer_name == "One_Exchange_Neighbourhood":
			    self.optimizer.set_has_gradients(False)
			    best_feasible_location = self.optimizer.give_nearest_feasible_point(best_all_space_location)
			    best_feasible_value = f(best_feasible_location)
			    x_opt, y_opt, opt_info, iterations = self.optimizer.optimize(f, best_feasible_location.copy(), bounds)
                        else:
                            x_opt, y_opt, opt_info = self.optimizer.optimize(f, best_all_space_location.copy(), bounds)
		else:
			bounds = [(0,1)]*self.num_dims*self.options['batch_size']
                        if self.options["acquisition"] != "PPESMOC" or (self.options["acquisition"] == "PPESMOC" and self.options["ppesmoc_type"] == "VANILLA"):
			    initial_point = np.random.rand(self.input_space.num_dims*self.options['batch_size'])
			    initial_point = initial_point.reshape((self.options['batch_size'], self.input_space.num_dims))
                        elif self.options["ppesmoc_type"] == "PS_PESMOC": 

                            # XXX DHL check this Finds one point using PESMOC, the other are random

                            true_acquisition_function = self.options["acquisition"]
                            self.options["acquisition"] = "parallel_sequential"
                            self.options["acquisition_for_parallel_sequential"] = "PESMC"
                            auxiliar_acquisition_function_name = "parallel_sequential"
                            auxiliar_acquisition_function = getattr(acquisition_functions, auxiliar_acquisition_function_name)
                            opt_copy = self.options.copy()
                            opt_copy['batch_size'] = 1
                            auxiliar_acq = auxiliar_acquisition_function(self.num_dims, grid=grid, input_space=self.input_space, opt = opt_copy)
                            #candidate_point = np.array([np.random.uniform(size = (self.options["batch_size"], self.input_space.num_dims))])
                            initial_point = auxiliar_acq.compute_optimum_points(self.objective_model_dict, self.constraint_models_dict, task_names, self.tasks)
                            self.options["acquisition"] = true_acquisition_function
                            initial_point = np.vstack((initial_point, \
                                np.random.rand((self.options['batch_size'] - 1), self.input_space.num_dims)))

                        #import pdb; pdb.set_trace();
                        #Comment this in final version
                        #acq.max_ep_iterations = 3
                        #Testing gradients outside the acquisition function, pondered by the hyperparameters.
                        #import pdb; pdb.set_trace();
                        # ien = initial_point.copy()
                        # ien = np.array([ien])
                        # ien[0][0] = initial_point[0][0]+epsilon
                        # oen = initial_point.copy()
                        # oen[0][0] = initial_point[0][0]-epsilon
                        # oen = np.array([oen])
                        # initial_point = np.array([initial_point])
                        # Definitions:
                        # First: (f(ien)[0][0]-f(initial_point)[0][0])/epsilon
                        # Second: (f(ien)[0][0]-f(oen)[0][0])/(2.0*epsilon)
                        # Must equal to:
                        # f(initial_point)[1][0]
                        # No tiene sentido hacer estas pruebas ya que el resultado no es determinista.
                        # Duda Daniel, ¿Que factores tengo que hacer fijos en el EP para forzar un resultado determinista?
                        # ¿Todos los factores que no dependen de los puntos de tests y forzar a que el nuevo EP no los calcule? Creo que es eso.
                        if not self.options["acquisition"] == "parallel_sequential":
                            def f_no_grad(x):
                                return f(x)[ 0 ]

                            #print(f_no_grad(initial_point.flatten()))
                            import time
                            start = time.time()
			    #x_opt, y_opt, opt_info = spo.fmin_l_bfgs_b(f_no_grad, initial_point.flatten(), \
                                #bounds=bounds, disp=1, approx_grad=True, maxiter = 3)
                            end = time.time()
                            print('Elapsed time for opt. with differences: ' + str(end-start))
                            #TODO DEBUG QUIT THIS print('X_opt with differences: ' + x_opt)
                            #TODO DEBUG QUIT THIS print('y_opt with differences: ' + y_opt)
                            #print('initial point with differences: ' + initial_point)
                            start = time.time()
                        
                            """
                            #TIME TEST: NUMPY FUNCTION VS AUTOGRAD FUNCTION.    
                            from spearmint.acquisition_functions.PPESMOC_gradients import compute_acq_fun_wrt_test_points
                            start = time.time()
                            acq.acquisition(self.objective_model_dict, self.constraint_models_dict,
                                        initial_point, None, compute_grad=False, tasks=task_names, tasks_values=self.tasks)
                            end = time.time()
                            print('Elapsed time for Numpy function: ' + str(end-start))
                            import pdb; pdb.set_trace();
                            start = time.time()                     
                            compute_acq_fun_wrt_test_points(initial_point, self.objective_model_dict,\
                                 self.constraint_models_dict, pareto_set, list_a, tasks)
                            end = time.time()
                            print('Elapsed time for Autograd function: ' + str(end-start))
                            """
                            x_opt, y_opt, opt_info = spo.fmin_l_bfgs_b(f, initial_point.flatten(), \
                                bounds=bounds, disp=1, approx_grad=not acq.has_gradients, maxiter= 3)
                            #print(f_no_grad(initial_point.flatten()))
                            end = time.time()
                            #Test time is OK!!! =)
                            print('Elapsed time for opt. with Autograd: ' + str(end-start))
                            import pdb; pdb.set_trace();
                            #Testing convergence.
                            print('Testing convergence with the value of the function in the neighbourhood of the optimum.')
                            for i in range(len(x_opt)):
                                print('Test number ' + str(i))
                                eps = np.zeros(x_opt.shape)
                                eps[i] = 1e-4
                                x_opt_plus = x_opt + eps
                                x_opt_minus = x_opt - eps
                                y_opt_plus = f_no_grad(x_opt_plus)
                                y_opt_minus = f_no_grad(x_opt_minus)
                                import pdb; pdb.set_trace();
                                assert y_opt[0] < y_opt_plus[0]
                                assert y_opt[0] < y_opt_minus[0]
                                print('OK!')
                            import pdb; pdb.set_trace();
                            print('X_opt with Autograd: ' + x_opt)
                            print('y_opt with Autograd: ' + y_opt)
                            #print('initial point with Autograd: ' + initial_point)
                            #print(y_opt)
                            #print(opt_info)
#			    x_opt, y_opt, opt_info = spo.fmin_l_bfgs_b(f, initial_point.flatten(), \
#                                bounds=bounds, disp=0, approx_grad=not acq.has_gradients)
                        else:
                            x_opt = acq.compute_optimum_points(self.objective_model_dict, self.constraint_models_dict, task_names, self.tasks)
                            y_opt = f(np.array([x_opt]))[0][0]
                            x_opt = x_opt.flatten()
                        #import pdb; pdb.set_trace();
                y_opt = -y_opt
                x_opt[x_opt > 1.0] = 1.0
                x_opt[x_opt < 0.0] = 0.0
		if self.optimizer_name == "One_Exchange_Neighbourhood":
                        #TESTING
                        #if(DefaultChooser.ITERATIONS > 10):
                         #       import pdb; pdb.set_trace();
			if y_opt > (-best_feasible_value[0]):
				best_acq_location = x_opt
                                best_acq_value = y_opt
                        else:
                                best_acq_location = best_feasible_location
                                best_acq_value = best_feasible_value
                elif self.options['batch_size'] == 1:
                        if y_opt > best_grid_acq_value:
                                best_acq_location = x_opt
                                best_acq_value = y_opt
                        else:
                                best_acq_value = best_grid_acq_value
                else: #No previous grid search as optimization is done by gradients since batch adds to much dimensionality to acq. fun.
                        best_acq_location = x_opt
                        best_acq_value = y_opt 

                #This if is put here to also apply if OEN is selected.
                if self.options["batch_size"] > 1:  
                    best_acq_location = best_acq_location.reshape((self.options['batch_size'], self.acq.num_dims))

	    if self.options['batch_size'] == 1:
            	logging.debug('Best %s before optimization: %f' % (acq_name, best_grid_acq_value))
            	logging.debug('Best %s after  optimization: %f' % (acq_name, best_acq_value))
	    else:
		logging.debug('Best %s after optimization with gradients %f' % (acq_name, best_acq_value))

        else:
            # do not optimize the acqusition function
            logging.debug('Best %s on grid: %f' % (acq_name, best_grid_acq_value))

       
        #Print plots of acquisition functions for visualization and debug purposes. 
        if self.options["print_acquisition"] == "ON":
            if self.num_dims == 1 and self.options["batch_size"] == 2:
                self.print_images_batch()
            else:
                self.print_images_2d()
        if self.options["print_frontier"] == "ON":
    	    self.print_frontier()

        return {"location" : best_acq_location, "value" : best_acq_value}

    @property
    def objective_model(self):
        return self.models[self.objective.name]    
    @property
    def obj_model(self):
        return self.models[self.objective.name]
    @property
    def objective_model_dict(self):
		if len(self.objectives) > 1:
        		return { self.objectives[ obj ].name : self.models[ self.objectives[ obj ].name ] for obj in self.objectives }
		else:
        		return { self.objective.name:self.models[self.objective.name] }
    @property
    def constraint_models_dict(self):
        return {c:self.models[c] for c in self.constraints}
    @property
    def constraint_models(self):
        return [self.models[c] for c in self.constraints]

    # Returns a boolean array of size pred.shape[0] indicating whether the prob con-constraint is satisfied there


    def probabilistic_constraint_over_hypers(self, pred, factor = 1.0, C = 0):
        return reduce(np.logical_and, 
            [ constraint_confidence_over_hypers(self.models[c], pred, C = C) >= (1.0-self.tasks[c].options['delta']) * factor
                for c in self.constraints ], 
                np.ones(pred.shape[0], dtype=bool))

    def probabilistic_constraint_over_hypers_negative(self, pred, factor = 1.0, C = 0):
        return reduce(np.logical_and, 
            [ (1.0 - constraint_confidence_over_hypers(self.models[c], pred, C = -C)) >= (1.0-self.tasks[c].options['delta']) * factor
                for c in self.constraints ], 
                np.ones(pred.shape[0], dtype=bool))


    def best(self):

        # We want to save and return 3 types of recommendations:
        # 1) The best according to the model (model_model)
        # 2) The best objective/constraint observation (obser_obser)
        # 3) The best objective observation at a place where the model thinks the constraint is satisfied (obser_model)
        # (The last one is weird and intended for cases where the objective isn't noisy but the constraint is)

        # If there is not enough data, just return something random...

        self.total_inputs = reduce(lambda x,y:x+y,map(lambda t: t._inputs.shape[0], self.tasks.values()), 0)
        if self.total_inputs < DEFAULT_NUMDESIGN:
            design_index = np.random.randint(0, self.grid.shape[0])

            # what is the point of this exactly? oh well
            rec =  {'model_model_input' : self.input_space.from_unit(self.grid[design_index:(design_index + 1), :]),
                    'model_model_value' : None,
                    'obser_obser_input' : None,
                    'obser_obser_value' : None,
                    'obser_model_input' : None,
                    'obser_model_value' : None}

	elif self.numConstraints() > 0 and len(self.objectives) > 1:


		# This is a multiobjective problem with constraints

		# If there is not at least a observation per each task we return something random

		if not np.all(np.array(map(lambda t: t._inputs.shape[0], self.tasks.values())) >= 1):

			design_index = np.random.randint(0, self.grid.shape[0])

			rec =  {'model_model_input' : self.input_space.from_unit(self.grid[design_index:(design_index + 1), :]),
				'model_model_value' : None, 'obser_obser_input' : None,
				'obser_obser_value' : None, 'obser_model_input' : None, 'obser_model_value' : None}
		else:

			moop = MOOP(self.objectives, self.models, self.input_space)
			moop_observ_model = MOOP(self.objectives, self.models, self.input_space)

			# We only solve the multi-objective-problem after a particular number of observations
			# The recommendations made before that are garbage

	                task_couplings = {task_name : self.tasks[task_name].options["group"] for task_name in self.tasks}

                        task_groups = defaultdict(list)
                        for task_name, group in task_couplings.iteritems():
                            task_groups[group].append(self.tasks[task_name])

                        total_evals = np.sum(np.array(map(lambda g: np.min(map(lambda t: t._inputs.shape[0], g)), task_groups.values())))

			if total_evals % (len(task_groups) * self.options['iters_done_to_optimize_means_multiobjective']) == 0:

	                	logging.info('\nSolving Multi-objective global optimization of posterior means with constraints!')

				# Currently it is only implemented the use of the grid for the solution. Future work includes
				# using evolutionary strategies etc. for that.

				assert self.options['moo_use_grid_only_to_solve_problem'] == True

				# Now we iteratively find the best points that are likely to be beasible

                                if self.options["use_sobol_grid_for_mo_optimization"]:
				    grid = sobol_grid.generate(self.input_space.num_dims, \
				        	self.input_space.num_dims * self.options['moo_grid_size_to_solve_problem'])
                                else:
                                    state = np.random.get_state()
                                    np.random.seed(self.options["seed_for_uniform_grid"])
                                    grid = np.random.uniform(size=(self.input_space.num_dims * self.options['moo_grid_size_to_solve_problem'],\
                                        self.input_space.num_dims))
                                    np.random.set_state(state)

				factor = 1.0

			
				if self.options["acquisition"] == "BMOO" or (self.options["acquisition"] == "parallel_sequential" \
					and "acquisition_for_parallel_sequential" in self.options and \
					self.options["acquisition_for_parallel_sequential"] == "BMOO"):
					grid_feasible = grid[ self.probabilistic_constraint_over_hypers_negative(grid, factor, \
                                            C = self.options['moo_const_C_value']), : ]
				else:
					grid_feasible = grid[ self.probabilistic_constraint_over_hypers(grid, factor, \
                                            C = self.options['moo_const_C_value']), : ]

				while len(grid_feasible) == 0:
					factor *= 0.99

					if factor < 1e-5:
						factor = 0.0

					if self.options["acquisition"] == "BMOO" or (self.options["acquisition"] == "parallel_sequential" \
                                        	and "acquisition_for_parallel_sequential" in self.options and \
	                                        self.options["acquisition_for_parallel_sequential"] == "BMOO"):
        	                                grid_feasible = grid[ self.probabilistic_constraint_over_hypers_negative(grid, factor, \
                	                            C = self.options['moo_const_C_value']), : ]
                        	        else:
                                	        grid_feasible = grid[ self.probabilistic_constraint_over_hypers(grid, factor, \
                                        	    C = self.options['moo_const_C_value']), : ]

				moop.solve_using_grid(grid_feasible)

                                # Now the optimization of the model observations

                                grid_observed = None

                                for key in self.models:
                                    if grid_observed is None:
                                        grid_observed = self.models[ key ].inputs
                                    else:
                                        grid_observed = np.vstack(( grid_observed, self.models[ key ].inputs ))

				factor = 1.0

				if self.options["acquisition"] == "BMOO" or (self.options["acquisition"] == "parallel_sequential" \
                                                and "acquisition_for_parallel_sequential" in self.options and \
                                                self.options["acquisition_for_parallel_sequential"] == "BMOO"):
					grid_observed_feasible = grid_observed[ self.probabilistic_constraint_over_hypers_negative(\
                                            grid_observed, factor, C = self.options['moo_const_C_value']), : ]
				else:
					grid_observed_feasible = grid_observed[ self.probabilistic_constraint_over_hypers(\
                                            grid_observed, factor, C = self.options['moo_const_C_value']), : ]


				while len(grid_observed_feasible) == 0:
					factor *= 0.99

					if factor < 1e-5:
						factor = 0.0

					if self.options["acquisition"] == "BMOO" and (self.options["acquisition"] == "parallel_sequential" and \
				        "acquisition_for_parallel_sequential" in self.options and \
                                        self.options["acquisition_for_parallel_sequential"] == "BMOO"):
						grid_observed_feasible = grid_observed[ self.probabilistic_constraint_over_hypers_negative( \
                                                    grid_observed, factor), : ]
					else:
						grid_observed_feasible = grid_observed[ self.probabilistic_constraint_over_hypers( \
                                                    grid_observed, factor), : ]

				moop.solve_using_grid(grid_feasible)
				moop_observ_model.solve_using_grid(grid_observed_feasible)

			else:

	                	logging.info('\nNot performing contrained multi-objective global optimization of posterior \
					means at this iteration!')
				moop.evolve(1, 8)
				moop_observ_model.evolve(1, 8)

			# We obtain the results

			result = moop.compute_pareto_front_and_set()
			result_observed_model  = moop_observ_model.compute_pareto_front_and_set()

			# We return the data to the original space

			for i in range(result['frontier'].shape[ 0 ]):

				ntask = 0

				for obj in self.objectives:
					result['frontier'][ i, ntask ] = self.objectives[ obj ].unstandardize_mean( \
						self.objectives[ obj ].unstandardize_variance(result['frontier'][ i, ntask ]))
				        ntask = ntask + 1

				result['pareto_set'][ i, : ] = self.input_space.from_unit(result['pareto_set'][ i, : ])

			for i in range(result_observed_model['frontier'].shape[ 0 ]):

				ntask = 0

				for obj in self.objectives:
					result_observed_model['frontier'][ i, ntask ] = self.objectives[ obj ].unstandardize_mean( \
						self.objectives[ obj ].unstandardize_variance(result_observed_model['frontier'][ i, ntask ]))
				        ntask = ntask + 1

				result_observed_model['pareto_set'][ i, : ] = self.input_space.from_unit(result_observed_model['pareto_set'][ i, : ])

	 		logging.info('\nConstrained Multi-objective Global Optimization finished. \
				Size of the pareto set: %d opt.\n' %  (result['frontier'].shape[ 0 ]))

			# We return the results

			rec = {'model_model_input': result['pareto_set'],
				'model_model_value': result['frontier'],
				'obser_obser_input' : None,
				'obser_obser_value' : None,
				'obser_model_input' : result_observed_model['pareto_set'],
				'obser_model_value' : result_observed_model['frontier']}

			self.stored_recommendation_multiobjective = rec


	elif self.numConstraints() == 0 and len(self.objectives) > 1:

		# If it is a multiobjective problem we solve the problem and return an approximation to 
		# the pareto set of the current functions (as estimated by the means of the GP).

		# If there is not at least a observation per each objective we return something random
		
		if not np.all(np.array(map(lambda t: t._inputs.shape[0], self.tasks.values())) >= 1):

			design_index = np.random.randint(0, self.grid.shape[0])

			rec =  {'model_model_input' : self.input_space.from_unit(self.grid[design_index:(design_index + 1), :]),
				'model_model_value' : None, 'obser_obser_input' : None,
				'obser_obser_value' : None, 'obser_model_input' : None, 'obser_model_value' : None}
		else:

			moop = MOOP(self.objectives, self.models, self.input_space)

			# We only solve the multi-objective-problem after a particular number of observations
			# The recommendations made before that are garbage

			if self.total_inputs % (len(self.objectives) * self.options['iters_done_to_optimize_means_multiobjective']) == 0:

	                	logging.info('\nSolving Multi-objective global optimization of posterior means!')
			
				# We check if we have to use nsga2 or a grid to solve the mo optimization problem

				if self.options['moo_use_grid_only_to_solve_problem'] == True:

					moop.solve_using_grid(grid = sobol_grid.generate(self.input_space.num_dims, \
						self.input_space.num_dims * self.options['moo_grid_size_to_solve_problem']))

#		        		local_grid = sobol_grid.generate(self.num_dims, grid_size = 20000)
#					local_grid = np.vstack((local_grid, self.grid[ self.options['grid_size'] : self.grid.shape[ 0 ], : ]))
#	
#					# We add to the population the best of each objective plus a spray of points
#	
#					for i in range(len(self.models)):
#	
#						current_best_location = self.find_optimum_gp(self.models[ self.models.keys()[ i ] ], local_grid)
#				
#						spray_points = npr.randn(self.options['num_spray'], self.num_dims) * \
#							self.options['spray_std'] + current_best_location
#						spray_points = np.minimum(np.maximum(spray_points,0.0),1.0) 
#	
#						to_add = np.vstack((spray_points, current_best_location))
#	
#						for i in range(to_add.shape[ 0 ]):
#							moop.append_to_population(to_add[ i, : ])
				else:

					moop.solve_using_grid(grid = sobol_grid.generate(self.input_space.num_dims, \
						self.input_space.num_dims * self.options['moo_grid_size_to_solve_problem']))

		        		local_grid = sobol_grid.generate(self.num_dims, grid_size = 20000)
					local_grid = np.vstack((local_grid, self.grid[ self.options['grid_size'] : self.grid.shape[ 0 ], : ]))
	
					# We add to the population the best of each objective plus a spray of points
	
					for i in range(len(self.models)):
	
						current_best_location = self.find_optimum_gp(self.models[ self.models.keys()[ i ] ], local_grid)
				
						spray_points = npr.randn(self.options['num_spray'], self.num_dims) * \
							self.options['spray_std'] + current_best_location
						spray_points = np.minimum(np.maximum(spray_points,0.0),1.0) 
	
						to_add = np.vstack((spray_points, current_best_location))
	
						for i in range(to_add.shape[ 0 ]):
							moop.append_to_population(to_add[ i, : ])

					pareto_set = moop.compute_pareto_front_and_set_summary(self.options['pop_size_nsga2'])['pareto_set']

					moop.initialize_population(np.maximum(self.options['pop_size_nsga2'] - pareto_set.shape[ 0 ], 0))

					for i in range(pareto_set.shape[ 0 ]):
						moop.append_to_population(pareto_set[ i, : ])

					moop.evolve_population_only(self.options['epochs_nsga2'])

					for i in range(pareto_set.shape[ 0 ]):
						moop.append_to_population(pareto_set[ i, : ])
			else:
	                	logging.info('\nNot performing the multi-objective global optimization of posterior means at this iteration!')
				moop.evolve(1, 8)
	
			result = moop.compute_pareto_front_and_set()

			# It it is a decoupled scenario we do not return non-dominated observations, because for each observation
			# we may only have observed one objective.
	
			task_couplings = {task_name : self.tasks[task_name].options["group"] for task_name in self.tasks}
	
			if len(set(task_couplings.values())) <= 1:
				result_o = moop.get_non_dominated_observations()
			else:
		
				# In the case of a decoupled evaluation for each observation we only have one objective
				# In this case we use the models to predict the missing values

				result_o = moop.get_non_dominated_observations_predict_missing_observations()
	
			# We return the data to the original space

			for i in range(result['frontier'].shape[ 0 ]):
				ntask = 0
				for obj in self.objectives:
					result['frontier'][ i, ntask ] = self.objectives[ obj ].unstandardize_mean( \
						self.objectives[ obj ].unstandardize_variance(result['frontier'][ i, ntask ]))
					ntask = ntask + 1
				result['pareto_set'][ i, : ] = self.input_space.from_unit(result['pareto_set'][ i, : ])

			for i in range(result_o['frontier'].shape[ 0 ]):
				ntask = 0
				for obj in self.objectives:
					result_o['frontier'][ i, ntask ] = self.objectives[ obj ].unstandardize_mean( \
						self.objectives[ obj ].unstandardize_variance(result_o['frontier'][ i, ntask ]))
					ntask = ntask + 1
				result_o['pareto_set'][ i, : ] = self.input_space.from_unit(result_o['pareto_set'][ i, : ])
	
	 		logging.info('\nGlobal Optimization finished. Size of the pareto set: %d opt. %d observed.\n' % \
				(result['frontier'].shape[ 0 ], result_o['frontier'].shape[ 0 ]))

			rec = {'model_model_input': result['pareto_set'],
				'model_model_value': result['frontier'],
				'obser_obser_input' : result_o['pareto_set'],
				'obser_obser_value' : result_o['frontier'],
			'obser_model_input' : None,
			'obser_model_value' : None}

			self.stored_recommendation_multiobjective = rec

			# We return the results

			rec = {'model_model_input': result['pareto_set'],
				'model_model_value': result['frontier'],
				'obser_obser_input' : result_o['pareto_set'],
				'obser_obser_value' : result_o['frontier'],
				'obser_model_input' : None,
				'obser_model_value' : None}

        elif self.numConstraints() == 0:
            logging.info('Computing current best...')
            
            val, loc = self.best_unconstrained()
            val_o, loc_o = self.bestObservedUnconstrained()

            rec =  {'model_model_input' : loc,
                    'model_model_value' : val,
                    'obser_obser_input' : loc_o,
                    'obser_obser_value' : val_o,
                    'obser_model_input' : loc_o,
                    'obser_model_value' : val_o}

        else:
            logging.info('Computing current best...')
            # instead of using the augmented grid here, we are going to re-augment it ourselves
            # just to deal with any possibly numerical issues of the probabilistic constraint
            # being violated somewhere that we actually observed already...
            # (above: is this comment out of date? i'm confused)
            pc = self.probabilistic_constraint_over_hypers(self.grid)
            if not np.any(pc) or self.objective.valid_values.size == 0:
                # If probabilistic constraint is violated everywhere
                # The reason for the OR above:
                # some weird stuff can happen here. if the first result is NaN
                # then obj has no valid values, so it has never been standardized, so it cannot be unstandardized
                # this is probably not going to happen because -- the MC should never be satisfied if you have no values... right?
                # unless you pick some really weak constraint that is satisfied in the prior...
                val_m, loc_m = self.best_constrained_no_solution()
                val_o, loc_o = self.bestObservedConstrained()

                rec =  {'model_model_input' : loc_m,
                        'model_model_value' : val_m,
                        'obser_obser_input' : loc_o,
                        'obser_obser_value' : val_o,
                        'obser_model_input' : None,
                        'obser_model_value' : None}
            else:
                val_m, loc_m = self.best_constrained_solution_exists(pc)
                val_o, loc_o = self.bestObservedConstrained()
                rec_obser_model_val, rec_obser_model_loc = self.best_obser_model_constrained_solution_exists()

                rec =  {'model_model_input' : loc_m,
                        'model_model_value' : val_m,
                        'obser_obser_input' : loc_o,
                        'obser_obser_value' : val_o,
                        'obser_model_input' : rec_obser_model_loc,
                        'obser_model_value' : rec_obser_model_val}

        self.stored_recommendation = rec

        return rec

    # This functions optimizes a GP. It is used by the multi-objective method to provide the spray of points

    def find_optimum_gp(self, obj_model, grid = None):

	if grid is None:
		grid = self.grid

	# Compute the GP mean
	
	obj_mean, obj_var = obj_model.function_over_hypers(obj_model.predict, grid)

	# find the min and argmin of the GP mean

        current_best_location = grid[np.argmin(obj_mean),:]
        best_ind = np.argmin(obj_mean)
        current_best_value = obj_mean[best_ind]

	def f(x):
		if x.ndim == 1:
			x = x[None,:]

		mn, var, mn_grad, var_grad = obj_model.function_over_hypers(obj_model.predict, x, compute_grad=True)

		return (mn.flatten(), mn_grad.flatten())

	bounds = [(0.0,1.0)]*self.num_dims

	x_opt, y_opt, opt_info = spo.fmin_l_bfgs_b(f, current_best_location.copy(), bounds=bounds, disp=0)

	# make sure bounds were respected

	x_opt[x_opt > 1.0] = 1.0
	x_opt[x_opt < 0.0] = 0.0

	return x_opt


    """
    When computing the best we cannot be Bayesian and average the bests 
    because these are x locations which do not make sense to average
    So we average over hypers and then optimize THAT
    """
    def best_unconstrained(self):
        """
        if options['optimize_best'] is False, we will just compute on a grid and take the best
        if it is True (default), then we try to use NLopt. Otherwise if NLopt isn't installed
        we will use some python SLSQP 
        """
        obj_model = self.models[self.objective.name]

	# XXX To remove!

	grid = sobol_grid.generate(self.input_space.num_dims, 1000 * self.input_space.num_dims)

        obj_mean, obj_var = obj_model.function_over_hypers(obj_model.predict, grid)
        current_best_location = grid[np.argmin(obj_mean),:]

        # Compute the GP mean
#        obj_mean, obj_var = obj_model.function_over_hypers(obj_model.predict, self.grid)

        # find the min and argmin of the GP mean
#        current_best_location = self.grid[np.argmin(obj_mean),:]
        best_ind = np.argmin(obj_mean)
        current_best_value = obj_mean[best_ind]
        
        # optimize with NLopt?
        if self.options['optimize_best']:
            if nlopt_imported:
                opt = nlopt.opt(self.nlopt_method, self.num_dims)
                
                logging.info('Optimizing current best with NLopt, %s' % opt.get_algorithm_name())

                opt.set_lower_bounds(0.0)
                opt.set_upper_bounds(1.0)

                # define the objective function
                def f(x, put_gradient_here):

                    if x.ndim == 1:
                        x = x[None,:]

                    if put_gradient_here.size > 0:
                        mn, var, mn_grad, var_grad = obj_model.function_over_hypers(obj_model.predict, x, compute_grad=True)
                        # set grad to the gradient, here
                        put_gradient_here[:] = mn_grad.flatten()
                    else:
                        mn, var = obj_model.function_over_hypers(obj_model.predict, x, compute_grad=False)
                    return float(mn)

                opt.set_min_objective(f)

                maxEvals = 1000
                opt.set_xtol_abs(self.tolerance)
                opt.set_maxeval(maxEvals)

                x_opt = opt.optimize(current_best_location.copy())

                returncode = opt.last_optimize_result()
                
                y_opt = f(x_opt, np.array([]))

                # overwrite the current best if optimization succeeded
                if (returncode > 0 or returncode==-4) and y_opt < current_best_value:
                    if returncode == 5:
                        logging.debug('Optimization reach max evals of %d.' % maxEvals)
                    logging.debug('Optimizing improved the best by %f.' % self.objective.unstandardize_variance(current_best_value-y_opt))
                    current_best_location = x_opt
            else:
                logging.info('Optimizing current best with scipy l_BFGS')
                
                def f(x):
                    if x.ndim == 1:
                        x = x[None,:]
                    mn, var, mn_grad, var_grad = obj_model.function_over_hypers(obj_model.predict, x, compute_grad=True)
                    return (mn.flatten(), mn_grad.flatten())

                bounds = [(0.0,1.0)]*self.num_dims

                x_opt, y_opt, opt_info = spo.fmin_l_bfgs_b(f, current_best_location.copy(), 
                                                           bounds=bounds, disp=0)

                # make sure bounds were respected
                x_opt[x_opt > 1.0] = 1.0
                x_opt[x_opt < 0.0] = 0.0

                if y_opt < current_best_value:
                    logging.debug('Optimizing improved the best by %f.' % self.objective.unstandardize_variance(current_best_value-y_opt))
                    current_best_location = x_opt


        # std_at_best = np.sqrt(obj_var[best_ind])
        current_best_location = current_best_location[None]
        current_best_value, var_at_best = obj_model.function_over_hypers(obj_model.predict, current_best_location)
        std_at_best = np.sqrt(var_at_best)

        # un-normalize the min of mean to original units
        unnormalized_best_value = self.objective.unstandardize_mean(self.objective.unstandardize_variance(current_best_value))
        unnormalized_std_at_best = self.objective.unstandardize_variance(std_at_best)
        # (this could throw an error in very weird case that the objective has a binomial/step likelihood)

        # Print out the minimum according to the model
        logging.info('\nMinimum expected objective value under model is %.5f (+/- %.5f), at location:' % (unnormalized_best_value, unnormalized_std_at_best))
        current_best_location_orig_space = self.input_space.from_unit(current_best_location).flatten()
	self.input_space.consider_single_point = True
        self.input_space.paramify_and_print(current_best_location_orig_space, left_indent=16)

        return unnormalized_best_value, current_best_location_orig_space

    def best_constrained_no_solution(self):            

        logging.info('\nNo feasible solution found (yet).\n')

        # Compute the product of the probabilities, and return None for the current best value
        total_probs = total_constraint_confidence_over_hypers(self.constraint_models, self.grid, compute_grad=False)
        best_probs_ind = np.argmax(total_probs)
        best_probs_location = self.grid[best_probs_ind,:]
        best_probs_value = np.max(total_probs)

        # logging.info('***Best total probs before opt: %f' % best_probs_value)
        # logging.info('***Grid len fit: %d' % total_probs.size)
        # logging.info('***Total probs at grid 0: %f' % total_constraint_confidence_over_hypers(self.constraint_models, self.grid[0][None], compute_grad=False))

        if self.options['optimize_best']:
            if nlopt_imported:
                # print 'Optimizing the current best with NLopt.'

                opt = nlopt.opt(self.nlopt_method, self.num_dims)
                opt.set_lower_bounds(0.0)
                opt.set_upper_bounds(1.0)

                # we want to MAXIMIZE the probability over all constraints
                def f(x, put_gradient_here):

                    if x.ndim == 1:
                        x = x[None,:]

                    if put_gradient_here.size > 0:
                        pv, pv_grad = total_constraint_confidence_over_hypers(self.constraint_models, x, compute_grad=True)
                        put_gradient_here[:] = pv_grad
                    else:
                        pv = total_constraint_confidence_over_hypers(self.constraint_models, x, compute_grad=False)

                    return float(pv) 

                opt.set_max_objective(f) # MAXIMIZE the probability
                opt.set_xtol_abs(self.tolerance)
                opt.set_maxeval(1000)

                # don't let this part take longer than the grid part
                opt.set_maxeval(self.options['grid_size'])

                x_opt = opt.optimize(best_probs_location.copy())

                returncode = opt.last_optimize_result()
                # y_opt = opt.last_optimum_value()
                y_opt = f(x_opt, np.array([]))

                # overwrite the current best if optimization succeeded
                if (returncode > 0 or returncode==-4) and y_opt > best_probs_value:
                    logging.debug('Optimizing improved the best by %f.' % abs(y_opt-best_probs_value))
                    best_probs_location = x_opt

            else:
                # Optimize with L_BFGS_B
                logging.debug('Optimizing the current best with scipy l_BFGS.')
                
                def f(x):
                    if x.ndim == 1:
                        x = x[None,:]
                    pv, pv_grad = total_constraint_confidence_over_hypers(self.constraint_models, x, compute_grad=True)
                    return (-pv.flatten(), -pv_grad.flatten())

                bounds = [(0.0,1.0)]*self.num_dims

                x_opt, y_opt, opt_info = spo.fmin_l_bfgs_b(f, best_probs_location.copy(), 
                                                           bounds=bounds, disp=0)
                y_opt = -y_opt # switch back to positive numbers

                # make sure bounds are respected
                x_opt[x_opt > 1.0] = 1.0
                x_opt[x_opt < 0.0] = 0.0

                if y_opt > best_probs_value:
                    logging.debug('Optimizing improved the best by %f.' % abs(best_probs_value-y_opt))
                    best_probs_location = x_opt

        best_probs_location = best_probs_location[None]

        # Recompute the probabilities
        best_total_probs_value = total_constraint_confidence_over_hypers(self.constraint_models, best_probs_location, compute_grad=False)
        logging.info('Maximum total probability of satisfying constraints = %.5f' % best_total_probs_value)

        for c, model in self.constraint_models_dict.iteritems():
            prob = constraint_confidence_over_hypers(model, best_probs_location, compute_grad=False)
            logging.info('  Probability of satisfying %18s constraint: %.5f' % (c, prob))

        logging.info('\nAt location:    ')
        best_probs_location_orig_space = self.input_space.from_unit(best_probs_location).flatten()
        self.input_space.paramify_and_print(best_probs_location_orig_space, left_indent=16)

	#import pdb; pdb.set_trace()
        return None, best_probs_location_orig_space

    def best_constrained_solution_exists(self, pc):
        # A feasible region has been found
        logging.info('Feasible solution found.\n')

        # Compute GP mean and find minimum
        obj_model = self.models[self.objective.name]

        mean, var = obj_model.function_over_hypers(obj_model.predict, self.grid)
        valid_mean = mean[pc]
        valid_var = var[pc]
        best_ind = np.argmin(valid_mean)
        current_best_location = (self.grid[pc])[best_ind,:]
        current_best_value = np.min(valid_mean)
        
        if self.options['optimize_best']:
            if nlopt_imported:

                opt = nlopt.opt(self.nlopt_method, self.num_dims)

                logging.info('Optimizing current best with NLopt, %s' % opt.get_algorithm_name())
                
                opt.set_lower_bounds(0.0)
                opt.set_upper_bounds(1.0)

                opt.set_xtol_abs(self.tolerance)

                # define the objective function
                # here we want to MAXIMIZE the probability
                # but NLopt minimizes... ok.
                def f(x, put_gradient_here):

                    if x.ndim == 1:
                        x = x[None,:]

                    if put_gradient_here.size > 0:
                        mn, var, mn_grad, var_grad = obj_model.function_over_hypers(obj_model.predict, x, compute_grad=True)
                        # set grad to the gradient, here
                        put_gradient_here[:] = mn_grad.flatten()
                    else:
                        mn, var = obj_model.function_over_hypers(obj_model.predict, x, compute_grad=False)
                    return float(mn)

                opt.set_min_objective(f)

                """
                Need to do it this way. Somehow defining individual constraint functions
                and using opt.add_inequality_constraint() does not work properly...
                """
                def g(put_result_here, x, put_gradient_here):

                    if x.ndim == 1:
                        x = x[None,:]

                    for i,constraint in enumerate(self.constraints):

                        if put_gradient_here.size > 0:
                            pv, pv_grad = constraint_confidence_over_hypers(self.models[constraint], x, compute_grad=True)
                            put_gradient_here[i,:] = -pv_grad # MINUS
                        else:
                            pv = constraint_confidence_over_hypers(self.models[constraint], x, compute_grad=False)

                        delta = self.tasks[constraint].options['delta']
                        put_result_here[i] = float(1.0-delta-pv) 

                # the NLOPT constraint tolerance-- the amount by which it is ok for NLOPT to 
                # violate the constraints
                tol = [self.tasks[constraint].options['constraint_tol'] for constraint in self.constraints]
                opt.add_inequality_mconstraint(g, tol)

                x_opt = opt.optimize(current_best_location.copy())

                returncode = opt.last_optimize_result()
                # y_opt = opt.last_optimum_value()
                y_opt = f(x_opt, np.array([]))

                # overwrite the current best if optimization succeeded
                if not (returncode > 0 or returncode==-4):
                    logging.debug('NLOPT returncode indicates failure--discarding')
                elif y_opt < current_best_value:
                    nlopt_constraints_results = np.zeros(self.numConstraints())
                    g(nlopt_constraints_results, x_opt, np.zeros(0))
                    # if np.all(nlopt_constraints_results<=tol):
                    logging.debug('Optimizing improved the best by %f.' % self.objective.unstandardize_variance(current_best_value-y_opt))    
                    current_best_location = x_opt
                    # else:
                    #     logging.debug('NLOPT violated %d constraint(s)--discarding.' % np.sum(nlopt_constraints_results>0)) 
                else:
                    logging.debug('NLOPT did not improve the objective--discarding.')

            else:
                # Optimize with SLSQP
                # See http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.optimize.fmin_slsqp.html
                logging.info('Optimizing current best with scipy SLSQP')

                def f(x):
                    if x.ndim == 1:
                        x = x[None]
                    mn, var = obj_model.function_over_hypers(obj_model.predict, x, compute_grad=False)
                    return mn.flatten()

                def f_prime(x):
                    if x.ndim == 1:
                        x = x[None]
                    mn, var, mn_grad, var_grad = obj_model.function_over_hypers(obj_model.predict, x, compute_grad=True)
                    return mn_grad.flatten()

                # with SLSQP in scipy, the constraints are written as c(x) >= 0
                def g(x):
                    if x.ndim == 1:
                        x = x[None]
                    
                    g_func = np.zeros(self.numConstraints())
                    for i_g,constraint in enumerate(self.constraints):
                        pv = constraint_confidence_over_hypers(self.models[constraint], x, compute_grad=False)
                        delta = self.tasks[constraint].options['delta']
                        g_func[i_g] = (pv-(1.0-delta)).flatten()
                    return g_func

                def g_prime(x):
                    if x.ndim == 1:
                        x = x[None]

                    g_grad_func = np.zeros((self.numConstraints(), self.num_dims))
                    for i_g,constraint in enumerate(self.constraints):
                        pv, pv_grad = constraint_confidence_over_hypers(self.models[constraint], x, compute_grad=True)
                        g_grad_func[i_g,:] = pv_grad.flatten()
                    return g_grad_func

                bounds = [(0.0,1.0)]*self.num_dims

                # todo: set tolerance
                x_opt = spo.fmin_slsqp(f, current_best_location.copy(), acc=1e-5,
                    bounds=bounds, iprint=0, fprime=f_prime, f_ieqcons=g, fprime_ieqcons=g_prime)
                # make sure bounds are respected
                x_opt[x_opt > 1.0] = 1.0
                x_opt[x_opt < 0.0] = 0.0

                y_opt = f(x_opt)

                if y_opt < current_best_value and np.all(g(x_opt)>=0):
                    logging.debug('Optimizing improved the best by %f.' % self.objective.unstandardize_variance(current_best_value-y_opt))
                    current_best_location = x_opt
                elif y_opt < current_best_value:
                    logging.debug('SLSQP violated the constraints--discarding.')
                else:
                    logging.debug('SLSQP did not improve the objective--discarding.')


        current_best_location = current_best_location[None]

        current_best_value, var_at_best = obj_model.function_over_hypers(obj_model.predict, current_best_location)
        std_at_best = np.sqrt(var_at_best)
        # ind = np.argmin(valid_mean)
        # current_best_value = valid_mean[ind]
        # std_at_best = np.sqrt(valid_var[ind])

        unnormalized_best = self.objective.unstandardize_mean(self.objective.unstandardize_variance(current_best_value))
        unnormalized_std_at_best = self.objective.unstandardize_variance(std_at_best) # not used -- not quite

        # right to report this -- i mean there is uncertainty in the constraints too
        # this is the variance at that location, not the standard deviation of the minimum... 
        # not sure if this distinction is a big deal

        conf_string = ','.join(['%s:%.1f%%' % (constraint, 100.0*constraint_confidence_over_hypers(self.models[constraint], current_best_location, compute_grad=False)) for constraint in self.constraints])
        logging.info('\nMinimum expected objective value satisfying constraints w/ high prob (%s): %f\n' % (conf_string, unnormalized_best))
        logging.info('At location:    ')
        current_best_location_orig_space = self.input_space.from_unit(current_best_location).flatten()
        self.input_space.paramify_and_print(current_best_location_orig_space, left_indent=16)

        # Return according to model, not observed
        return unnormalized_best, current_best_location_orig_space

    # Compute the best OBSERVED value seen so far, when there are no constraints
    def bestObservedUnconstrained(self):
        vals = self.objective.values # these are not normalized (right?!)
        inps = self.objective.inputs
        best_observed_value = np.min(vals)
        best_observed_location = inps[np.argmin(vals),:]

        # Don't need to un-normalize inputs here because these are the raw inputs
        logging.info('\nMinimum of observed values is %f, at location:' % best_observed_value)
	self.input_space.consider_single_point = True
        self.input_space.paramify_and_print(best_observed_location, left_indent=16)

        return best_observed_value, best_observed_location

    # Compute the obser_obser best, i.e. the best observation
    # So we must have made an observation that satisfied the constraints and also was the best
    def bestObservedConstrained(self):
        # Compute the best value OBSERVED so far
        # with np.errstate(invalid='ignore'):

        # First: At what inputs "x" are all the constraints satisfied?
        
        # If different tasks are evaluated at different inputs (descoupled scenario) then
        # this does not make sense and we return None here
        # (this is also why we have the obser_model type of recommendations)
        if len({self.tasks[t].values.size for t in self.tasks}) != 1:
            return None, None

        all_constraints_satisfied = np.all([self.constraintSatisfiedAtObservedInputs(c) for c in self.constraints], axis=0)

        if not np.any(all_constraints_satisfied):
            logging.info('No observed result satisfied all constraints.\n')
            return None, None
        else:
            inps = self.objective.inputs
            vals = self.objective.values
            # get rid of those that violate constraints
            vals[np.logical_not(all_constraints_satisfied)] = np.max(vals)            
            # get rid of NaNs -- set them to biggest not-nan value, then they won't be the minimum
            vals[np.isnan(vals)] = np.max(vals[np.logical_not(np.isnan(vals))])
            best_observed_value = np.min(vals)
            best_observed_location = inps[np.argmin(vals),:]
            # Don't need to un-normalize inputs here because these are the raw inputs
            logging.info('\nBest observed values satisfying constraints is %f, at location:' % best_observed_value)
            self.input_space.paramify_and_print(best_observed_location, left_indent=16)

            # would be better to return these, but I want to maintain backward compatibility
            return best_observed_value, best_observed_location

    # this is a "type 3" recommendation (see above)
    # in the case when there are constraints. basically, get the best observation of the objective
    # that satisfied the probabilistic constraints ACCORDING TO THE MODEL
    # assuming that a solutions exists
    def best_obser_model_constrained_solution_exists(self):
        vals = self.objective.values # these are not normalized (right?!)
        inps = self.objective.inputs

        pc_at_objective_observations = self.probabilistic_constraint_over_hypers(self.input_space.to_unit(inps))

        if not np.any(pc_at_objective_observations):
            return None, None

        valid_inps = inps[pc_at_objective_observations]
        valid_vals = vals[pc_at_objective_observations]

        best_index = np.argmin(valid_vals)
        best_observed_value = np.min(valid_vals)
        best_observed_location = valid_inps[best_index]

        return best_observed_value, best_observed_location


    # At which of its observed inputs is constraint c satisfied?
    def constraintSatisfiedAtObservedInputs(self, c, values=None):
        task = self.tasks[c]
        model = self.models[c]
        if values is None:
            values = task.values
        if model.options['likelihood'].lower() in ['binomial', 'step']:
            sat = values/float(model.options['binomial_trials']) >= model._one_minus_epsilon
        else:
            # we can use greater_equal rather than strictly greater() because we catch
            # the binomial/step likelihoods in the case above. if not we'd have to use greater
            # to catch the 1/0
            sat = np.greater_equal(values, 0.0)
        return sat

    def numConstraints(self):
        return len(self.constraints)


    def print_images(self):
	
	import matplotlib.pyplot as plt
        spacing = np.linspace(0,1,1000)[:,None]

	tasks = self.tasks.keys()

	inputs = None
	for obj in tasks:
		if inputs is None:
			inputs  = self.input_space.from_unit(self.models[ obj ].inputs)
		else:
			inputs_tmp  = self.input_space.from_unit(self.models[ obj ].inputs)
			
			for instance in inputs_tmp:
#				if not instance in inputs:
				if np.min(cdist(instance.reshape(1, len(instance)), inputs)) > 0:
					inputs = np.vstack((inputs, instance))

	acq_ap = function_over_hypers(self.models.values(), self.acq.acquisition,
                                self.objective_model_dict, self.constraint_models_dict,
                                spacing, self.stored_recommendation['model_model_value'], compute_grad=False, tasks=tasks)

	fig = plt.figure()
	plt.plot(inputs, inputs * 0, color='black', marker='x', markersize=10, linestyle='None')
	plt.plot(self.input_space.from_unit(spacing), acq_ap,  color='blue', marker='.', markersize=1)
	plt.savefig('./figures/' + str(inputs.shape[ 0 ]) + '-acq.pdf', format='pdf', dpi=1000)
	plt.close(fig)

	mean = dict()
	var = dict()

	for key in tasks:
		mean[ key ], var[ key ] = self.models[ key ].function_over_hypers(self.models[ key ].predict, spacing)
		mean[ key ] = self.objectives[ key ].unstandardize_mean(self.objectives[ key ].unstandardize_variance(mean[ key ]))
		var[ key ] = self.objectives[ key ].unstandardize_variance(var[ key ])

	fig = plt.figure()

	n_task = 0
	color = ['red','green', 'blue']
	marker = ['x','o','*']
	for key in tasks:
		plt.plot(self.input_space.from_unit(self.models[ key ].inputs), \
			self.objectives[ key ].unstandardize_mean(self.objectives[ key ].unstandardize_variance(self.models[ key ].values)), \
			color=color[ n_task ], marker=marker[ n_task ], markersize=10, linestyle='None')

		plt.plot(self.input_space.from_unit(spacing), mean[ key ], color = color[ n_task ], marker = '.')
		plt.plot(self.input_space.from_unit(spacing), mean[ key ] + np.sqrt(var[ key ]), \
			color = color[ n_task ], marker = '.', markersize = 1)
		plt.plot(self.input_space.from_unit(spacing), mean[ key ] - np.sqrt(var[ key ]), \
			color = color[ n_task ], marker = '.', markersize = 1)
		n_task += 1

	plt.savefig('./figures/' + str(inputs.shape[ 0 ]) + '-posterior.pdf', format='pdf', dpi=1000)
	plt.close(fig)
	
	fig = plt.figure()

	n_task = 0
	color = ['red','green', 'blue']
	marker = ['x','o','*']
	for key in tasks:
		plt.plot(self.input_space.from_unit(self.models[ key ].inputs), \
			self.objectives[ key ].unstandardize_mean(self.objectives[ key ].unstandardize_variance(self.models[ key ].values)), \
			color=color[ n_task ], marker=marker[ n_task ], markersize=10, linestyle='None')

		plt.plot(self.input_space.from_unit(spacing), mean[ key ], color = color[ n_task ], marker = '.')
		plt.plot(self.input_space.from_unit(spacing), mean[ key ] + np.sqrt(var[ key ]), \
			color = color[ n_task ], marker = '.', markersize = 1)
		plt.plot(self.input_space.from_unit(spacing), mean[ key ] - np.sqrt(var[ key ]), \
			color = color[ n_task ], marker = '.', markersize = 1)
		n_task += 1

        plt.plot(self.stored_recommendation['model_model_input'][:,0], 0 * self.stored_recommendation['model_model_input'][:,0], \
		'black', marker = 'x')

	plt.savefig('./figures/' + str(inputs.shape[ 0 ]) + '-pareto-set.pdf', format='pdf', dpi=1000)
	plt.close(fig)

	if len(tasks) == 2:

		fig = plt.figure()

		plt.plot(mean[ tasks[ 0 ] ], mean[ tasks[ 1 ] ], color = 'red', marker = 'x', markersize = 2)

		plt.plot(self.stored_recommendation['model_model_value'][:,0], self.stored_recommendation['model_model_value'][:,1], \
			'b.', marker = 'x', markersize = 3)

		plt.savefig('./figures/' + str(inputs.shape[ 0 ]) + '-frontier.pdf', format='pdf', dpi=1000)
		plt.close(fig)

	else:

		from mpl_toolkits.mplot3d import Axes3D
		
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.plot(mean[ tasks[ 0 ] ], mean[ tasks[ 1 ] ], mean[ tasks[ 2 ] ], c = 'red')
		ax.scatter(self.stored_recommendation['model_model_value'][:,0], \
			self.stored_recommendation['model_model_value'][:,1], \
			self.stored_recommendation['model_model_value'][:,2], c = 'blue', marker ='x')

		plt.savefig('./figures/' + str(inputs.shape[ 0 ]) + '-frontier.pdf', format='pdf', dpi=1000)
		plt.close(fig)


    def print_random_contour(self):
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        size = 32
        x = np.linspace(0, 1, size)
        y = np.linspace(0, 1, size)
        X, Y = np.meshgrid(x, y)
        spacing = np.zeros((size * size, 2))

        for i in range(size):
                for j in range(size):
                        spacing[ i + j * size, 0 ] = X[ i, j ]
                        spacing[ i + j * size, 1 ] = Y[ i, j ]

                        #This is useless...
                        #from_unit = self.input_space.from_unit(spacing[ i + j * size, ])
                        from_unit = spacing[ i + j * size, ]
                        X[ i, j ] = from_unit[ 0 ]
                        Y[ i, j ] = from_unit[ 1 ]

        acq_ap = np.random.random(spacing.shape[0])

        fig = plt.figure()
        im = plt.imshow(acq_ap.reshape((size, size)).T, interpolation = 'bilinear', origin = 'lower', 
                cmap = cm.gray, extent = (0, 1, 0, 1))
        CS = plt.contour(X, Y, acq_ap.reshape((size, size)).T)
        plt.clabel(CS, inline=1, fontsize=10)
        plt.savefig('./figures/random-acq.pdf', format='pdf', dpi=1000)
        plt.close(fig)

    def print_images_batch(self):
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        size = 16
        x = np.linspace(0, 1, size)
        y = np.linspace(0, 1, size)
        X, Y = np.meshgrid(x, y)
        spacing = np.zeros((size * size, 2))
        
        for i in range(size):
                for j in range(size):
                        spacing[ i + j * size, 0 ] = X[ i, j ]
                        spacing[ i + j * size, 1 ] = Y[ i, j ]
                
                        #This is useless...
                        #from_unit = self.input_space.from_unit(spacing[ i + j * size, ])
                        from_unit = spacing[ i + j * size, ]
                        X[ i, j ] = from_unit[ 0 ]
                        Y[ i, j ] = from_unit[ 1 ]

        tasks = self.tasks.keys()
        objectives = self.objectives.keys()
        
        inputs = None
        for obj in tasks:
                if inputs is None:
                        inputs  = self.input_space.from_unit(self.models[ obj ].inputs)
                else:
                        inputs_tmp  = self.input_space.from_unit(self.models[ obj ].inputs)

                        for instance in inputs_tmp:
#                               if not instance in inputs:
                                if np.min(cdist(instance.reshape(1, len(instance)), inputs)) > 0:
                                        inputs = np.vstack((inputs, instance))

        n_total = inputs.shape[ 0 ]

        task_couplings = {task_name : self.tasks[task_name].options["group"] for task_name in self.tasks}

        tasks = self.tasks.keys()
      
        #This is for decoupled setting. 
        if len(set(task_couplings.values())) > 1:

                #Why this? It is coupled. Why is also passed the standardized optimum? 
                for obj in tasks:
                        inputs  = self.input_space.from_unit(self.models[ obj ].inputs)

                        acq_ap = function_over_hypers(self.models.values(), self.acq.acquisition,\
                                self.objective_model_dict, self.constraint_models_dict, \
                                spacing, (self.stored_recommendation['model_model_value'] - self.objective.standardization_mean) \
                                / self.objective.standardization_variance,\
                                compute_grad=False, tasks=[ obj ])

                        fig = plt.figure()
                        plt.plot(inputs[ 0, : ], inputs[ 1, : ], color='blue', marker='x', markersize=10, linestyle='None')
#                       im = plt.imshow(acq_ap.reshape((size, size)).T, interpolation = 'bilinear', origin = 'lower', 
#                               cmap = cm.gray, extent = (0, 1, 0, 1))
                        CS = plt.contour(X, Y, acq_ap.reshape((size, size)).T)
                        plt.clabel(CS, inline=1, fontsize=10)
                        plt.savefig('./figures/' + str(n_total) + '-' + obj + '-acq.pdf', format='pdf', dpi=1000)
                        plt.close(fig)
        else:
                inputs = None
                for obj in tasks:
                        if inputs is None:
                                inputs  = self.input_space.from_unit(self.models[ obj ].inputs)
                        else:
                                inputs_tmp  = self.input_space.from_unit(self.models[ obj ].inputs)

                                for instance in inputs_tmp:
#                                       if not instance in inputs:
                                        if np.min(cdist(instance.reshape(1, len(instance)), inputs)) > 0:
                                                inputs = np.vstack((inputs, instance))
                acq_ap = function_over_hypers(self.models.values(), self.acq.acquisition,\
                        self.objective_model_dict, self.constraint_models_dict, \
                        spacing, (self.stored_recommendation['model_model_value'] - self.objective.standardization_mean) \
                        / self.objective.standardization_variance,\
                        compute_grad=False, tasks=tasks, tasks_values = self.tasks)
                if not np.all(acq_ap == 0):
                    fig = plt.figure()
                    #This fails to print more that one input.
                    plt.plot(inputs[ ::2 ], inputs[ 1::2 ], color='blue', marker='x', markersize=10, linestyle='None')
                    im = plt.imshow(acq_ap.reshape((size, size)).T, interpolation = 'bilinear', origin = 'lower', cmap = cm.gray, extent = (0, 1, 0, 1))
                    CS = plt.contour(X, Y, acq_ap[0:size*size].reshape((size, size)).T)
                    plt.clabel(CS, inline=True, fontsize=10)
                    plt.savefig('./figures/' + str(n_total) + '-acq.pdf', format='pdf', dpi=1000)
                    plt.close(fig)
                    #import pdb; pdb.set_trace();
         
        return 

    def print_images_2d(self):
	
	import matplotlib.pyplot as plt
	import matplotlib.cm as cm

	size = 32
        x = np.linspace(0, 1, size)
        y = np.linspace(0, 1, size)
        X, Y = np.meshgrid(x, y)
	spacing = np.zeros((size * size, 2))

	for i in range(size):
		for j in range(size):
			spacing[ i + j * size, 0 ] = X[ i, j ]
			spacing[ i + j * size, 1 ] = Y[ i, j ]
			#from_unit = self.input_space.from_unit(spacing[ i + j * size, ])
			from_unit = self.input_space.from_unit(spacing[ i + j * size, ])
			X[ i, j ] = from_unit[ 0 ]
			Y[ i, j ] = from_unit[ 1 ]

	tasks = self.tasks.keys()
	objectives = self.objectives.keys()

	inputs = None
	for obj in tasks:
		if inputs is None:
			inputs  = self.input_space.from_unit(self.models[ obj ].inputs)
		else:
			inputs_tmp  = self.input_space.from_unit(self.models[ obj ].inputs)
			
			for instance in inputs_tmp:
#				if not instance in inputs:
				if np.min(cdist(instance.reshape(1, len(instance)), inputs)) > 0:
					inputs = np.vstack((inputs, instance))

	n_total = inputs.shape[ 0 ]

	task_couplings = {task_name : self.tasks[task_name].options["group"] for task_name in self.tasks}
			
	tasks = self.tasks.keys()

	if len(set(task_couplings.values())) > 1:

		for obj in tasks:
			inputs  = self.input_space.from_unit(self.models[ obj ].inputs)

			acq_ap = function_over_hypers(self.models.values(), self.acq.acquisition,\
				self.objective_model_dict, self.constraint_models_dict, \
				spacing, (self.stored_recommendation['model_model_value'] - self.objective.standardization_mean) \
				/ self.objective.standardization_variance,\
				compute_grad=False, tasks=[ obj ])

			fig = plt.figure()
			plt.plot(inputs[ :, 0 ], inputs[ :, 1 ], color='blue', marker='x', markersize=10, linestyle='None')
#			im = plt.imshow(acq_ap.reshape((size, size)).T, interpolation = 'bilinear', origin = 'lower', 
#				cmap = cm.gray, extent = (0, 1, 0, 1))
			CS = plt.contour(X, Y, acq_ap.reshape((size, size)).T)
			plt.clabel(CS, inline=1, fontsize=10)
			plt.savefig('./figures/' + str(n_total) + '-' + obj + '-acq.pdf', format='pdf', dpi=1000)
			plt.close(fig)

	else:

		inputs = None
		for obj in tasks:
			if inputs is None:
				inputs  = self.input_space.from_unit(self.models[ obj ].inputs)
			else:
				inputs_tmp  = self.input_space.from_unit(self.models[ obj ].inputs)
				
				for instance in inputs_tmp:
#					if not instance in inputs:
					if np.min(cdist(instance.reshape(1, len(instance)), inputs)) > 0:
						inputs = np.vstack((inputs, instance))

		acq_ap = function_over_hypers(self.models.values(), self.acq.acquisition,\
			self.objective_model_dict, self.constraint_models_dict, \
			spacing, (self.stored_recommendation['model_model_value'] - self.objective.standardization_mean) \
			/ self.objective.standardization_variance,\
			compute_grad=False, tasks=tasks, tasks_values = self.tasks)
#		import pdb; pdb.set_trace();
                if not np.all(acq_ap == 0):
		    fig = plt.figure()
		    plt.plot(inputs[ :, 0 ], inputs[ :, 1 ], color='blue', marker='x', markersize=10, linestyle='None')
#		    im = plt.imshow(acq_ap.reshape((size, size)).T, interpolation = 'bilinear', origin = 'lower', cmap = cm.gray, extent = (0, 1, 0, 1))
		    CS = plt.contour(X, Y, acq_ap[0:size*size].reshape((size, size)).T)
	    	    plt.clabel(CS, inline=True, fontsize=10)
		    plt.savefig('./figures/' + str(n_total) + '-acq.pdf', format='pdf', dpi=1000)
		    plt.close(fig)

	mean = dict()
	var = dict()

	for key in tasks:
		mean[ key ], var[ key ] = self.models[ key ].function_over_hypers(self.models[ key ].predict, spacing)

                if key in self.objectives:
                    mean[ key ] = self.objectives[ key ].unstandardize_mean(self.objectives[ key ].unstandardize_variance(mean[ key ]))
                    var[ key ] = self.objectives[ key ].unstandardize_variance(var[ key ])
		else:
                    mean[ key ] = mean[ key ]
                    var[ key ] = var[ key ]

	for key in tasks:
                if not np.all(mean[ key ] == mean[ key ][ 0 ]):
		    inputs  = self.input_space.from_unit(self.models[ key ].inputs)
		    fig = plt.figure()
		    plt.plot(inputs[ :, 0 ], inputs[ :, 1 ], color='blue', marker='x', markersize=10, linestyle='None')
#		    im = plt.imshow(mean[ key ].reshape((size, size)).T, interpolation = 'bilinear', origin = 'lower', \
#			cmap = cm.gray, extent = (0, 1, 0, 1))
		    CS = plt.contour(X, Y, mean[ key ].reshape((size, size)).T)
		    plt.clabel(CS, inline=1, fontsize=10)
		    if self.stored_recommendation['model_model_input'].ndim > 1:
	            	plt.plot(self.stored_recommendation['model_model_input'][:, 0 ], self.stored_recommendation['model_model_input'][:, 1], \
				'black', marker = 'o', linestyle = 'None')
		    else:
			plt.plot(self.stored_recommendation['model_model_input'][ 0 ], self.stored_recommendation['model_model_input'][ 1 ], \
                                'black', marker = 'o', linestyle = 'None')
		    plt.savefig('./figures/' + str(n_total) + '-' + key + '-mean.pdf', format='pdf', dpi=1000)
		    plt.close(fig)
                if not np.all(var[ key ] == var[ key ][ 0 ]):
		    fig = plt.figure()
		    plt.plot(inputs[ :, 0 ], inputs[ :, 1 ], color='blue', marker='x', markersize=10, linestyle='None')
#		    im = plt.imshow(var[ key ].reshape((size, size)).T, interpolation = 'bilinear', origin = 'lower', \
#	    		cmap = cm.gray, extent = (0, 1, 0, 1))
		    CS = plt.contour(X, Y, var[ key ].reshape((size, size)).T)
		    plt.clabel(CS, inline=1, fontsize=10)
		    if self.stored_recommendation['model_model_input'].ndim == 1: 
	            	plt.plot(self.stored_recommendation['model_model_input'][0], self.stored_recommendation['model_model_input'][1], \
	    			'black', marker = 'o', linestyle = 'None')
		    else:
	            	plt.plot(self.stored_recommendation['model_model_input'][:, 0 ], self.stored_recommendation['model_model_input'][:, 1], \
	    			'black', marker = 'o', linestyle = 'None')
		    plt.savefig('./figures/' + str(n_total) + '-' + key + '-var.pdf', format='pdf', dpi=1000)
		    plt.close(fig)

	if len(tasks) == 2 or (len(self.constraints) > 0 and len(self.objectives) == 2):

		fig = plt.figure()

		plt.plot(mean[ objectives[ 0 ] ], mean[ objectives[ 1 ] ], color = 'red', marker = 'x', markersize = 2, linestyle = 'None')

		plt.plot(self.stored_recommendation['model_model_value'][ :, 0 ], self.stored_recommendation['model_model_value'][ :, 1 ], \
			'b.', marker = 'x', markersize = 3, linestyle = 'None')

		plt.savefig('./figures/' + str(n_total) + '-frontier.pdf', format='pdf', dpi=1000)
		plt.close(fig)

	else:
		if len(tasks) > 2:
			from mpl_toolkits.mplot3d import Axes3D	
			fig = plt.figure()
			ax = fig.add_subplot(111, projection='3d')
			ax.scatter(mean[ objectives[ 0 ] ], mean[ objectives[ 1 ] ], mean[ objectives[ 2 ] ], c = 'red', marker = 'o', alpha = 0.1)
			ax.scatter(self.stored_recommendation['model_model_value'][:,0], \
				self.stored_recommendation['model_model_value'][:,1], \
				self.stored_recommendation['model_model_value'][:,2], c = 'blue', marker ='x')
			ax.view_init(30, -135)
			plt.savefig('./figures/' + str(n_total) + '-frontier.pdf', format='pdf', dpi=1000)
			plt.close(fig)

    def print_frontier(self):

        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        size = 32
        x = np.linspace(0, 1, size)
        y = np.linspace(0, 1, size)
        X, Y = np.meshgrid(x, y)
        spacing = np.zeros((size * size, 2))

        for i in range(size):
                for j in range(size):
                        spacing[ i + j * size, 0 ] = X[ i, j ]
                        spacing[ i + j * size, 1 ] = Y[ i, j ]
                        from_unit = self.input_space.from_unit(spacing[ i + j * size, ])
                        X[ i, j ] = from_unit[ 0 ]
                        Y[ i, j ] = from_unit[ 1 ]

        tasks = self.tasks.keys()
        objectives = self.objectives.keys()

        inputs = None
        for obj in tasks:
                if inputs is None:
                        inputs  = self.input_space.from_unit(self.models[ obj ].inputs)
                else:
                        inputs_tmp  = self.input_space.from_unit(self.models[ obj ].inputs)

                        for instance in inputs_tmp:
#                               if not instance in inputs:
                                if np.min(cdist(instance.reshape(1, len(instance)), inputs)) > 0:
                                        inputs = np.vstack((inputs, instance))

        n_total = inputs.shape[ 0 ]

        task_couplings = {task_name : self.tasks[task_name].options["group"] for task_name in self.tasks}

        tasks = self.tasks.keys()

        mean = dict()
        var = dict()

        for key in tasks:
                mean[ key ], var[ key ] = self.models[ key ].function_over_hypers(self.models[ key ].predict, spacing)

                if key in self.objectives:
                    mean[ key ] = self.objectives[ key ].unstandardize_mean(self.objectives[ key ].unstandardize_variance(mean[ key ]))
                    var[ key ] = self.objectives[ key ].unstandardize_variance(var[ key ])
                else:
                    mean[ key ] = mean[ key ]
                    var[ key ] = var[ key ]

        if len(tasks) == 2 or (len(self.constraints) > 0 and len(self.objectives) == 2):

                fig = plt.figure()

                plt.plot(mean[ objectives[ 0 ] ], mean[ objectives[ 1 ] ], color = 'red', marker = 'x', markersize = 2, linestyle = 'None')

                plt.plot(self.stored_recommendation['model_model_value'][ :, 0 ], self.stored_recommendation['model_model_value'][ :, 1 ], \
                        'b.', marker = 'x', markersize = 3, linestyle = 'None')

                plt.savefig('./figures/' + str(n_total) + '-frontier.pdf', format='pdf', dpi=1000)
                plt.close(fig)

        else:

                from mpl_toolkits.mplot3d import Axes3D

                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(mean[ objectives[ 0 ] ], mean[ objectives[ 1 ] ], mean[ objectives[ 2 ] ], c = 'red', marker = 'o', alpha = 0.1)
                ax.scatter(self.stored_recommendation['model_model_value'][:,0], \
                        self.stored_recommendation['model_model_value'][:,1], \
                        self.stored_recommendation['model_model_value'][:,2], c = 'blue', marker ='x')
                ax.view_init(30, -135)
                plt.savefig('./figures/' + str(n_total) + '-frontier.pdf', format='pdf', dpi=1000)
                plt.close(fig)

    def print_images_2d_random(self):
	
	import matplotlib.pyplot as plt
	import matplotlib.cm as cm

	size = 32
        x = np.linspace(0, 1, size)
        y = np.linspace(0, 1, size)
        X, Y = np.meshgrid(x, y)
	spacing = np.zeros((size * size, 2))

	for i in range(size):
		for j in range(size):
			spacing[ i + j * size, 0 ] = X[ i, j ]
			spacing[ i + j * size, 1 ] = Y[ i, j ]
			from_unit = self.input_space.from_unit(spacing[ i + j * size, ])
			X[ i, j ] = from_unit[ 0 ]
			Y[ i, j ] = from_unit[ 1 ]

	tasks = self.tasks.keys()
	objectives = self.objectives.keys()

	inputs = None
	for obj in tasks:
		if inputs is None:
			inputs  = self.input_space.from_unit(self.models[ obj ].inputs)
		else:
			inputs_tmp  = self.input_space.from_unit(self.models[ obj ].inputs)
			
			for instance in inputs_tmp:
#				if not instance in inputs:
				if np.min(cdist(instance.reshape(1, len(instance)), inputs)) > 0:
					inputs = np.vstack((inputs, instance))

	n_total = inputs.shape[ 0 ]

	task_couplings = {task_name : self.tasks[task_name].options["group"] for task_name in self.tasks}
			
	tasks = self.tasks.keys()

	mean = dict()
	var = dict()

	for key in tasks:
		mean[ key ], var[ key ] = self.models[ key ].function_over_hypers(self.models[ key ].predict, spacing)

                if key in self.objectives:
                    mean[ key ] = self.objectives[ key ].unstandardize_mean(self.objectives[ key ].unstandardize_variance(mean[ key ]))
                    var[ key ] = self.objectives[ key ].unstandardize_variance(var[ key ])
		else:
                    mean[ key ] = mean[ key ]
                    var[ key ] = var[ key ]

	for key in tasks:
		inputs  = self.input_space.from_unit(self.models[ key ].inputs)
		fig = plt.figure()
		plt.plot(inputs[ :, 0 ], inputs[ :, 1 ], color='blue', marker='x', markersize=10, linestyle='None')
#		im = plt.imshow(mean[ key ].reshape((size, size)).T, interpolation = 'bilinear', origin = 'lower', \
#			cmap = cm.gray, extent = (0, 1, 0, 1))
#		CS = plt.contour(X, Y, mean[ key ].reshape((size, size)).T)
#		plt.clabel(CS, inline=1, fontsize=10)
	        #plt.plot(self.stored_recommendation['model_model_input'][:, 0 ], self.stored_recommendation['model_model_input'][:, 1], \
			#'black', marker = 'o', linestyle = 'None')
		plt.savefig('./figures/' + str(n_total) + '-' + key + '-observations.pdf', format='pdf', dpi=1000)
		plt.close(fig)
		#fig = plt.figure()
		#plt.plot(inputs[ :, 0 ], inputs[ :, 1 ], color='blue', marker='x', markersize=10, linestyle='None')
#		im = plt.imshow(var[ key ].reshape((size, size)).T, interpolation = 'bilinear', origin = 'lower', \
#			cmap = cm.gray, extent = (0, 1, 0, 1))
		#CS = plt.contour(X, Y, var[ key ].reshape((size, size)).T)
		#plt.clabel(CS, inline=1, fontsize=10)
	        #plt.plot(self.stored_recommendation['model_model_input'][:, 0 ], self.stored_recommendation['model_model_input'][:, 1], \
		#	'black', marker = 'o', linestyle = 'None')
	#	plt.savefig('./figures/' + str(n_total) + '-' + key + '-var.pdf', format='pdf', dpi=1000)
		#plt.close(fig)

	#if len(tasks) == 2 or (len(self.constraints) > 0 and len(self.objectives) == 2):
#
	#	fig = plt.figure()
#
	#	plt.plot(mean[ objectives[ 0 ] ], mean[ objectives[ 1 ] ], color = 'red', marker = 'x', markersize = 2, linestyle = 'None')
#
	#	plt.plot(self.stored_recommendation['model_model_value'][ :, 0 ], self.stored_recommendation['model_model_value'][ :, 1 ], \
		#	'b.', marker = 'x', markersize = 3, linestyle = 'None')
#
		#plt.savefig('./figures/' + str(n_total) + '-frontier.pdf', format='pdf', dpi=1000)
	#	plt.close(fig)

	#else:

		#from mpl_toolkits.mplot3d import Axes3D
		
		#fig = plt.figure()
		#ax = fig.add_subplot(111, projection='3d')
		#ax.scatter(mean[ objectives[ 0 ] ], mean[ objectives[ 1 ] ], mean[ objectives[ 2 ] ], c = 'red', marker = 'o', alpha = 0.1)
		#ax.scatter(self.stored_recommendation['model_model_value'][:,0], \
			#self.stored_recommendation['model_model_value'][:,1], \
			#self.stored_recommendation['model_model_value'][:,2], c = 'blue', marker ='x')
		#ax.view_init(30, -135)
		#plt.savefig('./figures/' + str(n_total) + '-frontier.pdf', format='pdf', dpi=1000)
		#plt.close(fig)

