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
import scipy.stats    as sps

from spearmint.grids import sobol_grid
from spearmint.acquisition_functions.abstract_acquisition_function import AbstractAcquisitionFunction
from spearmint.utils.moop            import _cull_algorithm
from spearmint.utils.hv import HyperVolume
from scipy.spatial.distance import cdist

from spearmint.utils.moop import _compute_pareto_front_and_set_summary_y_space


GRID_SIZE = 1000

BMOO_OPTION_DEFAULTS  = {
    'obj_low_border'      : -100,
    'obj_high_border'      : 100,
    'con_high_border'      : 100,
    'con_low_border'      : -100,
    'distribution_G_points_method': 'GRID',
    'bmoo_grid_size': 10000,
    'bmoo_max_ps_size' : 50,
    }

#Acquisition function dealing with multiple objectives and constraints based in the work of Paul Feliot:
#A Bayesian Approach to constrained single and multi-objective optimization.
class BMOO(AbstractAcquisitionFunction):

        cached_information = dict()
        n_objectives = 0
        n_constraints = 0
        n_objs_and_cons = 0
        
	def __init__(self, num_dims, verbose=True, input_space=None, grid=None, opt = None):

            # we want to cache these. we use a dict indexed by the state integer
            self.has_gradients = False
            self.num_dims = num_dims
            self.input_space = input_space
            self.options = BMOO_OPTION_DEFAULTS.copy()
            self.options.update(opt)

	def acquisition(self, obj_model_dict, con_model_dict, cand, current_best, compute_grad, minimize=True, tasks=None, tasks_values=None):

            models = obj_model_dict.values()

            # make sure all models are at the same state

            assert len({model.state for model in models}) == 1, "Models are not all at the same state"

            assert not compute_grad 

            # We check if we have already computed the cells associated to the model and other stuff            
            #key = tuple([ obj_model_dict[ obj ].state for obj in obj_model_dict ])
                        
	    self.generate_static_borders(obj_model_dict, con_model_dict)

            iteration = len(obj_model_dict.values()[0].values)
#	    key = str(iteration)
            key = tuple([ obj_model_dict[ obj ].state for obj in obj_model_dict ]) + tuple([ iteration ])
            BMOO.n_objectives = len(obj_model_dict)
            BMOO.n_constraints = len(con_model_dict)
	    BMOO.n_objs_and_cons = len(obj_model_dict) + len(con_model_dict)

	    # First iteration ?

	    if key not in BMOO.cached_information:
                        #new_points = self.get_new_points_current_state_by_sampling(obj_model_dict, con_model_dict, tasks_values)            
                        new_points = self.get_new_points_observations(obj_model_dict, con_model_dict, tasks_values)            
                        BMOO.cached_information[ key ] = dict()
			if self.options['distribution_G_points_method'] == 'ADAPTIVE':
				if ("parallel_sequential_evaluations" in self.options and \
                                        self.options["parallel_sequential_evaluations"] == iteration) or iteration == 1:
#					BMOO.cached_information[ key ]['y_n_minus_one'] = self.generate_grid(obj_model_dict, con_model_dict)
					BMOO.cached_information[ key ]['y_n_minus_one'] = self.generate_uniform_grid(obj_model_dict, con_model_dict)
        	                	BMOO.cached_information[ key ]['p_n_minus_one_points'] = self.recompute_pareto_set(np.array([]))  
                                        BMOO.cached_information[ key ]['p_n_points'] = self.recompute_pareto_set(np.array([])) 
				else:
                                        previous_key = tuple([ obj_model_dict[ obj ].state for obj in obj_model_dict ]) + tuple([ iteration-1 ])
					BMOO.cached_information[ key ]['y_n_minus_one'] = BMOO.cached_information[ previous_key ]['y_n']
        	                	BMOO.cached_information[ key ]['p_n_minus_one_points'] = BMOO.cached_information[ previous_key ]['p_n_points']   
                                        BMOO.cached_information[ key ]['p_n_points'] = self.recompute_pareto_set(new_points) 
                                
				BMOO.cached_information[ key ]['p_n_points'] = _compute_pareto_front_and_set_summary_y_space( \
                                	BMOO.cached_information[ key ]['p_n_points'], BMOO.cached_information[ key ]['p_n_points'], \
                               		self.options['bmoo_max_ps_size'])['frontier']
				BMOO.cached_information[ key ]['y_n'] = self.adaptive_multilevel_splitting_algorithm(\
					BMOO.cached_information[ key ]['y_n_minus_one'], \
					BMOO.cached_information[ key ]['p_n_minus_one_points'], \
					BMOO.cached_information[ key ]['p_n_points'])
			else:
				BMOO.cached_information[ key ]['p_n_points'] = self.recompute_pareto_set(new_points)

                                grid_size = self.options['bmoo_grid_size'] *(BMOO.n_objectives+BMOO.n_constraints)

				yn = self.generate_uniform_grid_yn(BMOO.cached_information[ key ]['p_n_points'], grid_size) 

                                while yn.shape[ 0 ] < 1000:
                                    grid_size *= 2
				    yn = self.generate_uniform_grid_yn(BMOO.cached_information[ key ]['p_n_points'], grid_size) 

				print "Size of initial grid:%d" % (grid_size)
				print "Size of grid:%d" % (yn.shape[ 0 ])
                        	print "Size of Pareto set:%d" % (BMOO.cached_information[ key ]['p_n_points'].shape[ 0 ])
				BMOO.cached_information[ key ]['y_n'] = yn

                        BMOO.cached_information[ key ]['volume_g_n'] = self.compute_volume_points_gn(BMOO.cached_information[ key ]['p_n_points'])

#            state = tuple([ obj_model_dict[ obj ].state for obj in obj_model_dict ])
#            print "Computing adq. for key " + str(state) + " and iteration " + str(iteration) + "."

#	    self.compute_acquisition_mc_bmoo(cand[ 0 : 10, : ], obj_model_dict, con_model_dict, new_points, tasks_values, n_samples = 10000)
#	    self.compute_acquisition(cand[ 0 : 10, : ], obj_model_dict, con_model_dict, BMOO.cached_information[key]['y_n'], tasks_values)

	    return self.compute_acquisition(cand, obj_model_dict, con_model_dict, BMOO.cached_information[ key ]['y_n'], tasks_values) \
		* BMOO.cached_information[ key ]['volume_g_n']

        def get_new_points_current_state_by_sampling(self, obj_model_dict, con_model_dict, tasks_values):

            new_values = np.zeros((obj_model_dict[ obj_model_dict.keys()[ 0 ] ].values.shape[ 0 ], BMOO.n_objs_and_cons))

	    k = 0
            for obj in obj_model_dict:
	       mean, variance = obj_model_dict[ obj ].predict(obj_model_dict[ obj ].inputs)
               values = tasks_values[ obj ].unstandardize_mean(tasks_values[obj].unstandardize_variance(\
			obj_model_dict[ obj ].sample_from_posterior_given_hypers_and_data( obj_model_dict[ obj ].inputs , 1)))
               new_values[ :, k ] = values
	       k += 1

	    k = 0
            for con in con_model_dict:
               mean, variance = con_model_dict[ con ].predict(con_model_dict[ con ].inputs)
	       values = con_model_dict[con].sample_from_posterior_given_hypers_and_data(con_model_dict[con].inputs, 1)
               new_values[ :, k + BMOO.n_objectives ] = values
	       k += 1

            assert np.all(new_values[ : , 0 : BMOO.n_objectives ] <= BMOO.y_o_upp)
            assert np.all(new_values[ : , 0 : BMOO.n_objectives ] >= BMOO.y_o_low)
            assert np.all(new_values[ : , BMOO.n_objectives : BMOO.n_objs_and_cons ] <= BMOO.y_c_upp)
            assert np.all(new_values[ : , BMOO.n_objectives : BMOO.n_objs_and_cons ] >= BMOO.y_c_low)

            return new_values
 
        def get_new_points_current_state(self, obj_model_dict, con_model_dict, tasks_values):

            new_values = np.zeros((obj_model_dict[ obj_model_dict.keys()[ 0 ] ].values.shape[ 0 ], BMOO.n_objs_and_cons))

	    k = 0
            for obj in obj_model_dict:
               values = tasks_values[ obj ].unstandardize_mean(tasks_values[obj].unstandardize_variance(\
			obj_model_dict[ obj ].predict(obj_model_dict[ obj ].inputs)[ 0 ]))

               new_values[ :, k ] = values
	       k += 1

	    k = 0
            for con in con_model_dict:
               values = con_model_dict[ con ].predict(con_model_dict[ con ].inputs)[ 0 ]

               new_values[ :, k + BMOO.n_objectives ] = values
	       k += 1

            assert np.all(new_values[ : , 0 : BMOO.n_objectives ] <= BMOO.y_o_upp)
            assert np.all(new_values[ : , 0 : BMOO.n_objectives ] >= BMOO.y_o_low)
            assert np.all(new_values[ : , BMOO.n_objectives : BMOO.n_objs_and_cons ] <= BMOO.y_c_upp)
            assert np.all(new_values[ : , BMOO.n_objectives : BMOO.n_objs_and_cons ] >= BMOO.y_c_low)

	    # We add the points due to the grid

#            grid = sobol_grid.generate(self.input_space.num_dims, self.input_space.num_dims * self.options['bmoo_grid_size'])
#            new_values_grid = np.zeros((grid.shape[ 0 ], BMOO.n_objs_and_cons))
#
#	    k = 0
#            for obj in obj_model_dict:
#               values = tasks_values[ obj ].unstandardize_mean(tasks_values[obj].unstandardize_variance(\
#			obj_model_dict[ obj ].predict(grid)[ 0 ]))
#
#               new_values_grid[ :, k ] = values
#	       k += 1
#
#	    k = 0
#            for con in con_model_dict:
#               values = con_model_dict[ con ].predict(grid)[ 0 ]
#
#              new_values_grid[ :, k + BMOO.n_objectives ] = values
#	       k += 1
#
#           assert np.all(new_values_grid[ : , 0 : BMOO.n_objectives ] <= BMOO.y_o_upp)
#            assert np.all(new_values_grid[ : , 0 : BMOO.n_objectives ] >= BMOO.y_o_low)
#            assert np.all(new_values_grid[ : , BMOO.n_objectives : BMOO.n_objs_and_cons ] <= BMOO.y_c_upp)
#            assert np.all(new_values_grid[ : , BMOO.n_objectives : BMOO.n_objs_and_cons ] >= BMOO.y_c_low)
#
#            return np.vstack((new_values_grid, new_values))
            return new_values
 
	def get_new_points_observations(self, obj_model_dict, con_model_dict, tasks_values):

            new_values = np.zeros((obj_model_dict[ obj_model_dict.keys()[ 0 ] ].values.shape[ 0 ], BMOO.n_objs_and_cons))

	    k = 0
            for obj in obj_model_dict:
               values = tasks_values[ obj ].unstandardize_mean(tasks_values[obj].unstandardize_variance(\
			obj_model_dict[ obj ].values))

               new_values[ :, k ] = values
	       k += 1

	    k = 0
            for con in con_model_dict:
               values = con_model_dict[ con ].values

               new_values[ :, k + BMOO.n_objectives ] = values
	       k += 1

            assert np.all(new_values[ : , 0 : BMOO.n_objectives ] <= BMOO.y_o_upp)
            assert np.all(new_values[ : , 0 : BMOO.n_objectives ] >= BMOO.y_o_low)
            assert np.all(new_values[ : , BMOO.n_objectives : BMOO.n_objs_and_cons ] <= BMOO.y_c_upp)
            assert np.all(new_values[ : , BMOO.n_objectives : BMOO.n_objs_and_cons ] >= BMOO.y_c_low)

	    # We add the points due to the grid

#            grid = sobol_grid.generate(self.input_space.num_dims, self.input_space.num_dims * self.options['bmoo_grid_size'])
#            new_values_grid = np.zeros((grid.shape[ 0 ], BMOO.n_objs_and_cons))
#
#	    k = 0
#            for obj in obj_model_dict:
#               values = tasks_values[ obj ].unstandardize_mean(tasks_values[obj].unstandardize_variance(\
#			obj_model_dict[ obj ].predict(grid)[ 0 ]))
#
#               new_values_grid[ :, k ] = values
#	       k += 1
#
#	    k = 0
#            for con in con_model_dict:
#               values = con_model_dict[ con ].predict(grid)[ 0 ]
#
#              new_values_grid[ :, k + BMOO.n_objectives ] = values
#	       k += 1
#
#           assert np.all(new_values_grid[ : , 0 : BMOO.n_objectives ] <= BMOO.y_o_upp)
#            assert np.all(new_values_grid[ : , 0 : BMOO.n_objectives ] >= BMOO.y_o_low)
#            assert np.all(new_values_grid[ : , BMOO.n_objectives : BMOO.n_objs_and_cons ] <= BMOO.y_c_upp)
#            assert np.all(new_values_grid[ : , BMOO.n_objectives : BMOO.n_objs_and_cons ] >= BMOO.y_c_low)
#
#            return np.vstack((new_values_grid, new_values))
            return new_values

        def get_new_points(self, obj_model_dict, con_model_dict, tasks_values):

	    initial_state = obj_model_dict[ obj_model_dict.keys()[ 0 ] ].state

            new_values = np.zeros((obj_model_dict[ obj_model_dict.keys()[ 0 ] ].values.shape[ 0 ], BMOO.n_objs_and_cons))

	    k = 0
            for obj in obj_model_dict:
               values = tasks_values[ obj ].unstandardize_mean(tasks_values[obj].unstandardize_variance(\
 			obj_model_dict[ obj ].function_over_hypers(obj_model_dict[ obj ].predict, obj_model_dict[ obj ].inputs)[ 0 ]))

               obj_model_dict[ obj ].set_state(initial_state)
               new_values[ :, k ] = values
	       k += 1

	    k = 0
            for con in con_model_dict:
               values = con_model_dict[ con ].function_over_hypers(con_model_dict[ con ].predict, con_model_dict[ con ].inputs)[ 0 ]

               con_model_dict[ con ].set_state(initial_state)
               new_values[ :, k + BMOO.n_objectives ] = values
	       k += 1

            assert np.all(new_values[ : , 0 : BMOO.n_objectives ] <= BMOO.y_o_upp)
            assert np.all(new_values[ : , 0 : BMOO.n_objectives ] >= BMOO.y_o_low)
            assert np.all(new_values[ : , BMOO.n_objectives : BMOO.n_objs_and_cons ] <= BMOO.y_c_upp)
            assert np.all(new_values[ : , BMOO.n_objectives : BMOO.n_objs_and_cons ] >= BMOO.y_c_low)

            return new_values
                        
        def generate_grid(self, obj_model_dict, con_model_dict):

            grid = sobol_grid.generate(len(obj_model_dict)+len(con_model_dict),\
                        GRID_SIZE)

            # Scale grid.

            grid[:,:BMOO.n_objectives] = grid[:,:BMOO.n_objectives] * \
                ( BMOO.y_o_upp - BMOO.y_o_low ) + BMOO.y_o_low
            grid[:,BMOO.n_objectives:] = grid[:,BMOO.n_objectives:] * \
                ( BMOO.y_c_upp - BMOO.y_c_low ) + BMOO.y_c_low
                
            return grid

        def generate_uniform_grid(self, obj_model_dict, con_model_dict):

            grid = np.random.uniform(size = ((GRID_SIZE, len(obj_model_dict)+len(con_model_dict))))

            # Scale grid.

            grid[:,:BMOO.n_objectives] = grid[:,:BMOO.n_objectives] * \
                ( BMOO.y_o_upp - BMOO.y_o_low ) + BMOO.y_o_low
            grid[:,BMOO.n_objectives:] = grid[:,BMOO.n_objectives:] * \
                ( BMOO.y_c_upp - BMOO.y_c_low ) + BMOO.y_c_low
                
            return grid
 
 
        def generate_static_borders(self, obj_model_dict, con_model_dict):

            n_objectives = len(obj_model_dict)
            n_constraints = len(con_model_dict)            
            BMOO.y_o_low = np.zeros([n_objectives])
            BMOO.y_o_upp = np.zeros([n_objectives])
            BMOO.y_c_low = np.zeros([n_constraints])
            BMOO.y_c_upp = np.zeros([n_constraints])
            
            # Modify this to simulate the borders.

#            BMOO.y_o_low[:] = -100.0 - 1e-10
#            BMOO.y_c_low[:] = -10.0 - 1e-10
#            BMOO.y_o_upp[:] = 100.0 + 1e-10
#            BMOO.y_c_upp[:] = 10.0 + 1e-10

            BMOO.y_o_low[:] = float(self.options["obj_low_border"]) - 1e-10
            BMOO.y_c_low[:] = float(self.options["con_low_border"]) - 1e-10
            BMOO.y_o_upp[:] = float(self.options["obj_high_border"]) + 1e-10
            BMOO.y_c_upp[:] = float(self.options["con_high_border"]) + 1e-10
            
        def return_array_difference(self, superarray, subarray):
            result = np.array([])
            elements_added=0
            for superelement in superarray:
                found = False
                for subelement in subarray:
                    if (subelement==superelement).all():
                        found = True
                        break
                if not found:
                    result = np.append(result, superelement)
                    elements_added+=1
            return result.reshape((elements_added,BMOO.n_constraints+BMOO.n_objectives))
            
        def set_is_contained(self, set_contained, other_set):
            contained = True
            for element in set_contained:
                element_found = False
                for other in other_set:
                    if (other==element).all():
                        element_found = True
                        break
                if not element_found:
                    contained = False
                    break
            return contained
            
        def point_in_set(self, point, _set):
            contained = False
            for element in _set:
                if (point==element).all():
                    contained = True
                    break
            return contained
            
        def adaptive_multilevel_splitting_algorithm(self, y_0, p_0, p_star, v=0.2):
            t = 0
            p_t = p_0
            y_t = np.copy(y_0)
            m = y_0.shape[0]
            print "AMS Algorithm"
            while not self.set_is_contained(p_star, p_t):
                print "P star points " + str(p_star)
                print "P_t points " + str(p_t)
                p = self.check_and_add_pstar_points_to_p(p_star, p_t, y_t, v, m)                            
                if not self.set_is_contained(p_star, p):
                    p = self.compute_p_u(p_star, p, v, m, y_t)
                p_t = p
                y_t = self.RRM_Algorithm_1(p_t, y_t)['points']
                t+=1
            return y_t

        def compute_p_u(self, p_star, p, v, m, y_t):
            p_star_minus_p = self.return_array_difference(p_star,p)
            y_star = p_star_minus_p[np.random.choice(len(p_star_minus_p), 1)[ 0 ]]
            constraint_part = y_star[ BMOO.n_objectives : BMOO.n_objs_and_cons  ]
            q_star = len([x for x in constraint_part if x <= 0])
            if q_star < BMOO.n_constraints:
                return self.compute_p_u_not_all_constraints_passed(constraint_part, v, m, y_star, p, y_t)
            else:
                return self.compute_p_u_all_constraints_passed(y_star, v, m, p, y_t)

        def compute_p_u_not_all_constraints_passed(self, constraint_part, v, m, y_star, p, y_t):
            y_anchor = BMOO.y_o_upp
            for k in range(len(constraint_part)):
                dimension = constraint_part[k]
                if dimension <= 0:
                    y_anchor = np.append(y_anchor, 0.0)
                else:
                    y_anchor = np.append(y_anchor, BMOO.y_c_upp[k])
            y_anchor = y_anchor.reshape((1,BMOO.n_objectives+BMOO.n_constraints))
            p_u = self.find_no_killer_particles_point(v, m, y_star, y_anchor, p, y_t)
            return p_u
            
        def compute_p_u_all_constraints_passed(self, y_star, v, m, p, y_t):
            y_o_anchor = np.append(BMOO.y_o_upp,np.zeros([BMOO.n_constraints])).reshape((1, BMOO.n_objectives+BMOO.n_constraints))
            y_k_anchors = np.array([])
            for k in range(0, BMOO.n_constraints, 1):
                y_k_anchors = np.append(y_k_anchors, BMOO.y_o_upp)
                for j in range(0, BMOO.n_constraints, 1):
                    if j==k:
                        y_k_anchors = np.append(y_k_anchors, BMOO.y_c_upp[j])
                    else:
                        y_k_anchors = np.append(y_k_anchors, 0.0)
                y_k_anchors = y_k_anchors.reshape((k+1,BMOO.n_objectives+BMOO.n_constraints))
            particle_number = self.compute_particle_number(y_o_anchor, y_t)
            if particle_number >= v*m:
                p_u = self.find_no_killer_particles_point(v, m, y_star, y_o_anchor, p, y_t)
            else:
                print "Entering CPUACP else part"
                y_u_points = np.zeros([ y_k_anchors.shape[ 0 ], BMOO.n_objs_and_cons ])
                particle_number = 0
                alpha = 0.5
                u_border = 1.0
                l_border = 0.0
                point_found = False
                while not point_found:
                    p_u = np.copy(p)
                    alpha = (u_border + l_border) / 2.0
                    for k in range(0,len(y_k_anchors),1):
                        y_u_points[ k, : ] = self.generate_interpolation_point(y_o_anchor, y_k_anchors[ k, : ], alpha)
                        p_u = np.append(p_u, y_u_points[k]).reshape((len(p_u)+1,BMOO.n_objectives+BMOO.n_constraints))
                    particle_number = self.compute_particle_number(p_u, y_t)
                    if particle_number >= v*m-((v*m)*0.1) and particle_number <= v*m+((v*m)*0.1):
                        point_found = True
                    else:
                        if particle_number >= v*m-((v*m)*0.1):
                            l_border = alpha
                        else:
                            u_border = alpha
            return p_u
            
        def check_and_add_pstar_points_to_p(self, p_star, p_t, y_t, v, m):
            p = np.copy(p_t)
            added = 0
            for point in p_star:
                if not self.point_in_set(point, p):
                    p_try = np.append(p, point).reshape((len(p)+1,len(point)))
                    particle_number = self.compute_particle_number(p_try, y_t)
                    print "Trying to add point " + str(point)
                    print "Particles alive " + str(particle_number)
                    if particle_number >= v*m:
                        p = np.append( p, point ).reshape((len(p)+1,len(point)))                
                        added += 1
            return p
            
        def find_no_killer_particles_point(self, v, m, y_star, y_o_anchor, p, y_t):
            particle_number = 0
            u_border = 1.0
            l_border = 0.0        
            point_found = False
            while not point_found:
                alpha = (u_border + l_border) / 2.0
                y_u = self.generate_interpolation_point(y_star, y_o_anchor, alpha)
                p_u = np.append(p, y_u).reshape((len(p)+1,BMOO.n_objectives+BMOO.n_constraints))
                particle_number = self.compute_particle_number(p_u, y_t)
                if particle_number >= v*m-((v*m)*0.1) and particle_number <= v*m+((v*m)*0.1):
                    point_found = True
                else:
                    if particle_number >= v*m-((v*m)*0.1):
                        l_border = alpha
                    else:
                        u_border = alpha
            return p_u
            
        def RRM_Algorithm_1(self, G_n, y_n_minus_1):
            
            dimensions = BMOO.n_constraints + BMOO.n_objectives
            #Remove.
            m = len(y_n_minus_1)
            y_n_0 = np.array([])
            added = 0
            for particle in y_n_minus_1:
                if not self.particle_is_dominated(particle, G_n):
                    y_n_0 = np.append(y_n_0, particle).reshape((added+1,dimensions))
                    added += 1
                
            m_0 = len(y_n_0)

            #Resample.
           
            y_n_1 = np.append(y_n_0, y_n_0[ np.random.choice(np.arange(y_n_0.shape[ 0 ]), m - m_0) ])\
                .reshape((m, BMOO.n_constraints+BMOO.n_objectives))
            
            #Move.

            return self.metropolis_hastings_algorithm(y_n_1, G_n)
            
#        def metropolis_hastings_algorithm(self, old_points, target_distribution_points, steps = 500, \
#		desired_accepted_rate = 0.2, scale_cov_factor = 1.0):
#
#            #Target distribution is uniform in a set of points. If the point belongs to Gn its value is 1, otherwise is 0.
#            #Proposal distribution is a Gaussian Distribution with parameters fitting the input points.
#            #3 steps for each particle, Gaussian Random Walk.
#            #counter = 0
#            #particles_moved = False
#            #while counter < 20 and not particles_moved:
#
#            min_var = np.min(np.var(old_points, axis = 0)) * scale_cov_factor**2
#            accepts = np.zeros((old_points.shape[ 0 ], steps))
#
#            for i in range(steps):
#
#            	new_points = old_points.copy()
#
#                for j in range(old_points.shape[ 0 ]):
#
#                    candidate = old_points[ j, : ] + np.random.normal(size = old_points.shape[ 1 ]) * np.sqrt(min_var)
#
#                    if not self.particle_is_dominated(candidate, target_distribution_points) and self.particle_is_in_hypercube(candidate):
#                        new_points[ j, : ] = candidate
#                        accepts[ j, i ] += 1
#
#                old_points = new_points
#
#            accepted_rate = np.min(np.mean(accepts, axis = 1))
#
#            print "MH Algorithm, Accepted rate: " + str(accepted_rate) + " %" + " Min_sd: " + str(np.sqrt(min_var))
#
#            return {'points':new_points, 'accepted_rate':accepted_rate}

        def metropolis_hastings_algorithm(self, old_points, target_distribution_points, steps = 100, \
		desired_accepted_rate = 0.2, scale_cov_factor = 1.0):

            #Target distribution is uniform in a set of points. If the point belongs to Gn its value is 1, otherwise is 0.
            #Proposal distribution is a Gaussian Distribution with parameters fitting the input points.
            #Or an isotropic gaussian distribution with mean equal to zero and the minimum variance.
            #3 steps for each particle, Gaussian Random Walk.
            #counter = 0
            #particles_moved = False
            #while counter < 20 and not particles_moved:

	    pareto_points = self.apply_extended_domination_rule_to_frontier_mc(target_distribution_points)
            pareto_indices = _cull_algorithm(pareto_points)
	    pareto_points = pareto_points[ pareto_indices, : ]

            min_var = np.min(np.var(old_points, axis = 0)) * scale_cov_factor**2

            flag = True

            while flag:

                accepts = np.zeros((old_points.shape[ 0 ], steps))

                for i in range(steps):

                    new_points = old_points.copy()

                    for j in range(old_points.shape[ 0 ]):

                        candidate = old_points[ j, : ] + np.random.normal(size = old_points.shape[ 1 ]) * np.sqrt(min_var)

                        if not self.particle_is_dominated_original_space(candidate, pareto_points) and self.particle_is_in_hypercube(candidate):
                            new_points[ j, : ] = candidate
                            accepts[ j, i ] += 1

                    old_points = new_points

                accepted_rate = np.min(np.mean(accepts, axis = 1))

                if (accepted_rate > desired_accepted_rate):
                    flag = False
                else:
                    print "Rerunning MH Accepted - rate: " + str(accepted_rate) + " %" + " Min_sd: " + str(np.sqrt(min_var))
                    min_var *= 0.5

            print "Successful MH Algorithm, Accepted rate: " + str(accepted_rate) + " %" + " Min_sd: " + str(np.sqrt(min_var))

            return {'points':new_points, 'accepted_rate':accepted_rate}
            
        def particle_is_in_hypercube(self, candidate):
            return np.all(candidate[ 0 : BMOO.n_objectives ] >= BMOO.y_o_low) and \
                np.all(candidate[ 0 : BMOO.n_objectives ] <= BMOO.y_o_upp) and \
                np.all(candidate[ BMOO.n_objectives : BMOO.n_objs_and_cons ] >= BMOO.y_c_low) and \
                np.all(candidate[ BMOO.n_objectives : BMOO.n_objs_and_cons ] <= BMOO.y_c_upp)
                
        def generate_interpolation_point(self, y_star, y_anchor, alpha):
            return y_anchor + alpha * (y_star - y_anchor)
            
        def compute_particle_number(self, p_try, y_t):
            particle_number = 0
            for particle in y_t:
                if not self.particle_is_dominated(particle, p_try):
                    particle_number += 1
            return particle_number   
            
        def particle_is_dominated(self, particle, frontier):

	    # DHL XXX Inefficient in metropolis-hasting
	
            particle_extended, frontier_extended = self.apply_extended_domination_rule(particle, frontier)
            dominated = False
            for pareto_particle in frontier_extended:
                if np.all(particle_extended >= pareto_particle) and np.any(particle_extended > pareto_particle):
                        dominated = True
                        break
                if dominated:
                    break
            return dominated

        def particle_is_dominated_original_space(self, particle, frontier):
            dominated = False
            for pareto_particle in frontier:
                if np.all(particle >= pareto_particle) and np.any(particle > pareto_particle):
                        dominated = True
                        break
                if dominated:
                    break
            return dominated
            
        def recompute_pareto_set(self, points):

            if len(points) == 0:
                return np.array([])

            objective_part = points[ :, 0 : BMOO.n_objectives ]
            constraint_part = points[ :, BMOO.n_objectives : (BMOO.n_objectives + BMOO.n_constraints) ]

            assert np.all(objective_part <= BMOO.y_o_upp)
            assert np.all(objective_part >= BMOO.y_o_low)
            assert np.all(constraint_part <= BMOO.y_c_upp)
            assert np.all(constraint_part >= BMOO.y_c_low)

            values, values_not_modified = self.apply_extended_domination_rule_to_values(objective_part, constraint_part)
            pareto_indices = _cull_algorithm(values)

            return values_not_modified[ pareto_indices, : ]
                
                
        def apply_extended_domination_rule(self, particle, frontier):

            extended_frontier = np.zeros(frontier.shape)

            if np.any(particle[ BMOO.n_objectives : BMOO.n_objs_and_cons ] > 0):
                objective_values = BMOO.y_o_upp
                constraint_values = np.maximum(0.0, particle[ BMOO.n_objectives : BMOO.n_objs_and_cons ])
            else:
                objective_values = particle[ 0 : BMOO.n_objectives ]
                constraint_values = np.zeros(BMOO.n_constraints)

            extended_particle = np.append(objective_values, constraint_values)

            val_index = 0
            for value in frontier:
                if np.any(value[ BMOO.n_objectives : BMOO.n_objs_and_cons ] > 0):                            
                    objective_values = BMOO.y_o_upp                                                             
                    constraint_values = np.maximum(0.0, value[ BMOO.n_objectives : BMOO.n_objs_and_cons ])   
                else:                                                                                           
                    objective_values = value[ 0 : BMOO.n_objectives ]                                        
                    constraint_values = np.zeros(BMOO.n_constraints)                                            
                extended_frontier[ val_index, : ] = np.append(objective_values, constraint_values)
                val_index += 1
            return extended_particle, extended_frontier
            
        def apply_extended_domination_rule_to_values(self, obj_mean_values, con_mean_values):

            new_values = np.zeros((obj_mean_values.shape[ 0 ], obj_mean_values.shape[ 1 ] + con_mean_values.shape[ 1 ]))
            old_values = np.zeros((obj_mean_values.shape[ 0 ], obj_mean_values.shape[ 1 ] + con_mean_values.shape[ 1 ]))

            val_index=0
            for value in obj_mean_values:
                if np.any(con_mean_values[ val_index, : ] > 0):
                    objective_values = BMOO.y_o_upp
                    constraint_values = np.maximum(0.0, con_mean_values[ val_index, : ])
                    new_values[ val_index, : ] = np.append(objective_values, constraint_values)
                else:
                    objective_values = value
                    constraint_values = np.zeros(con_mean_values.shape[ 1 ])
                    new_values[ val_index, : ] = np.append(objective_values, constraint_values)

                old_values[ val_index, : ] = np.append(obj_mean_values[ val_index, : ],con_mean_values[ val_index, : ])
                val_index += 1
            return new_values, old_values
            
	# This method is the one that actually does the computation of the acquisition_function

	def compute_acquisition(self, cand, obj_model_dict, con_model_dict, evaluations, tasks_values):

            n_objectives = len(obj_model_dict)
            n_constraints = len(con_model_dict)

            # We compute the mean and the variances at each candidate point

            mean = np.zeros((cand.shape[ 0 ], n_objectives))
            mean_constraint = np.zeros((cand.shape[ 0 ], n_constraints))
            var = np.zeros((cand.shape[ 0 ], n_objectives))
            var_constraint = np.zeros((cand.shape[ 0 ], n_constraints))

            n_objective = 0
            for obj in obj_model_dict:
                mean[ :, n_objective ], var[ :, n_objective ] = obj_model_dict[ obj ].predict(cand) 
                mean[:, n_objective ] = tasks_values[obj].unstandardize_mean(tasks_values[obj].\
                    unstandardize_variance(mean[:, n_objective ]))
                var[:, n_objective ] = tasks_values[obj].unstandardize_variance(var[:, n_objective ])
                n_objective += 1
	
            n_constraint = 0
            for con in con_model_dict:
                mean_constraint[ :, n_constraint], var_constraint[ :, n_constraint] = con_model_dict[ con ].predict(cand)
                mean_constraint[ :, n_constraint] = mean_constraint[ :, n_constraint]
                var_constraint[ :, n_constraint] = var_constraint[ :, n_constraint]
                n_constraint += 1

            # We loop over the evaluations, computing the acquisition function for all of them.

 #           total_acquisition = np.zeros(cand.shape[ 0 ])
 #           for evaluation in evaluations:
 #               acquisition = np.zeros(cand.shape[ 0 ])
 #               objective_part = evaluation[:len(obj_model_dict)]
 #               constraint_part = evaluation[-len(con_model_dict):]
 #   
 #               if(np.all(constraint_part<=0)):                    
 #                   objective_factor = np.ones(cand.shape[ 0 ])
 #                   constraint_factor = np.ones(cand.shape[ 0 ])
 #                   for k in range(n_objectives):
 #                       objective_factor *= sps.norm.cdf((objective_part[ k ] - mean[ :, k ]) / np.sqrt(var[ : , k ]))
 #                   for c in range(n_constraints):
 #                       constraint_factor *= sps.norm.cdf((- mean_constraint[ :, c ]) / np.sqrt(var_constraint[ : , c ]))                        
 #                   acquisition = objective_factor * constraint_factor                  
 #               else:                    
 #                   constraint_factor = np.ones(cand.shape[ 0 ])
 #                   for c in range(n_constraints):
 #                       constraint_value = constraint_part[c] if constraint_part[c] > 0 else 0  
 #                       constraint_factor *= sps.norm.cdf((constraint_value - mean_constraint[ :, c ]) / np.sqrt(var_constraint[ : , c ]))
 #                   acquisition = constraint_factor
 #                   
 #               total_acquisition += acquisition
 #
 #          # Montecarlo Algorithm. Approximating the integral by averaging samples.
 #
 #           total_acquisition = total_acquisition / len(evaluations)

	    objective_part = evaluations[ :, 0 : len(obj_model_dict) ]
            constraint_part = evaluations[ :, len(obj_model_dict) : (len(obj_model_dict) + len(con_model_dict)) ]
        
            result = np.ones((cand.shape[ 0 ], evaluations.shape[ 0 ]))
            feasible_y = np.prod(constraint_part <= 0, axis = 1) == 1
	    infeasible_y = np.logical_not(feasible_y)
	    n_feasible_y = np.sum(feasible_y)
	    n_infeasible_y = np.sum(infeasible_y)
        
	    for k in range(n_objectives):
		result[ :, feasible_y ] *= sps.norm.cdf((np.outer(np.ones(cand.shape[ 0 ]), objective_part[ feasible_y, k ]) - \
			np.outer(mean[ :, k ], np.ones(n_feasible_y))) / np.outer(np.sqrt(var[ : , k ]), np.ones(n_feasible_y)))
	    for k in range(n_constraints):
		result[ :, feasible_y ] *= sps.norm.cdf((-np.outer(mean_constraint[ :, k ], np.ones(n_feasible_y))) \
			/ np.outer(np.sqrt(var_constraint[ : , k ]), np.ones(n_feasible_y)))
        
	    for k in range(n_constraints):
		result[ :, infeasible_y ] *= sps.norm.cdf((np.outer(np.ones(cand.shape[ 0 ]), \
			np.maximum(0.0, constraint_part[ infeasible_y, k ])) - np.outer(mean_constraint[ :, k ], \
			np.ones(n_infeasible_y))) / np.outer(np.sqrt(var_constraint[ : , k ]), np.ones(n_infeasible_y)))
	    
	    total_acquisition = np.mean(result, axis = 1)
        
            return total_acquisition
        
	#def plot_particles(self, yn):
	#	import matplotlib.pyplot as plt
	#	import matplotlib.cm as cm
	#	fig = plt.figure()
	#	plt.plot(yn[ :, 0 ], yn[ :, 1 ], color='blue', marker='x', markersize=10, linestyle='None')
	#	fig = plt.figure()
	#	plt.plot(yn[ :, 2 ], yn[ :, 3 ], color='red', marker='x', markersize=10, linestyle='None')

	def apply_extended_domination_rule_to_frontier_mc(self, frontier):
       
            extended_frontier = np.zeros((frontier.shape[ 0 ], BMOO.n_objectives+BMOO.n_constraints))
            n_extra_points = 0

            val_index=0
            extra_points = np.array([])
            for value in frontier:

                if np.any(value[ BMOO.n_objectives : BMOO.n_objs_and_cons ] > 0):

                    objective_values = BMOO.y_o_upp
                    constraint_values = np.maximum(0.0, value[ BMOO.n_objectives : BMOO.n_objs_and_cons ])

		    extra_constraint_values = constraint_values.copy()
		    extra_constraint_values[ extra_constraint_values == 0.0 ] = BMOO.y_c_low[ extra_constraint_values == 0.0 ]

		    if n_extra_points == 0:
		           extra_points = np.array([ np.append(np.ones(BMOO.n_objectives) * BMOO.y_o_low, extra_constraint_values).tolist() ])
		    else:
		           extra_points = np.concatenate((extra_points, np.array([ np.append(np.ones(BMOO.n_objectives) * \
				BMOO.y_o_low, extra_constraint_values) ])))
 
		    n_extra_points += 1
                else:
                    objective_values = value[ 0 : BMOO.n_objectives ]
                    constraint_values = np.zeros(BMOO.n_constraints)

		    if n_extra_points == 0:
		           extra_points = np.array([ np.append(objective_values, np.ones(BMOO.n_constraints) * BMOO.y_c_low).tolist() ])
		           extra_points = np.concatenate((extra_points, np.array([ np.append(np.ones(BMOO.n_objectives) * \
				BMOO.y_o_low, np.zeros(BMOO.n_constraints)) ])))
		    else:
 		           extra_points = np.concatenate((extra_points, np.array([ np.append(objective_values, \
				np.ones(BMOO.n_constraints) * BMOO.y_c_low) ])))
		           extra_points = np.concatenate((extra_points, np.array([ np.append(np.ones(BMOO.n_objectives) * \
				BMOO.y_o_low, np.zeros(BMOO.n_constraints)) ])))

		    # We add all constraints equal to Y_c_low that are infeasible

                    for i in range(1, 2**BMOO.n_constraints - 1):
			constraints = np.ones(BMOO.n_constraints) * BMOO.y_c_low
			for k in range(BMOO.n_constraints):
				if i & (1 << k) == 0:
					constraints[ k ] = 0.0
			extra_points = np.concatenate((extra_points, np.array([ np.append(np.ones(BMOO.n_objectives) * \
				BMOO.y_o_low, constraints) ])))

		    n_extra_points += 2 + len(range(1, 2**BMOO.n_constraints - 1))

                extended_frontier[ val_index, : ] = np.append(objective_values, constraint_values)
                val_index+=1

            for i in range(extra_points.shape[ 0 ]):
		if np.min(cdist(extra_points[ i : (i + 1), : ], extended_frontier)) > 1e-5:
			extended_frontier = np.concatenate((extended_frontier, extra_points[ i : (i + 1), : ]))

            return extended_frontier

	def apply_extended_domination_rule_to_particle_mc(self, particle):

            if np.any(particle[ BMOO.n_objectives : BMOO.n_objs_and_cons ] > 0):
                objective_values = BMOO.y_o_upp
                constraint_values = np.maximum(0.0, particle[ BMOO.n_objectives : BMOO.n_objs_and_cons ])
		extra_constraint_values = constraint_values.copy()
		extra_constraint_values[ extra_constraint_values == 0.0 ] = BMOO.y_c_low[ extra_constraint_values == 0.0 ]
                extra_points = np.array([ np.append(np.ones(BMOO.n_objectives) * BMOO.y_o_low, extra_constraint_values).tolist() ])

            else:

                objective_values = particle[ 0 : BMOO.n_objectives ]
                constraint_values = np.zeros(BMOO.n_constraints)

		extra_points = np.array([ np.append(objective_values, np.ones(BMOO.n_constraints) * BMOO.y_c_low).tolist() ])
		extra_points = np.concatenate((extra_points, np.array([ np.append(np.ones(BMOO.n_objectives) * \
			BMOO.y_o_low, np.zeros(BMOO.n_constraints)) ])))

                for i in range(1, 2**BMOO.n_constraints - 1):
			constraints = np.ones(BMOO.n_constraints) * BMOO.y_c_low
			for k in range(BMOO.n_constraints):
				if i & (1 << k) == 0:
					constraints[ k ] = 0.0
			extra_points = np.concatenate((extra_points, np.array([ np.append(np.ones(BMOO.n_objectives) * \
				BMOO.y_o_low, constraints) ])))

            extended_particle = np.append(objective_values, constraint_values)
            return np.concatenate((np.array([ extended_particle ]), extra_points))

	def compute_acquisition_mc_bmoo(self, cand, obj_model_dict, con_model_dict, frontier, tasks_values, n_samples = 1000):

            n_objectives = len(obj_model_dict)
            n_constraints = len(con_model_dict)

            samples = np.random.normal(size = (n_samples, n_objectives + n_constraints))

            mean = np.zeros((cand.shape[ 0 ], n_objectives))
            var = np.zeros((cand.shape[ 0 ], n_objectives))
            
            mean_constraints = np.zeros((cand.shape[0], n_constraints))
            var_constraints = np.zeros((cand.shape[0], n_constraints))
            
            n_objective = 0
            for obj in obj_model_dict:
                mean[ :, n_objective ], var[ :, n_objective ] = obj_model_dict[ obj ].predict(cand)
                mean[:, n_objective ] = tasks_values[obj].unstandardize_mean(tasks_values[obj].\
                    unstandardize_variance(mean[:, n_objective ]))
                var[:, n_objective ] = tasks_values[obj].unstandardize_variance(var[:, n_objective ])
                n_objective += 1

            n_constraint = 0
            for con in con_model_dict:
                mean_constraints[ :, n_constraint ], var_constraints[ :, n_constraint ] = \
                    con_model_dict[ con ].predict(cand) 	
                mean_constraints[ :, n_constraint] = mean_constraints[ :, n_constraint]
                var_constraints[ :, n_constraint] = var_constraints[ :, n_constraint]
                n_constraint += 1

            frontier = self.apply_extended_domination_rule_to_frontier_mc(frontier)
            reference_point = np.append(BMOO.y_o_upp, BMOO.y_c_upp)
            hv = HyperVolume(reference_point.tolist())
            hv_frontier = hv.compute(frontier.tolist())

            acquisition_values = np.zeros(cand.shape[ 0 ])

            for i in range(cand.shape[ 0 ]):

                value = 0.0			

                for j in range(n_samples):

                    new_point_frontier = samples[ j, : ] * np.sqrt(np.append(var[ i, : ], var_constraints[i, :])) + \
                        np.append(mean[ i, : ], mean_constraints[ i, : ])

                    new_points_frontier = self.apply_extended_domination_rule_to_particle_mc(new_point_frontier)
                    new_frontier = np.vstack((frontier, new_points_frontier))
                    new_hv_frontier = hv.compute(new_frontier.tolist())
                    value += np.maximum(0, new_hv_frontier - hv_frontier)
                    
                value /= n_samples
                acquisition_values[ i ] = value

            return acquisition_values

	def  compute_volume_points_gn(self, frontier):

		reference_point = np.append(BMOO.y_o_upp, BMOO.y_c_upp)
		frontier = self.apply_extended_domination_rule_to_frontier_mc(frontier)
	
		hv = HyperVolume(reference_point)
		hv_frontier = hv.compute(frontier.tolist())

		total_hv = np.prod(reference_point - np.append(BMOO.y_o_low, BMOO.y_c_low))

		return total_hv - hv_frontier

	def generate_sobol_grid_yn(self, frontier, size = 1000):

		grid = 	sobol_grid.generate(frontier.shape[ 1 ], size)
		grid[:,:BMOO.n_objectives] = grid[:,:BMOO.n_objectives] * ( BMOO.y_o_upp - BMOO.y_o_low ) + BMOO.y_o_low
		grid[:,BMOO.n_constraints:] = grid[:,BMOO.n_constraints:] * ( BMOO.y_c_upp - BMOO.y_c_low ) + BMOO.y_c_low
 
		frontier = self.apply_extended_domination_rule_to_frontier_mc(frontier)

		n_added = 0

		for point in grid:
			if not self.particle_is_dominated_original_space(point, frontier):
				if n_added == 0:
					new_grid = np.array([ point ])
				else:
					new_grid = np.concatenate((new_grid, np.array([ point ])))
				n_added += 1
		
		return new_grid	

	def generate_uniform_grid_yn(self, frontier, size = 1000):

		grid = np.random.uniform(size = ((size, frontier.shape[ 1 ])))
		grid[:,:BMOO.n_objectives] = grid[:,:BMOO.n_objectives] * ( BMOO.y_o_upp - BMOO.y_o_low ) + BMOO.y_o_low
		grid[:,BMOO.n_objectives:] = grid[:,BMOO.n_objectives:] * ( BMOO.y_c_upp - BMOO.y_c_low ) + BMOO.y_c_low
 
		frontier = self.apply_extended_domination_rule_to_frontier_mc(frontier)

		n_added = 0
                
                new_grid = np.array([])
		for point in grid:
			if not self.particle_is_dominated_original_space(point, frontier):
				if n_added == 0:
					new_grid = np.array([ point ])
				else:
					new_grid = np.concatenate((new_grid, np.array([ point ])))
				n_added += 1
		return new_grid	
