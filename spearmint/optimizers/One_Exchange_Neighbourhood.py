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


from abstract_optimizer import AbstractOptimizer
import scipy.optimize as spo
from spearmint.grids import sobol_grid
import numpy as np
import numpy.random   as npr

class One_Exchange_Neighbourhood(AbstractOptimizer):

    INTEGER_TRANSFORMATION = "Integer"
    CATEGORICAL_TRANSFORMATION = "Categorical"
    STARTING_VALUE = np.inf
    NO_OPTIMIZATION = "No categorical or integer variables exist in the problem, returning initial value."
    OPTIMIZATION_OK = "One Exchange Neighbourhood has finished successfully."
 
    TESTING_COUNTER = 0 ### TESTING.

    def __init__(self, has_gradients, options):
	self.has_gradients = False
	self.options = options
	self.no_cats = True
	self.no_ints = True
	self.no_reals = True
	self.discrete_dimensions = []
	self.real_dimensions = []
	self.fill_indexes()
	return

    def set_has_gradients(self, has_gradients):
	self.has_gradients = has_gradients

    def fill_indexes(self):
	discrete_dimensions = []
	variables = len(self.options["variables"])
	discrete_variables = 0
	total_number_of_variables = 0
	total_number_of_vars_transf = 0
	for transformation in self.options["transformations_for_opt"]:
		if One_Exchange_Neighbourhood.INTEGER_TRANSFORMATION in transformation:
			self.integer_transformations = transformation[One_Exchange_Neighbourhood.INTEGER_TRANSFORMATION]
			self.no_ints = False
			discrete_variables += len(transformation['Integer']['integer_dimensions'])
			total_number_of_variables += len(transformation['Integer']['integer_dimensions'])
			total_number_of_vars_transf += len(transformation['Integer']['integer_dimensions'])
			for dim in transformation['Integer']['integer_dimensions']:
				self.discrete_dimensions.append(int(dim))
		elif One_Exchange_Neighbourhood.CATEGORICAL_TRANSFORMATION in transformation:	
			self.categorical_transformations = transformation[One_Exchange_Neighbourhood.CATEGORICAL_TRANSFORMATION]
			self.no_cats = False
			dimensions = transformation['Categorical']['categorical_dimensions']
			discrete_variables += len(transformation['Categorical']['categorical_dimensions'])
			total_number_of_vars_transf += len(transformation['Categorical']['categorical_dimensions'])
			number_of_dim_dimensions = transformation['Categorical']['num_values']
			i = 0
			for dim in dimensions:
				number_dim_dim = number_of_dim_dimensions[i]
				cat_dim = np.linspace(dim,dim+number_dim_dim-1,number_dim_dim)
				for cd in cat_dim:
					self.discrete_dimensions.append(int(cd))
					total_number_of_variables+=1
				i+=1	
			
	total_number_of_variables += variables - total_number_of_vars_transf
	self.real_dimensions = set(np.linspace(0, total_number_of_variables-1, total_number_of_variables, dtype=int).tolist())
	self.discrete_dimensions = set(self.discrete_dimensions)
	self.real_dimensions = list(self.real_dimensions - self.discrete_dimensions)
	self.real_dimensions = sorted(self.real_dimensions)
	self.discrete_dimensions = sorted(list(self.discrete_dimensions))
	if discrete_variables != variables:
		self.no_reals = False

	if not self.no_ints:
		self.prepare_integer_dimension_information()
	return 

    #Returns the most feasible point from a given one in the feasible optimizer space.
    #It is just OK by calling the internal method denormalize point.
    def give_nearest_feasible_point(self, given_point):
	return self.denormalize_point(given_point)

    #Assumes a unit-hypercube boundary.
    def prepare_integer_dimension_information(self):
	
	#First, we have to compute intervals to move from a position to another position of the acq. fun.
	self.integer_transformations["intervals"] = []
	for value in self.integer_transformations['num_values']:
		self.integer_transformations["intervals"].append(np.linspace(0,1,value))
	
	'''
	Not necessary, but useful for being integrated into Spearmint core!
	We now assume that the user does this logic while configurating the config.json file,
	but as it is a mess, this code would do it automatically for integer valued variables.
	#As the problem may involve categorical variables, we have to transform the integer dimensions indexes.
	dimension_index = 0
	for int_dim in self.integer_transformations['integer_dimensions']:
		previous_cat_dims = [i for i in self.categorical_transformations["categorical_dimensions"] if i < int_dim]
		dimensions = np.linspace(0, int_dim-1, int_dim)
		dim_counter = 0
		cat_counter = 0
		for counter in range(dimensions.shape[0]):
			if previous_cat_dims[cat_counter] == dimensions[counter]:
				dim_counter += self.categorical_transformations["num_values"][counter]
				cat_counter += 1
			else:
				dim_counter+=1
		self.integer_transformations['integer_dimensions'][dimension_index] = dim_counter
		dimension_index += 1
	'''
	return

    #Optimize is a simple local search that minimises the value of acq_fun returning the location and value of the minimized function.
    #Include the real values with L-BGFS in the neighbours.

    def optimize(self, acq_fun, best_acq_location, af_bounds):

	if self.no_cats and self.no_ints:
		return spo.fmin_l_bfgs_b(acq_fun, best_acq_location, bounds=af_bounds, disp=0, approx_grad=not self.has_gradients)
	else:

                # If the real dims are not optimized first of this point, OEN lacks a lot of advantage w.r.t L-BFGS in the worst case
                # that the neighbours are not better than the original point since real variables are not going to be optimized.

                new_point = self.give_nearest_feasible_point(best_acq_location)
                new_point = self.optimize_real_dimensions(np.array([best_acq_location]), acq_fun, af_bounds)[0]
                new_af_value = acq_fun(new_point)[ 0 ]
		old_af_value = One_Exchange_Neighbourhood.STARTING_VALUE
		iterations = 0

		while new_af_value < old_af_value:

			old_af_value = new_af_value.copy()
			old_point = new_point.copy()
			neigh_points = self.get_neighbourhood(old_point)

			if not self.no_reals:
				neigh_points = self.optimize_real_dimensions(neigh_points, acq_fun, af_bounds)

			neigh_values = acq_fun(neigh_points)

                        # We check if there is a better neighbor

                        if np.min(neigh_values) < old_af_value:

                            to_sel = np.argmin(neigh_values)
                            new_af_value = neigh_values[ to_sel ]
                            new_piont = neigh_points[ to_sel, : ]

			iterations+=1

		return new_point, new_af_value, One_Exchange_Neighbourhood.OPTIMIZATION_OK + ". Iterations = " + str(iterations) + ".", iterations

    #Watch out and test this: Discrete variables must remain with the same value, they are in the frontier here, grad infinity?

    def optimize_real_dimensions(self, neigh_points, acq_fun, af_bounds):

	neighbourhood = np.zeros((len(neigh_points), neigh_points[0].shape[0]))

	i = 0

	for point in neigh_points:

                initial = neigh_points[ i, : ]

                def acq_fun_real(x):
                    candiate = initial.copy()
                    candiate[ self.real_dimensions ] = x
                    return acq_fun(candiate)

		new_point, solution, info = spo.fmin_l_bfgs_b(acq_fun_real, initial[ self.real_dimensions ], \
                        bounds = [ af_bounds[ k ] for k in self.real_dimensions ], disp = 0, approx_grad = not self.has_gradients)

                neighbourhood[ i, : ] = initial
		neighbourhood[ i, self.real_dimensions ] = new_point

		i+=1

	return neighbourhood

    def denormalize_point(self, x):
	#First the categorical variables.
	if not self.no_cats:
		for i in range(len(self.categorical_transformations["num_values"])):
			index_of_cat_dimension = self.categorical_transformations["categorical_dimensions"][i]
			num_dims_of_cat_dimension = self.categorical_transformations["num_values"][i]

			lower_bound = index_of_cat_dimension
			upper_bound = index_of_cat_dimension+num_dims_of_cat_dimension

			categorical_variable = x[lower_bound:upper_bound]
			normalized_categorical_variable = np.zeros(categorical_variable.shape[0]) + 1e-02
			normalized_categorical_variable[np.argmax(categorical_variable)] = 1 - 1e-02
			x[lower_bound:upper_bound] = normalized_categorical_variable
			
        #Then, the integer variables.
	if not self.no_ints:
		for i in range(len(self.integer_transformations["num_values"])):
			index_of_int_dimension = self.integer_transformations["integer_dimensions"][i]
	                intervals = self.integer_transformations["intervals"][i]

			int_variable = x[index_of_int_dimension]
			normalized_int_variable = intervals[np.argmin(np.absolute(int_variable - intervals))]
			x[index_of_int_dimension] = normalized_int_variable
	return x

    # Gets the neighbours of x

    def get_neighbourhood(self, x):

	neighbourhood = None

	if not self.no_ints:
		for i in range(len(self.integer_transformations["num_values"])):

			index_of_int_dimension = self.integer_transformations["integer_dimensions"][i]
			intervals = self.integer_transformations["intervals"][i]
			int_variable = x[ index_of_int_dimension ]
			index_values = []
			current_index_of_value = np.where(intervals == int_variable)[0][0]

			if current_index_of_value == 0:
				index_values.append(1)
			elif current_index_of_value == intervals.shape[0]-1:
				index_values.append(current_index_of_value-1)
			else:
				index_values.append(current_index_of_value-1)
				index_values.append(current_index_of_value+1)

			neighs_of_int_dim = intervals[ index_values ]

                        for j in range(len(neighs_of_int_dim)):

        		    y = x.copy()
			    y[ index_of_int_dimension ] = neighs_of_int_dim[ j ]
                            
                            if neighbourhood is None:
     			        neighbourhood = y
                            else:
     			        neighbourhood = np.vstack((neighbourhood, y))

	if not self.no_cats:
		for i in range(len(self.categorical_transformations["num_values"])):	

			index_of_cat_dimension = self.categorical_transformations["categorical_dimensions"][ i ]
                	num_dims_of_cat_dimension = self.categorical_transformations["num_values"][ i ]

			lower_bound = index_of_cat_dimension
        	        upper_bound = index_of_cat_dimension + num_dims_of_cat_dimension

                	categorical_variable = x[lower_bound : upper_bound]		

			for j in range(categorical_variable.shape[0]):

				if categorical_variable[ j ] != (1.0-1e-02):
					cat_var_neigh = np.zeros(categorical_variable.shape[0]) + 1e-02
					cat_var_neigh[ j ] = 1.0 - 1e-02
					neighbour = x.copy()
					neighbour[ lower_bound : upper_bound ] = cat_var_neigh

                                        if neighbourhood is None:
			                    neighbourhood = neighbour
                                        else:
			                    neighbourhood = np.vstack((neighbourhood, neighbour))
	return neighbourhood

