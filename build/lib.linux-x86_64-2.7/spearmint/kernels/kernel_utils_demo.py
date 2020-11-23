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


import numpy as np
import weave
from scipy.spatial.distance import cdist


DEFAULT_CONDITIONED_VALUE = -1
def dist2(ls, x1, x2=None, conditioning=None):
    # Assumes NxD and MxD matrices.
    # Compute the squared distance matrix, given length scales.
    
    if x2 is None:
        # Find distance with self for x1.

        # Rescale.
        xx1 = x1 / ls        
        xx2 = xx1

    else:
        # Rescale.
        xx1 = x1 / ls
        xx2 = x2 / ls
    
    if conditioning is None:
    	return cdist(xx1,xx2,'sqeuclidean')
    else: #Conditional treatment processing.
	return conditional_dist2(xx1, xx2, conditioning)

#Naive version of the case conditional transformation.
#Conditional transformation performed in every point of x1 and x2 before computing cdist.
def conditional_dist2(x1, x2, conditioning):
	number_instances_x1 = x1.shape[0]
	number_instances_x2 = x2.shape[0]
	instance_dimension = x1.shape[1]
	covariance_matrix = np.zeros((number_instances_x1, number_instances_x2))

	#Apply conditional transformation on the values of each pair.
	for conditioning_variable in conditioning:
		dependant_variables = conditioning[conditioning_variable]
		values_number_of_conditioning_variable = len(dependant_variables)
		
		#This for happens due to the fact that categorical variables are encoded in one dimension per value.
		#And we are interested in every dimension of the categorical variable to apply the transformation.
		#Apply every conditioning to the points.
		for index_x1 in range(x1.shape[0]):
			for index_x2 in range(x2.shape[0]):
				point_a = x1[index_x1]
				point_b = x2[index_x2]
				categorical_variable_a_values = point_a[conditioning_variable:conditioning_variable+values_number_of_conditioning_variable]
				categorical_variable_b_values = point_b[conditioning_variable:conditioning_variable+values_number_of_conditioning_variable]

				#Case A, equal values, non conditioned variables of this space converted to default.
				#c=a, v_a=x, v_b=y
				#c=a, v_a=a, v_b=b
				#Transformed in:
				#c=a, v_a=x, v_b=DEF
				#c=a, v_a=a, v_b=DEF
				if all(categorical_variable_a_values == categorical_variable_b_values):
					#Index of the categorical variable.
					cat_var_value = np.where(categorical_variable_a_values != 0)[0][0]
					index_of_non_conditioned_variables = [var for index_var, var in enumerate(dependant_variables) if index_var != cat_var_value]
					#Flat list.
					index_of_non_conditioned_variables = [item for sublist in index_of_non_conditioned_variables for item in sublist]
					point_a[index_of_non_conditioned_variables] = DEFAULT_CONDITIONED_VALUE
					point_b[index_of_non_conditioned_variables] = DEFAULT_CONDITIONED_VALUE
			
				#Case B: Different values for categorical variable, all the conditioned by all values are switched to default.
				#c=a, v_a=x, v_b=y
				#c=b, v_a=a, v_b=b
				#c=a, v_a=DEF, v_b=DEF
				#c=b, v_a=DEF, v_b=DEF
				else:
					default_variables_indexes = [item for sublist in dependant_variables for item in sublist] #Flat list.
					point_a[default_variables_indexes] = DEFAULT_CONDITIONED_VALUE
                                        point_b[default_variables_indexes] = DEFAULT_CONDITIONED_VALUE

				#Insert the euclidean distance between the transformed points.
				covariance_matrix[index_x1][index_x2] = cdist(np.array([point_a]), np.array([point_b]), 'sqeuclidean')

	#Compute the distances given by the calling kernel (this is a first computation on stationary kernels,
	#other transformations performed by the kernel are computed in the caller kernel class),
	#between those points.
	import pdb; pdb.set_trace();
	return covariance_matrix

# The gradient of the squared distance with respect to x1

def grad_dist2(ls, x1, x2=None):
    if x2 is None:
        x2 = x1
        
    # Rescale.
    x1 = x1 / ls
    x2 = x2 / ls
    
    N = x1.shape[0]
    M = x2.shape[0]
    D = x1.shape[1]
    gX = np.zeros((x1.shape[0],x2.shape[0],x1.shape[1]))

    code = \
    """
    for (int i=0; i<N; i++)
      for (int j=0; j<M; j++)
        for (int d=0; d<D; d++)
          gX(i,j,d) = (2/ls(d))*(x1(i,d) - x2(j,d));
    """
    weave.inline(code, ['x1','x2','gX','ls','M','N','D'], \
                       type_converters=weave.converters.blitz, \
                       compiler='gcc')

    # The C code weave above is 10x faster than this:
    #for i in xrange(0,x1.shape[0]):
    #    gX[i,:,:] = 2*(x1[i,:] - x2[:,:])*(1/ls)

    return gX

def dist_Mahalanobis(U, x1, x2=None):
    W = np.dot(U,U.T)

# This function is useful if the data can appear in multiple forms
# but the code is being changed so that the data will always be an array.

# def extract_data(func):
#     """
#     Decorator function.

#     If the input arguments are dicts instead of ndarrays then this extracts
#     the ndarrays at the key 'inputs'. It makes the rest of the kernel cleaner
#     since they don't have to do any bookkeeping.
#     """
#     def inner(cls_instance, *args):
#         new_args = []
#         for data in args:
#             if isinstance(data, dict):
#                 if not data.has_key('inputs'):
#                     raise Exception('Data dict must have key "inputs".')
#                 new_args.append(data['inputs'])
#             elif isinstance(data, np.ndarray):
#                 new_args.append(data)
#             else:
#                 raise Exception('Data of type %s not supported in kernels.' % data.__class__)

#         return func(cls_instance, *new_args)
#     return inner


#Case transformation.
#Conditioning parameter format. It is a dictionary.
# keys - Indexes of conditional variable, that affect others. Only given the first, next are extracting by using the size of the value vector.
# values - List of lists. Each list contains the variables that need to be put to default is index_value of conditional variable is chosen.
# That is to say, the variables that do not exist for the value of the conditional variable.
#Conditioning {"conditional_var_1" : [[var_conditioned_value_a]]}

#How to implement it?:
# Naive version: Develop two for loops, transform the points, and then call cdist with the points transformed. With the results of iterative cdist, build cross cov matrix. It will be VERY slow, but it would test that the transformation works from a mathematical point of view and BO could run.
# Good version: Modify cdist, when real distance is computed between 2 points, transformation needs to be done. It should not be difficult. Only necessary parameter is conditioning, with the independencies between variables.
#Example: Index 0 make 2 DEF. Index 1 make 3 DEF.
if __name__ == '__main__':
	cond = {0: [[2],[3]]}
        x1 = np.array([[ 0.          ,1.          ,0.          ,0.          ,0.4         ,0.,
   0.31652832],
 [ 1.,          0.,          0.,          0.,          0.8,         0.5,
   0.54408264],
 [ 0.,          1.,          0.,          1.,          1.,          1.,
   0.55135673],
 [ 0.,          1.,          0.,          1.,          0.6,         1.,
   0.23842019],
 [ 1.,          0.,          0.,          0.,          0.,          1.,
   0.02063379]])
        x2 = np.array([[ 0.          ,1.          ,0.          ,0.          ,0.4         ,0.,
   0.31652832],
[ 1.,          0.,          0.2,          0.,          0.,          1.,
   0.02063379], [ 0.,          1.,          0.,          0.,          0.,          1.,
   0.02063379]])
	ls = 1.5
	print(dist2(ls, x1, x2, conditioning=cond))
