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
import scipy.stats    as sps
import scipy.linalg   as spla
import numpy.linalg   as npla
import scipy.optimize as spo
import copy
import traceback
import warnings
import sys

from collections import defaultdict
from spearmint.grids import sobol_grid
from spearmint.acquisition_functions.abstract_acquisition_function import AbstractAcquisitionFunction
from spearmint.acquisition_functions.PPESMOC_gradients import compute_acq_fun_wrt_test_points
from spearmint.utils.numerics import logcdf_robust
from spearmint.models.gp import GP
from spearmint.utils.moop            import MOOP_basis_functions
from spearmint.utils.moop            import _cull_algorithm
from spearmint.utils.moop            import _compute_pareto_front_and_set_summary_x_space
from scipy.spatial.distance import cdist
from spearmint.kernels import kernel_utils


from spearmint.models.abstract_model import function_over_hypers
from autograd import jacobian
from autograd import grad
import logging

import traceback
import warnings
import sys

try:
    import nlopt
except:
    nlopt_imported = False
else:
    nlopt_imported = True
# see http://ab-initio.mit.edu/wiki/index.php/NLopt_Python_Reference

NUM_RANDOM_FEATURES = 1000
PARETO_SET_SIZE = 10
NSGA2_POP = 100
NSGA2_EPOCHS = 100
GRID_SIZE = 1000
USE_GRID_ONLY = False

PESM_OPTION_DEFAULTS  = {
    'pesm_num_random_features'      : 1000,
    'pesm_pareto_set_size'      : 10,
    'pesm_grid_size'      : 1000,
    'pesm_not_constrain_predictions' : False,
    'pesm_samples_per_hyper' : 1,
    'pesm_use_grid_only_to_solve_problem' : False,
    'pesm_nsga2_pop' : 100,
    'pesm_nsga2_epochs' : 100
    }


def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    log = file if hasattr(file,'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))

def write_log(fname, froute, ret_variables):
    log = open(froute, 'a')
    log.write('=========================\n')
    log.write(fname + '\n')
    log.write('=========================\n')
    for key, value in ret_variables.items():
        log.write('%s == %s \n' %(key, value))
    log.write('=========================\n\n')
    log.close()

"""
FOR GP MODELS ONLY
"""
# get samples of the solution to the problem

def sample_solution(num_dims, objectives_gp, constraint_gps):

        # We find the inputs to be added to the grid (we avoid adding repeated inputs)

        X = objectives_gp[ 0 ].inputs 

        for i in range(len(objectives_gp) - 1):
            X_to_add = objectives_gp[ i + 1 ].inputs

            for j in range(X_to_add.shape[ 0 ]):
                if np.min(cdist(X, X_to_add[ j : (j + 1), : ])) > 1e-8:
                    X = np.vstack((X , X_to_add[ j : (j + 1), : ]))

        for i in range(len(constraint_gps)):
            X_to_add = constraint_gps[ i ].inputs

            for j in range(X_to_add.shape[ 0 ]):
                if np.min(cdist(X, X_to_add[ j : (j + 1), : ])) > 1e-8:
                    X = np.vstack((X , X_to_add[ j : (j + 1), : ]))

        inputs = X

	MAX_ATTEMPTS = 100
	num_attempts = 0

	while num_attempts < MAX_ATTEMPTS:

		# 1. The procedure is: sample all f on the grid "cand" (or use a smaller grid???)
		# 2. Look for the pareto set 

		gp_samples = dict()

		gp_samples['objectives'] = [ sample_gp_with_random_features(objective_gp, NUM_RANDOM_FEATURES) for \
			objective_gp in objectives_gp ]
		gp_samples['constraints'] = [sample_gp_with_random_features(constraint_gp, NUM_RANDOM_FEATURES) for \
			constraint_gp in constraint_gps]

		pareto_set = global_optimization_of_GP_approximation(gp_samples, num_dims, inputs)

		if pareto_set is not None: # success
			logging.debug('successfully sampled X* in %d attempt(s)' % (num_attempts+1))
		
			return pareto_set

		num_attempts += 1

	return None

#	raise npla.linalg.LinAlgError("Unable to sample valid pareto set that satisfies the constrains !")

# Compute log of the normal CDF of x in a robust way
# Based on the fact that log(cdf(x)) = log(1-cdf(-x))
# and log(1-z) ~ -z when z is small, so  this is approximately
# -cdf(-x), which is just the same as -sf(x) in scipy
def logcdf_robust(x):
    
    if isinstance(x, np.ndarray):
        ret = sps.norm.logcdf(x)
        ret[x > 5] = -sps.norm.sf(x[x > 5])
    elif x > 5:
        ret = -sps.norm.sf(x)
    else:
        ret = sps.norm.logcdf(x)

    return ret

# Compute log(exp(a)+exp(b)) in a robust way. 
def logSumExp_scalar(a, b):

    if a > b:
        # compute log(exp(a)+exp(b)) 
        # this is just the log-sum-exp trick but with only 2 terms in the sum
        # we chooser to factor out the largest one
        # log(exp(a)+exp(b)) = log( exp(a) [1 + exp(b-a) ] )
        # = a + log(1 + exp(b-a))
        return a + log_1_plus_exp_x(b-a)
    else:
        return b + log_1_plus_exp_x(a-b)

def logSumExp(a,b):
    if (not isinstance(a, np.ndarray) or a.size==1) and (not isinstance(b, np.ndarray) or b.size==1):
        return logSumExp_scalar(a,b)

    result = np.zeros(a.shape)
    result[a>b] =  a[a>b]  + log_1_plus_exp_x(b[a>b] -a[a>b])
    result[a<=b] = b[a<=b] + log_1_plus_exp_x(a[a<=b]-b[a<=b])
    return result

def log_1_plus_exp_x_scalar(x):
# Compute log(1+exp(x)) in a robust way
    if x < np.log(1e-6):
        # if exp(x) is very small, i.e. less than 1e-6, then we can apply the taylor expansion:
        # log(1+x) approx equals x when x is small
        return np.exp(x)
    elif x > np.log(100):
        # if exp(x) is very large, i.e. greater than 100, then we say the 1 is negligible comared to it
        # so we just return log(exp(x))=x
        return x
    else:
        return np.log(1.0+np.exp(x))

def log_1_plus_exp_x(x):
    if not isinstance(x, np.ndarray) or x.size==1:
        return log_1_plus_exp_x_scalar(x)

    result = np.log(1.0+np.exp(x)) # case 3
    result[x < np.log(1e-6)] = np.exp(x[x < np.log(1e-6)])
    result[x > np.log(100) ] = x [x > np.log(100) ]
    return result

# Compute log(1-exp(x)) in a robust way, when exp(x) is between 0 and 1 
# well, exp(x) is always bigger than 0
# but it cannot be above 1 because then we have log of a negative number
def log_1_minus_exp_x_scalar(x):
    if x < np.log(1e-6): 
        # if exp(x) is very small, i.e. less than 1e-6, then we can apply the taylor expansion:
        # log(1-x) approx equals -x when x is small
        return -np.exp(x)
    elif x > -1e-6:
        # if x > -1e-6, i.e. exp(x) > exp(-1e-6), then we do the Taylor expansion of exp(x)=1+x+...
        # then the argument of the log, 1- exp(x), becomes, approximately, 1-(1+x) = -x
        # so we are left with log(-x)
        return np.log(-x)
    else:
        return np.log(1.0-np.exp(x))

def log_1_minus_exp_x(x):
    if not isinstance(x, np.ndarray) or x.size==1:
        return log_1_minus_exp_x_scalar(x)

    assert np.all(x <= 0)

    case1 = x < np.log(1e-6) # -13.8
    case2 = x > -1e-6
    case3 = np.logical_and(x >= np.log(1e-6), x <= -1e-6)
    assert np.all(case1+case2+case3 == 1)

    result = np.zeros(x.shape)
    result[case1] = -np.exp(x[case1])
    with np.errstate(divide='ignore'): # if x is exactly 0, give -inf without complaining
        result[case2] = np.log(-x[case2])
    result[case3] = np.log(1.0-np.exp(x[case3]))

    return result

def chol2inv(chol):
    return spla.cho_solve((chol, False), np.eye(chol.shape[0]))

def matrixInverse(M):
    return chol2inv(spla.cholesky(M, lower=False))

############################################################################################
# build_set_of_points_that_conditions_GPs: Function that builds the set of points that is 
# going to be used to condition the GP and where factors are approximated.
#
# PPESMOC: Now the set of points will also contain the candidate (test) ones.
#
# INPUT:
# obj_models: GPs with the objectives.
# con_models: GPs with the constraints.
#
# OUTPUT:
# X: Set of points.
# n_obs, n_pset, n_total: Counters of the points.
############################################################################################
def build_set_of_points_that_conditions_GPs(obj_models, con_models, pareto_set, Xtest):

    #We first include the observations.
    X = np.array([np.array([])])
    for t in obj_models:
        Xtask = obj_models[ t ].observed_inputs
	X = np.array([Xtask[ 0, ]])
        for i in range(1, Xtask.shape[ 0 ]):
            if np.min(cdist(Xtask[ i : (i + 1), : ], X)) > 1e-8:
	        X = np.vstack((X, Xtask[ i, ]))

    for t in con_models:
        Xtask = con_models[ t ].observed_inputs
        for i in range(Xtask.shape[ 0 ]):
            if np.min(cdist(Xtask[ i : (i + 1), : ], X)) > 1e-8:
                X = np.vstack((X, Xtask[ i, ]))

    #Then, we include the Pareto Set points.
    for i in range(pareto_set.shape[ 0 ]):
        #if np.min(cdist(pareto_set[ i : (i + 1), : ], X)) > 1e-8:
        X = np.vstack((X, pareto_set[ i, ]))

    n_obs = X.shape[ 0 ] - pareto_set.shape[ 0 ]

    #Finally, we include the candidate points, without comparing with the previous points.
    n_test = Xtest.shape[ 0 ]
    for i in range(Xtest.shape[ 0 ]):
        X = np.vstack((X, Xtest[ i, ]))

    """ OJO: Esto esta haciendo petar a PPESMOC con mas de 100 evaluaciones, consultar a Daniel si simplemente esta
        correcto pasar de este codigo. Yo lo encuentro de poca utilidad y lo descartaría simplemente. De no hacerlo 
        hay que modificar mas el codigo de forma fea y a mi particularmente no me gusta.
    if n_obs >= 100:
        X_sel = X[ np.random.choice(n_obs, 100, False), : ]
        X = np.vstack((pareto_set, X_sel))
        n_obs = 100
    """

    n_total = X.shape[ 0 ]
    n_pset = pareto_set.shape[ 0 ]

    return X, n_obs, n_pset, n_test, n_total

############################################################################################
# build_unconditioned_predictive_distributions: It returns the mean and variance of the 
# predictive distributions over the points computed in self.build_set_of_points_that_conditions_GPs
# along with other information related to the distribution.
#
# PPESMOC: The inverse can now, as it is a block matrix, be calculated through Cholesky yet? Daniel: YES.
# Why are cholKstarstar matrices computed?
#
# INPUT:
#
# all_tasks: GPs with the objectives.
# all_constraints: GPs with the constraints.
# X: Set of points to extract the predictive distribution using the GPs.
#
# OUTPUT:
#
# mPred: Predictive mean of the all_tasks GPs.
# Vpred: Predictive variance of the all_tasks_GPs.
# cholVpred: Cholesky decomposition of the covariance matrix of all_tasks GPs.
# VpredInv: Inverse of the covariance matrix of all_tasks GPs computed through Cholesky.
# cholKstarstar: Cholesky decomposition of the covariance matrix computed by the noiseless_kernel.
# mPred_cons: Predictive mean of the all_constraints GPs.
# Vpred_cons: Predictive variance of the all_constraints GPs.
# cholVpred_cons: Cholesky decomposition of the covariance matrix of all_constraints GPs.
# VpredInv_cons: Inverse of the covariance matrix of all_constraints GPs computed through Cholesky.
# cholKstarstar_cons: Cholesky decomposition of the covariance matrix computed by the noiseless_kernel.
############################################################################################
def build_unconditioned_predictive_distributions(all_tasks, all_constraints, X):
    mPred         = dict()
    Vpred         = dict()
    cholVpred     = dict()
    VpredInv      = dict()
    cholKstarstar = dict()

    mPred_cons         = dict()
    Vpred_cons         = dict()
    cholVpred_cons     = dict()
    VpredInv_cons      = dict()
    cholKstarstar_cons = dict()

    for t in all_tasks:
        mPred[t], Vpred[t] = all_tasks[t].predict(X, full_cov=True)
        cholVpred[t]       = spla.cholesky(Vpred[t])
        VpredInv[t]        = chol2inv(cholVpred[t])
        # Perform a redundant computation of this thing because predict() doesn't return it...
        cholKstarstar[t]   = spla.cholesky(all_tasks[t].noiseless_kernel.cov(X))

    for t in all_constraints:
        mPred_cons[t], Vpred_cons[t] = all_constraints[t].predict(X, full_cov=True)
        cholVpred_cons[t]       = spla.cholesky(Vpred_cons[t])
        VpredInv_cons[t]        = chol2inv(cholVpred_cons[t])
        # Perform a redundant computation of this thing because predict() doesn't return it...
        cholKstarstar_cons[t]   = spla.cholesky(all_constraints[t].noiseless_kernel.cov(X))

    return mPred, Vpred, cholVpred, VpredInv, cholKstarstar, mPred_cons, Vpred_cons, cholVpred_cons, VpredInv_cons, cholKstarstar_cons

#This function gets the jitter of black box function.
def get_jitter(obj_models, con_models):
    jitter = dict()
    for task in obj_models:
        jitter[ task ] = obj_models[ task ].jitter_value()
    for cons in con_models:
        jitter[ cons ] = con_models[ cons ].jitter_value()
    return jitter

############################################################################################
# create_data_structure_for_EP_computations_and_posterior_approximation: Function that creates
# the data structure that is going to be used for the computations of EP that condition the
# predictive distribution of the GPs created before.
#
# PPESMOC: This data structure will now have to contain the candidate points and the candidate factors.
#
# INPUT:
# obj_models: GPs with the objectives.
# con_models: GPs with the constraints.
# n_obs, n_pset, n_total: Counters of the set of points where the EP factors must approximate the real factors.
# q, c: Number of objectives and constraints.
# mPred, Vpred, VpredInv, cholKstarstar, mPred_cons, Vpred_cons, VpredInv_cons, cholKstarstar_cons: Predictive distribution information.
# jitter: Jitter added to the GP models.
# X: Set of points which objective function are predicted (modelled) by the GP and where the acq. fun. needs the EP factors to suggest a new point.
#
# OUTPUT:
# a: Data structure. 
############################################################################################
def create_data_structure_for_EP_computations_and_posterior_approximation(obj_models, con_models, n_obs, n_pset, n_test, n_total, q, c, \
			mPred, Vpred, VpredInv, cholKstarstar, mPred_cons, Vpred_cons, VpredInv_cons, cholKstarstar_cons, jitter, X):
    a = {
	#Models.
        'objs'     : obj_models, #Posterior GP of the objectives.
        'cons'     : con_models, #Posterior GP of the constraints.
	#EP Factors.
        'ahfhat'   : np.zeros((n_obs, n_pset, q, 2, 2)), #EP Factors of Observations. Covariance matrix of the omega factor, dominance of two points.
        'bhfhat'   : np.zeros((n_obs, n_pset, q, 2)), #EP factors of Observations. Mean vector of the omega factor, dominance of two points.
        'chfhat'   : np.zeros((n_pset, n_pset, q, 2, 2)), #EP factor of Pareto Set. Covariance matrix of the omega factor, dominance of two points.
        'dhfhat'   : np.zeros((n_pset, n_pset, q, 2)), #EP factor of Pareto Set. Mean vector of the omega factor, dominance of two points.
        'ehfhat'   : np.zeros((n_pset, c)), #EP factor of Pareto Set. Variance of the point, feasibility.
        'fhfhat'   : np.zeros((n_pset, c)), #EP factor of Pareto Set. Mean of the point, feasibility.
	##################################################################################################################################
	#New EP test factors, see modification of predictEP. 
        'ghfhat'   : np.zeros((n_test, n_pset, q, 2, 2)), #EP factor of test points. Covariance matrix of the omega factor.
        'hhfhat'   : np.zeros((n_test, n_pset, q, 2)), #EP factor of Pareto Set. Mean vector of the omega factor, dominance of two points.
        'g_c_hfhat'   : np.zeros((n_test, n_pset, c)), #EP factor of Test points. Variance of the phi factor in omega. Feasibility.
        'h_c_hfhat'   : np.zeros((n_test, n_pset, c)), #EP factor of Test points. Mean of the phi factor in omega. Feasibility.
	##################################################################################################################################
        'a_c_hfhat'   : np.zeros((n_obs, n_pset, c)), #EP Factors of Observations. Variance of the phi factor in omega. Feasibility.
        'b_c_hfhat'   : np.zeros((n_obs, n_pset, c)), #EP Factors of Observations. Mean of the phi factor in omega. Feasibility
        'c_c_hfhat'   : np.zeros((n_pset, n_pset, c)), #EP Factors of Pareto Set. Variance of the phi factor in omega. Feasibility
        'd_c_hfhat'   : np.zeros((n_pset, n_pset, c)), #EP Factors of Pareto Set. Mean of the phi factor in omega. Feasibility.
	##################################################################################################################################
	#Objective and constraint conditional predictive distributions.
        'm'        : defaultdict(lambda: np.zeros(n_total)),  #Mean vector of the conditional predictive (in n_total points) distribution.
        'm_nat'    : defaultdict(lambda: np.zeros(n_total)),  #Mean vector in natural form of the conditional predictive distribution.
        'V'        : defaultdict(lambda: np.zeros((n_total, n_total))), #Covariance matrix of the conditional predictive distribution.
        'Vinv'     : defaultdict(lambda: np.zeros((n_total, n_total))), #Inverse of the Covariance matrix of the CPD.
        'm_cons'        : defaultdict(lambda: np.zeros(n_total)),  #Mean vector of the conditional predictive (in n_total points) distribution.
        'm_nat_cons'    : defaultdict(lambda: np.zeros(n_total)),  #Mean vector in natural form of the conditional predictive distribution.
        'V_cons'        : defaultdict(lambda: np.zeros((n_total, n_total))), #Covariance matrix of the CPD.
        'Vinv_cons'     : defaultdict(lambda: np.zeros((n_total, n_total))), #Inverse of the Covariance matrix of the CPD.
	#################################################################################################################################
	# PredictEP data structures. I think that they are not useful.
	#'m_final'        : defaultdict(lambda: np.zeros(n_total + n_test)),  #Mean vector of the CPD.
        #'V_final'        : defaultdict(lambda: np.zeros((n_total + n_test, n_total + n_test))), #Covariance matrix of the CPD.
        #'m_cons_final'        : defaultdict(lambda: np.zeros(n_total + n_test)),  #Mean vector of the CPD.
        #'V_cons_final'        : defaultdict(lambda: np.zeros((n_total + n_test, n_total + n_test))), #Covariance matrix of the CPD.
	#################################################################################################################################
	'cholV'    : dict(), #Dictionaries that will hold the Cholesky decomposition of the covariance matrix of the CPD.
        'cholV_cons'    : dict(), #Dictionaries that will hold the Cholesky decomposition of the covariance matrix of the CPD.
	#Counters.
        'n_obs'    : n_obs,
        'n_total'  : n_total, #n_pset+n_obs+n_test.
        'n_pset'   : n_pset,
        'n_test'   : n_test,
        'q'        : q, #Number of objectives.
        'c'        : c, #Number of constraints.
	#Objective and Constrained Predictive distribution. (Unconditioned).
	'mPred'    : mPred, #Mean vector of the predictive (in n_total points) distribution.
        'Vpred'    : Vpred, #Covariance matrix of the predictive distribution.
        'VpredInv' : VpredInv, #Inverse of the Covariance matrix of the predictive distribution.
        'cholKstarstar' : cholKstarstar, #Cholesky decomposition of the covariance matrix computed by noiseless kernel.
        'mPred_cons'    : mPred_cons, #Mean vector of the predictive (in n_total points) distribution.
        'Vpred_cons'    : Vpred_cons, #Covariance matrix of the predictive distribution.
        'VpredInv_cons' : VpredInv_cons, #Inverse of the Covariance matrix of the predictive distribution.
        'cholKstarstar_cons' : cholKstarstar_cons, #Cholesky decomposition of the covariance matrix computed by noiseless kernel.
	#Other data.
        'jitter'   : jitter, #Jitter for the covariance matrices.
        'X'        : X, #Observations+Pareto Set points+Test points.
    }

    return a

############################################################################################
# ep: Function that computes the EP factors non dependant on the candidate/s and other useful 
# information about the covariance matrices used in the predict_EP methods. There are factors
# for every point of the set computed in self.build_set_of_points_that_conditions_GPs.
# The EP factors are Gaussian Approximations of the non-Gaussian factors that the PPESMOC expression
# imposes to the points computed in self.build_set_of_points_that_conditions_GPs.
#
# INPUT
# obj_models: GPs with the objectives.
# pareto_set: Pareto set of points. (A single Pareto set)
# minimize: If FALSE, it maximizes.
# con_models: GPs with the constraints.
# input_space: Input space of the problem to solve.
#
# OUTPUT:
# a: Dictionary with int(self.options['pesm_samples_per_hyper']) EP approximations 
# of the factors that do not depend on the candidate and other useful information.
############################################################################################
def ep(obj_models, pareto_set, minimize=True, con_models=[], input_space=None, Xtest=None):

    all_tasks = obj_models.copy()
    all_constraints= con_models.copy()

    # The old order of this set is pareto-obs. The new order must be obs-pareto-candidate.
    # I think that this may be done in predictEP to preserve how it has been implemented.
    #X, n_obs, n_pset, n_test, n_total = build_set_of_points_that_conditions_GPs(obj_models, con_models, all_tasks, all_constraints, pareto_set, Xtest)

    X, n_obs, n_pset, n_total = build_set_of_points_that_conditions_GPs(obj_models, con_models, pareto_set)
    q = len(all_tasks)
    c = len(all_constraints)

    #Computation of predictive unconditional distributions of the GPs.
    mPred, Vpred, cholVpred, VpredInv, cholKstarstar, mPred_cons, Vpred_cons, cholVpred_cons, VpredInv_cons, cholKstarstar_cons = \
		build_unconditioned_predictive_distributions(all_tasks, all_constraints, X)		
    jitter = get_jitter(obj_models, con_models)
        
    # We create the posterior approximation
    # Some modifications are done in the structure to add the test points and factors.

    a = create_data_structure_for_EP_computations_and_posterior_approximation(obj_models, con_models, n_obs, n_pset, n_total, n_test, q, c, \
                        mPred, Vpred, VpredInv, cholKstarstar, mPred_cons, Vpred_cons, VpredInv_cons, cholKstarstar_cons, jitter, X, Xtest)
    
    # We condition the PD to the PESMOC conditions.
    a = perform_EP_algorithm(a, minimize, all_tasks, all_constraints)
    return a

##########################################################################################################################
# perform_EP_algorithm: It performs the EP approximations to the non Gaussian Factors of the Predictive Distribution.
#
# INPUT:
# a: Data structure with CPDs, PDs and EP factors. Unconditioned.
# minimize: If FALSE, it maximizes.
# 
# OUTPUT:
# a: Data structure with CPDs, PDs and EP factors. Conditioned.
##########################################################################################################################
def perform_EP_algorithm(a, minimize, all_tasks, all_constraints):
    convergence = False
    damping     = 0.5
    iteration   = 1
    a = updateMarginals(copy.deepcopy(a))
    aOld = copy.deepcopy(a)
    aOriginal = copy.deepcopy(a)
    while not convergence:
        aNew, a = compute_updates_and_reduce_damping_if_fail(a, damping, minimize, aOld)
        aOld = copy.deepcopy(a)
        a = copy.deepcopy(aNew)
        convergence = compute_convergence_criterion(a, convergence, all_tasks, iteration, aOld, damping, all_constraints)
        damping   *= 0.99
        iteration += 1
    return a

##########################################################################################################################
# compute_updates_marginals_and_damping_if_fail: This method computes the EP factors, updates the marginals and apply a
# reduction of damping if the updates fail.
#
# INPUT:
# a: Data structure with CPDs, PDs and EP factors.
# damping: Current porcentage of applied damping.
# minimize: If FALSE, it maximizes.
#
# OUTPUT: 
# a: Old Data structure with CPDs, PDs and EP factors without the update.
# aNew: New Data structure with CPDs, PDs and EP factors with the update.
##########################################################################################################################
def compute_updates_and_reduce_damping_if_fail(a, damping, minimize ,aOld):
	update_correct = False
        damping_inner = damping
        fail = False
        second_update = False
        while update_correct == False:
            error = False
            try:
            	aNew = updateFactors_fast_daniel_no_constraints_pareto_set_robust(copy.deepcopy(a), damping_inner, minimize = minimize,
                	 no_negative_variances_nor_nands = False)
            	aNew = updateMarginals(copy.deepcopy(aNew))
            except npla.linalg.LinAlgError as e:
                error = True
            if error == False:
                if fail == True and second_update == False:
                        a = aNew.copy()
                        second_update = True
                else:
                        update_correct = True
            else:
                a = copy.deepcopy(aOld)
                damping_inner = damping_inner * 0.5
                fail = True
                second_update = False
                print 'Reducing damping factor to guarantee EP update! Damping: %f' % (damping_inner)

	return aNew, a

##########################################################################################################################
# compute_convergence_criterion: This method computes a convergence criterion based on how much does the means and variances 
# of all tasks change after being updated by EP.
#
# PPESMOC: Duda, no verificamos en el criterio de convergencia si las restricciones se refinan... Me parece raro, podrian no refinarse
# para cuando los objetivos ya esten refinados.
#
# INPUT:
# a: Data structure with CPDs, PDs and EP factors.
# convergence: Boolean flag with the state of convergence.
# all_tasks: Dictionary with the objectives.
# iteration: Iteration of the EP algorithm.
#
# OUTPUT:
#
# convergence: TRUE, if the convergence criterion has been satisfied.
##########################################################################################################################
def compute_convergence_criterion(a, convergence, all_tasks, iteration, aOld, damping, all_constraints):
	change = 0.0
        for t in all_tasks:
            change = max(change, np.max(np.abs(a['m'][t] - aOld['m'][t])))
            change = max(change, np.max(np.abs(a['V'][t] - aOld['V'][t])))

	for c in all_constraints:
            change = max(change, np.max(np.abs(a['m_cons'][c] - aOld['m_cons'][c])))
            change = max(change, np.max(np.abs(a['V_cons'][c] - aOld['V_cons'][c])))


        print '%d:\t change=%f \t damping: %f' % (iteration, change, damping)

        if change < 1e-3 and iteration > 2:
            convergence = True
	return convergence

#This method computes the cholesky decomposition of the CPDs.
# DHL: I have changed this since it seems it is only required outside EP
def compute_choleskys(a, all_tasks, all_constraints):
    for obj in all_tasks:
        a['cholV'][ obj ] = spla.cholesky(a['V'][obj], lower=False)
    for cons in all_constraints:
        a['cholV_cons'][ cons ] = spla.cholesky(a['V_cons'][cons], lower=False)

    return a

def two_by_two_symmetric_matrix_inverse(a, b, c):
	
	det = a * b - c * c 
	a_new = 1.0 / det * b
	b_new = 1.0 / det * a
	c_new = 1.0 / det * - c

	return a_new, b_new, c_new

def two_by_two_matrix_inverse(a, b, c, d):
	
	det = a * d - c * b 
	a_new = 1.0 / det * d
	b_new = 1.0 / det * -b
	c_new = 1.0 / det * - c
	d_new = 1.0 / det * a

	return a_new, b_new, c_new, d_new

def two_by_two_symmetric_matrix_product_vector(a, b, c, v_a, v_b):
	
	return a * v_a + c * v_b, c * v_a + b * v_b

#############################################################################################################################
# updateFactors_fast_daniel_no_constraints_pareto_set_robust: Computes EP approximations for factors that do not depend on the candidate/s.
#
# INPUT: 
# a: Data structure with CPDs, PDs and EP factors. Unconditioned.
# damping: Damping factor.
# minimize: If FALSE, it maximizes.
# no_negative_variances_nor_nands: Boolean flag that control robustness.
# OUTPUT:
# a: Data structure with CPDs, PDs and EP factors. Conditioned.
#############################################################################################################################
def update_full_Factors(a, damping, minimize=True, no_negative_variances_nor_nands = False, no_negatives = True):
    # used to switch between minimizing and maximizing
    sgn = -1.0 if minimize else 1.0

    # We update the h factors
    all_tasks = a['objs']
    all_constraints = a['cons']
    n_obs = a['n_obs']
    n_pset = a['n_pset']
    n_test = a['n_test']
    n_total = a['n_total']
    q = a['q']
    c = a['c']

    alpha = np.zeros(a['q'])
    s = np.zeros(a['q'])
    ratio_cons = np.zeros(c)
    
    # First we update the factors corresponding to the observed data

    # We compute an "old" distribution 

    # Data structures for objective npset nobs cavities (a, b).
    m_pset = np.zeros((q, n_pset, n_obs))
    m_obs = np.zeros((q, n_pset, n_obs))
    v_pset = np.zeros((q, n_pset, n_obs))
    v_obs = np.zeros((q, n_pset, n_obs))
    v_cov = np.zeros((q, n_pset, n_obs))

    # Data structures for constraint npset nobs cavities (c_a, c_b).

    c_m = np.zeros((c, n_pset, n_obs))
    c_v = np.zeros((c, n_pset, n_obs))

    # We do the update of the external factors.            

    c_m_external = np.zeros((c, n_pset))
    c_v_external = np.zeros((c, n_pset))
    alpha_external = np.zeros(c)
    
    # First we update the external (easy) factors that only affect the constraints

    n_task = 0
    for cons in all_constraints:
        c_m_external[ n_task, : ] = a['m_cons'][ cons ][ n_obs : n_obs + n_pset ]
        c_v_external[ n_task, : ] = np.diag(a['V_cons'][ cons ])[ n_obs : n_obs + n_pset ]
        n_task += 1

    c_vTilde_external = a['ehfhat'][ :, : ].T
    c_mTilde_external = a['fhfhat'][ :, : ].T

    c_vOld_external = 1.0 / (1.0 / c_v_external - c_vTilde_external)
    c_mOld_external = c_vOld_external * (c_m_external / c_v_external - c_mTilde_external)

    if np.any(c_vOld_external < 0):
        raise npla.linalg.LinAlgError("Negative value in the sqrt!")
    
    alpha_external = c_mOld_external / np.sqrt(c_vOld_external)
    logZ = logcdf_robust(alpha_external)
    
    ratio = np.exp(sps.norm.logpdf(alpha_external) - logZ)
    dlogZdmfOld_external = ratio / np.sqrt(c_vOld_external) 
    dlogZdmfOld2_external = - ratio * (alpha_external + ratio) / c_vOld_external
    
     # We find the parameters of the updated factors
		
    ehfhat_act = (- dlogZdmfOld2_external / (1.0 + dlogZdmfOld2_external * c_vOld_external)).T
    fhfhat_act = ((dlogZdmfOld_external - c_mOld_external * dlogZdmfOld2_external) / (1.0 + dlogZdmfOld2_external * c_vOld_external)).T
    
    if no_negative_variances_nor_nands == True:
        neg = np.where(ehfhat_act < 0)

        ehfhat_act[ neg ] = 0.0
        fhfhat_act[ neg ] = 0.0
 
    a['ehfhat'][ :, : ] = ehfhat_act * damping + (1 - damping) * a['ehfhat'][ :, : ] 
    a['fhfhat'][ :, : ] = fhfhat_act * damping + (1 - damping) * a['fhfhat'][ :, : ]

    # Done!
  
    n_task = 0 
    for obj in all_tasks:
        m_obs[ n_task, :, : ] = np.tile(a['m'][ obj ][ 0 : n_obs ], n_pset).reshape((n_pset, n_obs))
        m_pset[ n_task, :, : ] = np.tile(a['m'][ obj ][ n_obs : n_obs + n_pset ], n_obs).reshape((n_obs, n_pset)).T
        v_cov[ n_task, :, : ] = a['V'][ obj ][ n_obs : n_obs + n_pset, 0 : n_obs ]
        v_obs[ n_task, :, : ] = np.tile(np.diag(a['V'][ obj ])[ 0 : n_obs ], n_pset).reshape((n_pset, n_obs))
        v_pset[ n_task, :, : ] = np.tile(np.diag(a['V'][ obj ])[ n_obs : n_obs + n_pset ], n_obs).reshape((n_obs, n_pset)).T
        n_task += 1
        
    n_task = 0
    
    for cons in all_constraints:
        c_m[ n_task, :, : ] = a['m_cons'][ cons ][ 0 : n_obs ]
        c_v[ n_task, :, : ] = np.diag(a['V_cons'][ cons ])[ 0 : n_obs ]
        n_task += 1
   
    #ECGM: I think that this remains being the same. 
    vTilde_obs = a['ahfhat'][ :, :, :, 0, 0 ].T
    vTilde_pset = a['ahfhat'][ :, :, :, 1, 1 ].T
    covTilde = a['ahfhat'][ :, :, :, 0, 1 ].T
    mTilde_obs = a['bhfhat'][ :, :, :, 0, ].T
    mTilde_pset = a['bhfhat'][ :, :, :, 1, ].T
    
    c_vTilde = a['a_c_hfhat'][ :, :, : ].T
    c_mTilde = a['b_c_hfhat'][ :, :, : ].T

    # Obtaining cavities.

    inv_v_obs, inv_v_pset, inv_v_cov = two_by_two_symmetric_matrix_inverse(v_obs, v_pset, v_cov)
    inv_c_v = 1.0 / c_v
    
    inv_vOld_obs = inv_v_obs - vTilde_obs
    inv_vOld_pset = inv_v_pset - vTilde_pset
    inv_vOld_cov =  inv_v_cov - covTilde
    inv_c_vOld = inv_c_v - c_vTilde

    vOld_obs, vOld_pset, vOld_cov = two_by_two_symmetric_matrix_inverse(inv_vOld_obs, inv_vOld_pset, inv_vOld_cov)
    c_vOld = 1.0 / inv_c_vOld

    mOld_obs, mOld_pset  = two_by_two_symmetric_matrix_product_vector(inv_v_obs, inv_v_pset, inv_v_cov, m_obs, m_pset)
    mOld_obs = mOld_obs - mTilde_obs
    mOld_pset = mOld_pset - mTilde_pset
    mOld_obs, mOld_pset  = two_by_two_symmetric_matrix_product_vector(vOld_obs, vOld_pset, vOld_cov, mOld_obs, mOld_pset)
    
    c_mOld = c_vOld * (c_m / c_v - c_mTilde)
                
    #Computing factors.

    s = vOld_pset + vOld_obs - 2 * vOld_cov
    s_cons = c_vOld
    
    if np.any(vOld_pset < 0):
        raise npla.linalg.LinAlgError("Negative variance in the sqrt!")
        
    if np.any(vOld_obs < 0):
        raise npla.linalg.LinAlgError("Negative variance in the sqrt!")
 
    if np.any(c_vOld < 0):
        raise npla.linalg.LinAlgError("Negative value in the sqrt!")

    alpha_cons = c_mOld / np.sqrt(c_vOld)

    scale = 1.0 - 1e-4
    while np.any(s / (vOld_pset + vOld_obs) < 1e-6):
        scale = scale**2
        s = vOld_pset + vOld_obs - 2 * vOld_cov * scale

    alpha = (mOld_obs - mOld_pset) / np.sqrt(s) * sgn

    log_phi = logcdf_robust(alpha)
    log_phi_cons = logcdf_robust(alpha_cons)

    logZ_orig = log_1_minus_exp_x(np.sum(log_phi, axis = 0))

    if np.any(np.logical_not(logZ_orig == -np.inf)):
        sel = (logZ_orig == -np.inf)
        logZ_orig[ sel ] = logcdf_robust(-np.min(alpha[ :, sel ], axis = 0))

    logZ_term1 = np.sum(log_phi_cons, axis = 0) + logZ_orig 
    logZ_term2 = log_1_minus_exp_x(np.sum(log_phi_cons, axis = 0))

    if np.any(np.logical_not(logZ_term2 == -np.inf)):
        sel = (logZ_term2 == -np.inf)
        logZ_term2[ sel ] = logcdf_robust(-np.min(alpha_cons[ :, sel ], axis = 0))

    # TODO: This should be done robustly

    max_value = np.maximum(logZ_term1, logZ_term2)
    
    logZ = np.tile(np.log(np.exp(logZ_term1 - max_value) + np.exp(logZ_term2 - max_value)) + \
        max_value, q).reshape((n_pset, q, n_obs)).swapaxes(0, 1)
    logZ_cons = np.tile(np.log(np.exp(logZ_term1 - max_value) + np.exp(logZ_term2 - max_value)) + \
        max_value, c).reshape((n_pset, c, n_obs)).swapaxes(0, 1)

    log_phi_sum = np.tile(np.sum(log_phi, axis = 0), q).reshape((n_pset, q, n_obs)).swapaxes(0, 1)
    log_phi_sum_cons = np.tile(np.sum(log_phi_cons, axis = 0), q).reshape((n_pset, q, n_obs)).swapaxes(0, 1)

    ratio = - np.exp(sps.norm.logpdf(alpha) - logZ + log_phi_sum - logcdf_robust(alpha) + log_phi_sum_cons)

    logZ_orig_cons = np.tile(logZ_orig, c).reshape((n_pset, c, n_obs)).swapaxes(0, 1)
    log_phi_sum_cons = np.tile(np.sum(log_phi_cons, axis = 0), c).reshape((n_pset, c, n_obs)).swapaxes(0, 1)

    ratio_cons = np.exp(sps.norm.logpdf(alpha_cons) - logZ_cons + logZ_orig_cons + log_phi_sum_cons - logcdf_robust(alpha_cons)) - \
        np.exp(sps.norm.logpdf(alpha_cons) - logZ_cons + log_phi_sum_cons - logcdf_robust(alpha_cons))
    
    # Derivatives, non vector form.

    dlogZdmfOld_obs = ratio / np.sqrt(s) * sgn
    dlogZdmfOld_pset = ratio / np.sqrt(s) * -1.0 * sgn

    dlogZdmfOld_obs2 = - ratio / s * (alpha + ratio) 
    dlogZdmfOld_pset2 = - ratio / s * (alpha + ratio) 
    dlogZdmfOld_cov2 = - ratio / s * (alpha + ratio) * -1.0
	
    dlogZdmcOld = ratio_cons / np.sqrt(s_cons)
    dlogZdmcOld2 = - ratio_cons / s_cons * (alpha_cons + ratio_cons)

    a_VfOld_times_dlogZdmfOld2 = vOld_obs * dlogZdmfOld_obs2 + vOld_cov * dlogZdmfOld_cov2 + 1.0
    b_VfOld_times_dlogZdmfOld2 = vOld_obs * dlogZdmfOld_cov2 + vOld_cov * dlogZdmfOld_pset2 
    c_VfOld_times_dlogZdmfOld2 = vOld_cov * dlogZdmfOld_obs2 + vOld_pset * dlogZdmfOld_cov2 
    d_VfOld_times_dlogZdmfOld2 = vOld_cov * dlogZdmfOld_cov2 + vOld_pset * dlogZdmfOld_pset2 + 1.0 

    a_inv, b_inv, c_inv, d_inv = two_by_two_matrix_inverse(a_VfOld_times_dlogZdmfOld2, b_VfOld_times_dlogZdmfOld2, \
        c_VfOld_times_dlogZdmfOld2, d_VfOld_times_dlogZdmfOld2)

    vTilde_obs_new = - (dlogZdmfOld_obs2 * a_inv + dlogZdmfOld_cov2 * c_inv)
    vTilde_pset_new = - (dlogZdmfOld_cov2 * b_inv + dlogZdmfOld_pset2 * d_inv)
    vTilde_cov_new = - (dlogZdmfOld_obs2 * b_inv + dlogZdmfOld_cov2 * d_inv)

    v_1, v_2 = two_by_two_symmetric_matrix_product_vector(dlogZdmfOld_obs2, \
        dlogZdmfOld_pset2, dlogZdmfOld_cov2, mOld_obs, mOld_pset)

    v_1 = dlogZdmfOld_obs - v_1 
    v_2 = dlogZdmfOld_pset - v_2 
    mTilde_obs_new = v_1 * a_inv + v_2 * c_inv
    mTilde_pset_new = v_1 * b_inv + v_2 * d_inv

    vTilde_cons =  - dlogZdmcOld2 / (1.0 + dlogZdmcOld2 * c_vOld)
    mTilde_cons = (dlogZdmcOld - c_mOld * dlogZdmcOld2) / (1.0 + dlogZdmcOld2 * c_vOld)

    if no_negative_variances_nor_nands == True:

        finite = np.logical_and(np.logical_and(np.logical_and(np.logical_and(np.isfinite(vTilde_obs_new), np.isfinite(vTilde_pset_new)), \
            np.isfinite(vTilde_cov_new)), np.isfinite(mTilde_obs_new)), np.isfinite(mTilde_pset_new))
        
        c_finite = np.logical_and(np.isfinite(vTilde_cons), np.isfinite(mTilde_cons))

        neg1 = np.where(np.logical_or(np.logical_not(finite), vTilde_obs_new < 0))
        neg2 = np.where(np.logical_or(np.logical_not(finite), vTilde_pset_new < 0))
        c_neg = np.where(np.logical_or(np.logical_not(c_finite), vTilde_cons < 0))

        vTilde_obs_new[ neg1 ] = 0.0
        vTilde_obs_new[ neg2 ] = 0.0
        vTilde_pset_new[ neg1 ] = 0.0
        vTilde_pset_new[ neg2 ] = 0.0
        vTilde_cov_new[ neg1 ] = 0.0
        vTilde_cov_new[ neg2 ] = 0.0
        mTilde_obs_new[ neg1 ] = 0.0
        mTilde_obs_new[ neg2 ] = 0.0
        mTilde_pset_new[ neg1 ] = 0.0
        mTilde_pset_new[ neg2 ] = 0.0
        vTilde_cons[ c_neg ] = 0.0
        mTilde_cons[ c_neg ] = 0.0

    # We do the actual update

    a_c_hfHatNew = vTilde_cons
    b_c_hfHatNew = mTilde_cons

    a['ahfhat'][ :, :, :, 0, 0 ] = vTilde_obs_new.T * damping + (1 - damping) * a['ahfhat'][ :, :, :, 0, 0 ] 
    a['ahfhat'][ :, :, :, 1, 1 ] = vTilde_pset_new.T * damping + (1 - damping) * a['ahfhat'][ :, :, :, 1, 1 ] 
    a['ahfhat'][ :, :, :, 0, 1 ] = vTilde_cov_new.T * damping + (1 - damping) * a['ahfhat'][ :, :, :, 0, 1 ] 
    a['ahfhat'][ :, :, :, 1, 0 ] = vTilde_cov_new.T * damping + (1 - damping) * a['ahfhat'][ :, :, :, 1, 0 ] 
    a['bhfhat'][ :, :, :, 0 ] = mTilde_obs_new.T * damping + (1 - damping) * a['bhfhat'][ :, :, :, 0 ] 
    a['bhfhat'][ :, :, :, 1 ] = mTilde_pset_new.T * damping + (1 - damping) * a['bhfhat'][ :, :, :, 1 ]
    a['a_c_hfhat'][ :, :, : ] = a_c_hfHatNew.T * damping + (1 - damping) * a['a_c_hfhat'][ :, :, : ] 
    a['b_c_hfhat'][ :, :, : ] = b_c_hfHatNew.T * damping + (1 - damping) * a['b_c_hfhat'][ :, :, : ]
    
    # Second we update the factors corresponding to the pareto set

    # We compute an "old" distribution 

    m_pset1 = np.zeros((q, n_pset, n_pset))
    m_pset2 = np.zeros((q, n_pset, n_pset))
    v_pset1 = np.zeros((q, n_pset, n_pset))
    v_pset2 = np.zeros((q, n_pset, n_pset))
    v_cov = np.zeros((q, n_pset, n_pset))
    
    c_m_pset = np.zeros((c, n_pset, n_pset))
    c_v_pset = np.zeros((c, n_pset, n_pset))

    #Changes done as now the observations lie first and pareto set points second.
    n_task = 0
    for obj in all_tasks:
        m_pset1[ n_task, :, : ] = np.tile(a['m'][ obj ][ n_obs : n_obs + n_pset ], n_pset).reshape((n_pset, n_pset))
        m_pset2[ n_task, :, : ] = np.tile(a['m'][ obj ][ n_obs : n_obs + n_pset ], n_pset).reshape((n_pset, n_pset)).T
        v_cov[ n_task, :, : ] = a['V'][ obj ][ n_obs : n_obs + n_pset, n_obs : n_obs + n_pset ]
        v_cov[ n_task, :, : ] = v_cov[ n_task, :, : ] - np.diag(np.diag(v_cov[ n_task, :, : ])) 
        v_pset1[ n_task, :, : ] = np.tile(np.diag(a['V'][ obj ])[ n_obs : n_obs + n_pset ], n_pset).reshape((n_pset, n_pset))
        v_pset2[ n_task, :, : ] = np.tile(np.diag(a['V'][ obj ])[ n_obs : n_obs + n_pset ], n_pset).reshape((n_pset, n_pset)).T
        n_task += 1

    n_task = 0
    for cons in all_constraints:
        c_m_pset[ n_task, :, : ] = a['m_cons'][ cons ][ n_obs : n_obs + n_pset ]
        c_v_pset[ n_task, :, : ] = np.diag(a['V_cons'][ cons ])[n_obs : n_obs + n_pset]
        n_task += 1
    
    vTilde_pset1 = a['chfhat'][ :, :, :, 0, 0 ].T
    vTilde_pset2 = a['chfhat'][ :, :, :, 1, 1 ].T
    covTilde = a['chfhat'][ :, :, :, 0, 1 ].T
    mTilde_pset1 = a['dhfhat'][ :, :, :, 0 ].T
    mTilde_pset2 = a['dhfhat'][ :, :, :, 1 ].T
    
    c_vTilde_pset = a['c_c_hfhat'][ :, :, : ].T
    c_mTilde_pset = a['d_c_hfhat'][ :, :, : ].T

    inv_v_pset1, inv_v_pset2, inv_v_cov = two_by_two_symmetric_matrix_inverse(v_pset1, v_pset2, v_cov)

    inv_vOld_pset1 = inv_v_pset1 - vTilde_pset1
    inv_vOld_pset2 = inv_v_pset2 - vTilde_pset2
    inv_vOld_cov =  inv_v_cov - covTilde

    vOld_pset1, vOld_pset2, vOld_cov = two_by_two_symmetric_matrix_inverse(inv_vOld_pset1, inv_vOld_pset2, inv_vOld_cov)

    mOld_pset1, mOld_pset2  = two_by_two_symmetric_matrix_product_vector(inv_v_pset1, inv_v_pset2, inv_v_cov, m_pset1, m_pset2)
    mOld_pset1 = mOld_pset1 - mTilde_pset1
    mOld_pset2 = mOld_pset2 - mTilde_pset2
    mOld_pset1, mOld_pset2  = two_by_two_symmetric_matrix_product_vector(vOld_pset1, vOld_pset2, vOld_cov, mOld_pset1, mOld_pset2)

    s = vOld_pset1 + vOld_pset2 - 2 * vOld_cov
	
    if np.any(vOld_pset1  < 0):
        raise npla.linalg.LinAlgError("Negative value in the sqrt!")

    if np.any(vOld_pset2  < 0):
        raise npla.linalg.LinAlgError("Negative value in the sqrt!")
        
    scale = 1.0 - 1e-4
    while np.any(s / (vOld_pset1 + vOld_pset2) < 1e-6):
        print "Reducing scale to guarantee positivity!"
        scale = scale**2
        s = vOld_pset1 + vOld_pset2 - 2 * vOld_cov * scale

    alpha = (mOld_pset1 - mOld_pset2) / np.sqrt(s) * sgn    
    
    log_phi = logcdf_robust(alpha)
    logZ = log_1_minus_exp_x(np.sum(log_phi, axis = 0))

    if np.any(np.logical_not(logZ == -np.inf)):
        sel = (logZ == -np.inf)
        logZ[ sel ] = logcdf_robust(-np.min(alpha[ :, sel ], axis = 0))

    log_phi_sum = np.tile(np.sum(log_phi, axis = 0), q).reshape((n_pset, q, n_pset)).swapaxes(0, 1)

    ratio = - np.exp(sps.norm.logpdf(alpha) - logZ + log_phi_sum - logcdf_robust(alpha))

    # Derivatives, non vector form.
    
    dlogZdmfOld_pset1 = ratio / np.sqrt(s) * sgn
    dlogZdmfOld_pset2 = ratio / np.sqrt(s) * -1.0 * sgn

    dlogZdmfOld_pset12 = - ratio / s * (alpha + ratio) 
    dlogZdmfOld_pset22 = - ratio / s * (alpha + ratio) 
    dlogZdmfOld_cov2 = - ratio / s * (alpha + ratio) * -1.0
	
    a_VfOld_times_dlogZdmfOld2 = vOld_pset1 * dlogZdmfOld_pset12 + vOld_cov * dlogZdmfOld_cov2 + 1.0
    b_VfOld_times_dlogZdmfOld2 = vOld_pset1 * dlogZdmfOld_cov2 + vOld_cov * dlogZdmfOld_pset22 
    c_VfOld_times_dlogZdmfOld2 = vOld_cov * dlogZdmfOld_pset12 + vOld_pset2 * dlogZdmfOld_cov2 
    d_VfOld_times_dlogZdmfOld2 = vOld_cov * dlogZdmfOld_cov2 + vOld_pset2 * dlogZdmfOld_pset22 + 1.0 

    a_inv, b_inv, c_inv, d_inv = two_by_two_matrix_inverse(a_VfOld_times_dlogZdmfOld2, b_VfOld_times_dlogZdmfOld2, \
        c_VfOld_times_dlogZdmfOld2, d_VfOld_times_dlogZdmfOld2)

    vTilde_pset1_new = - (dlogZdmfOld_pset12 * a_inv + dlogZdmfOld_cov2 * c_inv)
    vTilde_pset2_new = - (dlogZdmfOld_cov2 * b_inv + dlogZdmfOld_pset22 * d_inv)
    vTilde_cov_new = - (dlogZdmfOld_pset12 * b_inv + dlogZdmfOld_cov2 * d_inv)

    v_1, v_2 = two_by_two_symmetric_matrix_product_vector(dlogZdmfOld_pset12, \
        dlogZdmfOld_pset22, dlogZdmfOld_cov2, mOld_pset1, mOld_pset2)

    v_1 = dlogZdmfOld_pset1 - v_1 
    v_2 = dlogZdmfOld_pset2 - v_2 
    mTilde_pset1_new = v_1 * a_inv + v_2 * c_inv
    mTilde_pset2_new = v_1 * b_inv + v_2 * d_inv

    n_task = 0
    for obj in all_tasks:
        vTilde_pset1_new[ n_task, :, : ] -= np.diag(np.diag(vTilde_pset1_new[ n_task, :, :, ]))
        vTilde_pset2_new[ n_task, :, : ] -= np.diag(np.diag(vTilde_pset2_new[ n_task, :, :, ]))
        vTilde_cov_new[ n_task, :, : ] -= np.diag(np.diag(vTilde_cov_new[ n_task, :, : ]))
        mTilde_pset1_new[ n_task, :, : ] -= np.diag(np.diag(mTilde_pset1_new[ n_task, :, : ]))
        mTilde_pset2_new[ n_task, :, : ] -= np.diag(np.diag(mTilde_pset2_new[ n_task, :, : ]))
        n_task += 1

    if no_negative_variances_nor_nands == True:
        finite = np.logical_and(np.logical_and(np.logical_and(np.logical_and(np.isfinite(vTilde_pset1_new), np.isfinite(vTilde_pset2_new)), \
            np.isfinite(vTilde_cov_new)), np.isfinite(mTilde_pset1_new)), np.isfinite(mTilde_pset2_new))
        
        neg1 = np.where(np.logical_or(np.logical_not(finite), vTilde_pset1_new < 0))
        neg2 = np.where(np.logical_or(np.logical_not(finite), vTilde_pset2_new < 0))

        vTilde_pset1_new[ neg1 ] = 0.0
        vTilde_pset1_new[ neg2 ] = 0.0
        vTilde_pset2_new[ neg1 ] = 0.0
        vTilde_pset2_new[ neg2 ] = 0.0
        vTilde_cov_new[ neg1 ] = 0.0
        vTilde_cov_new[ neg2 ] = 0.0
        mTilde_pset1_new[ neg1 ] = 0.0
        mTilde_pset1_new[ neg2 ] = 0.0
        mTilde_pset2_new[ neg1 ] = 0.0
        mTilde_pset2_new[ neg2 ] = 0.0

    # We do the actual update

    #i not equal to j condition.

    a['chfhat'][ :, :, :, 0, 0 ] = vTilde_pset1_new.T * damping + (1 - damping) * a['chfhat'][ :, :, :, 0, 0 ] 
    a['chfhat'][ :, :, :, 1, 1 ] = vTilde_pset2_new.T * damping + (1 - damping) * a['chfhat'][ :, :, :, 1, 1 ] 
    a['chfhat'][ :, :, :, 0, 1 ] = vTilde_cov_new.T * damping + (1 - damping) * a['chfhat'][ :, :, :, 0, 1 ] 
    a['chfhat'][ :, :, :, 1, 0 ] = vTilde_cov_new.T * damping + (1 - damping) * a['chfhat'][ :, :, :, 1, 0 ] 
    a['dhfhat'][ :, :, :, 0 ] = mTilde_pset1_new.T * damping + (1 - damping) * a['dhfhat'][ :, :, :, 0 ] 
    a['dhfhat'][ :, :, :, 1 ] = mTilde_pset2_new.T * damping + (1 - damping) * a['dhfhat'][ :, :, :, 1 ]


    #Test factors.

    # Data structures for objective npset ntest cavities (a, b).

    m_pset = np.zeros((q, n_pset, n_test))
    m_test = np.zeros((q, n_pset, n_test))
    v_pset = np.zeros((q, n_pset, n_test))
    v_test = np.zeros((q, n_pset, n_test))
    v_cov = np.zeros((q, n_pset, n_test))

    # Data structures for constraint npset nobs cavities (c_a, c_b).

    c_m = np.zeros((c, n_pset, n_test))
    c_v = np.zeros((c, n_pset, n_test))

    n_task = 0 
    for obj in all_tasks:
        m_test[ n_task, :, : ] = np.tile(a['m'][ obj ][ n_obs + n_pset : n_total ], n_pset).reshape((n_pset, n_test))
        m_pset[ n_task, :, : ] = np.tile(a['m'][ obj ][ n_obs : n_obs + n_pset ], n_test).reshape((n_test, n_pset)).T
        v_cov[ n_task, :, : ] = a['V'][ obj ][ n_obs : n_obs + n_pset, n_obs + n_pset : n_total ]
        v_test[ n_task, :, : ] = np.tile(np.diag(a['V'][ obj ])[ n_obs + n_pset : n_total ], n_pset).reshape((n_pset, n_test))
        v_pset[ n_task, :, : ] = np.tile(np.diag(a['V'][ obj ])[ n_obs : n_obs + n_pset ], n_test).reshape((n_test, n_pset)).T
        n_task += 1
        
    n_task = 0
    
    for cons in all_constraints:
        c_m[ n_task, :, : ] = a['m_cons'][ cons ][ n_obs + n_pset : n_total ]
        c_v[ n_task, :, : ] = np.diag(a['V_cons'][ cons ])[ n_obs + n_pset : n_total ]
        n_task += 1
   
    vTilde_test = a['ghfhat'][ :, :, :, 0, 0 ].T
    vTilde_pset = a['ghfhat'][ :, :, :, 1, 1 ].T
    vTilde_cov = a['ghfhat'][ :, :, :, 0, 1 ].T
    mTilde_test = a['hhfhat'][ :, :, :, 0 ].T
    mTilde_pset = a['hhfhat'][ :, :, :, 1 ].T

    vTilde_test_cons = a['g_c_hfhat'][:, :, :].T
    mTilde_test_cons = a['h_c_hfhat'][:, :, :].T

    # Obtaining cavities.

    inv_v_test, inv_v_pset, inv_v_cov = two_by_two_symmetric_matrix_inverse(v_test, v_pset, v_cov)
    inv_c_v = 1.0 / c_v
    
    inv_vOld_test = inv_v_test - vTilde_test
    inv_vOld_pset = inv_v_pset - vTilde_pset
    inv_vOld_cov =  inv_v_cov - vTilde_cov
    inv_c_vOld = inv_c_v - vTilde_test_cons

    vOld_test, vOld_pset, vOld_cov = two_by_two_symmetric_matrix_inverse(inv_vOld_test, inv_vOld_pset, inv_vOld_cov)
    c_vOld = 1.0 / inv_c_vOld

    mOld_test, mOld_pset = two_by_two_symmetric_matrix_product_vector(inv_v_test, inv_v_pset, inv_v_cov, m_test, m_pset)
    mOld_test = mOld_test - mTilde_test
    mOld_pset = mOld_pset - mTilde_pset
    mOld_test, mOld_pset  = two_by_two_symmetric_matrix_product_vector(vOld_test, vOld_pset, vOld_cov, mOld_test, mOld_pset)
    
    c_mOld = c_vOld * (c_m / c_v - mTilde_test_cons)
                
    # Computing factors.

    s = vOld_pset + vOld_test - 2 * vOld_cov
    s_cons = c_vOld
    
    if np.any(vOld_pset < 0):
        raise npla.linalg.LinAlgError("Negative variance in the sqrt!")
        
    if np.any(vOld_test < 0):
        raise npla.linalg.LinAlgError("Negative variance in the sqrt!")
 
    if np.any(c_vOld < 0):
        raise npla.linalg.LinAlgError("Negative value in the sqrt!")

    alpha_cons = c_mOld / np.sqrt(c_vOld)

    scale = 1.0 - 1e-4
    while np.any(s / (vOld_pset + vOld_test) < 1e-6):
        scale = scale**2
        s = vOld_pset + vOld_test - 2 * vOld_cov * scale

    alpha = (mOld_test - mOld_pset) / np.sqrt(s) * sgn

    log_phi = logcdf_robust(alpha)
    log_phi_cons = logcdf_robust(alpha_cons)

    logZ_orig = log_1_minus_exp_x(np.sum(log_phi, axis = 0))

    if np.any(np.logical_not(logZ_orig == -np.inf)):
        sel = (logZ_orig == -np.inf)
        logZ_orig[ sel ] = logcdf_robust(-np.min(alpha[ :, sel ], axis = 0))

    logZ_term1 = np.sum(log_phi_cons, axis = 0) + logZ_orig 
    logZ_term2 = log_1_minus_exp_x(np.sum(log_phi_cons, axis = 0))

    if np.any(np.logical_not(logZ_term2 == -np.inf)):
        sel = (logZ_term2 == -np.inf)
        logZ_term2[ sel ] = logcdf_robust(-np.min(alpha_cons[ :, sel ], axis = 0))

    # TODO: This should be done robustly

    max_value = np.maximum(logZ_term1, logZ_term2)
    
    logZ = np.tile(np.log(np.exp(logZ_term1 - max_value) + np.exp(logZ_term2 - max_value)) + \
        max_value, q).reshape((n_pset, q, n_test)).swapaxes(0, 1)
    logZ_cons = np.tile(np.log(np.exp(logZ_term1 - max_value) + np.exp(logZ_term2 - max_value)) + \
        max_value, c).reshape((n_pset, c, n_test)).swapaxes(0, 1)

    log_phi_sum = np.tile(np.sum(log_phi, axis = 0), q).reshape((n_pset, q, n_test)).swapaxes(0, 1)
    log_phi_sum_cons = np.tile(np.sum(log_phi_cons, axis = 0), q).reshape((n_pset, q, n_test)).swapaxes(0, 1)

    ratio = - np.exp(sps.norm.logpdf(alpha) - logZ + log_phi_sum - logcdf_robust(alpha) + log_phi_sum_cons)

    logZ_orig_cons = np.tile(logZ_orig, c).reshape((n_pset, c, n_test)).swapaxes(0, 1)
    log_phi_sum_cons = np.tile(np.sum(log_phi_cons, axis = 0), c).reshape((n_pset, c, n_test)).swapaxes(0, 1)

    ratio_cons = np.exp(sps.norm.logpdf(alpha_cons) - logZ_cons + logZ_orig_cons + log_phi_sum_cons - logcdf_robust(alpha_cons)) - \
        np.exp(sps.norm.logpdf(alpha_cons) - logZ_cons + log_phi_sum_cons - logcdf_robust(alpha_cons))
    
    # Derivatives, non vector form.

    dlogZdmfOld_test = ratio / np.sqrt(s) * sgn
    dlogZdmfOld_pset = ratio / np.sqrt(s) * -1.0 * sgn

    dlogZdmfOld_test2 = - ratio / s * (alpha + ratio) 
    dlogZdmfOld_pset2 = - ratio / s * (alpha + ratio) 
    dlogZdmfOld_cov2 = - ratio / s * (alpha + ratio) * -1.0
	
    dlogZdmcOld = ratio_cons / np.sqrt(s_cons)
    dlogZdmcOld2 = - ratio_cons / s_cons * (alpha_cons + ratio_cons)

    a_VfOld_times_dlogZdmfOld2 = vOld_test * dlogZdmfOld_test2 + vOld_cov * dlogZdmfOld_cov2 + 1.0
    b_VfOld_times_dlogZdmfOld2 = vOld_test * dlogZdmfOld_cov2 + vOld_cov * dlogZdmfOld_pset2 
    c_VfOld_times_dlogZdmfOld2 = vOld_cov * dlogZdmfOld_test2 + vOld_pset * dlogZdmfOld_cov2 
    d_VfOld_times_dlogZdmfOld2 = vOld_cov * dlogZdmfOld_cov2 + vOld_pset * dlogZdmfOld_pset2 + 1.0 

    a_inv, b_inv, c_inv, d_inv = two_by_two_matrix_inverse(a_VfOld_times_dlogZdmfOld2, b_VfOld_times_dlogZdmfOld2, \
        c_VfOld_times_dlogZdmfOld2, d_VfOld_times_dlogZdmfOld2)

    vTilde_test_new = - (dlogZdmfOld_test2 * a_inv + dlogZdmfOld_cov2 * c_inv)
    vTilde_pset_new = - (dlogZdmfOld_cov2 * b_inv + dlogZdmfOld_pset2 * d_inv)
    vTilde_cov_new = - (dlogZdmfOld_test2 * b_inv + dlogZdmfOld_cov2 * d_inv)

    v_1, v_2 = two_by_two_symmetric_matrix_product_vector(dlogZdmfOld_test2, \
        dlogZdmfOld_pset2, dlogZdmfOld_cov2, mOld_test, mOld_pset)

    v_1 = dlogZdmfOld_test - v_1 
    v_2 = dlogZdmfOld_pset - v_2 
    mTilde_test_new = v_1 * a_inv + v_2 * c_inv
    mTilde_pset_new = v_1 * b_inv + v_2 * d_inv

    vTilde_cons =  - dlogZdmcOld2 / (1.0 + dlogZdmcOld2 * c_vOld)
    mTilde_cons = (dlogZdmcOld - c_mOld * dlogZdmcOld2) / (1.0 + dlogZdmcOld2 * c_vOld)

    if no_negative_variances_nor_nands == True:

        finite = np.logical_and(np.logical_and(np.logical_and(np.logical_and(np.isfinite(vTilde_test_new), np.isfinite(vTilde_pset_new)), \
            np.isfinite(vTilde_cov_new)), np.isfinite(mTilde_test_new)), np.isfinite(mTilde_pset_new))
        
        c_finite = np.logical_and(np.isfinite(vTilde_cons), np.isfinite(mTilde_cons))

        neg1 = np.where(np.logical_or(np.logical_not(finite), vTilde_test_new < 0))
        neg2 = np.where(np.logical_or(np.logical_not(finite), vTilde_pset_new < 0))
        c_neg = np.where(np.logical_or(np.logical_not(c_finite), vTilde_cons < 0))

        vTilde_test_new[ neg1 ] = 0.0
        vTilde_test_new[ neg2 ] = 0.0
        vTilde_pset_new[ neg1 ] = 0.0
        vTilde_pset_new[ neg2 ] = 0.0
        vTilde_cov_new[ neg1 ] = 0.0
        vTilde_cov_new[ neg2 ] = 0.0
        mTilde_test_new[ neg1 ] = 0.0
        mTilde_test_new[ neg2 ] = 0.0
        mTilde_pset_new[ neg1 ] = 0.0
        mTilde_pset_new[ neg2 ] = 0.0
        vTilde_cons[ c_neg ] = 0.0
        mTilde_cons[ c_neg ] = 0.0

    # We do the actual update

    g_c_hfHatNew = vTilde_cons
    h_c_hfHatNew = mTilde_cons

    a['ghfhat'][ :, :, :, 0, 0 ] = vTilde_test_new.T * damping + (1 - damping) * a['ghfhat'][ :, :, :, 0, 0 ] 
    a['ghfhat'][ :, :, :, 1, 1 ] = vTilde_pset_new.T * damping + (1 - damping) * a['ghfhat'][ :, :, :, 1, 1 ] 
    a['ghfhat'][ :, :, :, 0, 1 ] = vTilde_cov_new.T * damping + (1 - damping) * a['ghfhat'][ :, :, :, 0, 1 ] 
    a['ghfhat'][ :, :, :, 1, 0 ] = vTilde_cov_new.T * damping + (1 - damping) * a['ghfhat'][ :, :, :, 1, 0 ] 
    a['hhfhat'][ :, :, :, 0 ] = mTilde_test_new.T * damping + (1 - damping) * a['hhfhat'][ :, :, :, 0 ] 
    a['hhfhat'][ :, :, :, 1 ] = mTilde_pset_new.T * damping + (1 - damping) * a['hhfhat'][ :, :, :, 1 ]
    a['g_c_hfhat'][ :, :, : ] = g_c_hfHatNew.T * damping + (1 - damping) * a['g_c_hfhat'][ :, :, : ] 
    a['h_c_hfhat'][ :, :, : ] = h_c_hfHatNew.T * damping + (1 - damping) * a['h_c_hfhat'][ :, :, : ]

    return a


#############################################################################################################################
# updateFactors_fast_daniel_no_constraints_pareto_set_robust: Computes EP approximations for factors that do not depend on the candidate/s.
#
# INPUT: 
# a: Data structure with CPDs, PDs and EP factors. Unconditioned.
# damping: Damping factor.
# minimize: If FALSE, it maximizes.
# no_negative_variances_nor_nands: Boolean flag that control robustness.
# OUTPUT:
# a: Data structure with CPDs, PDs and EP factors. Conditioned.
#############################################################################################################################
def update_full_Factors_no_test_factors(a, damping, minimize=True, no_negative_variances_nor_nands = False, no_negatives = True):
    # used to switch between minimizing and maximizing
    sgn = -1.0 if minimize else 1.0

    # We update the h factors
    all_tasks = a['objs']
    all_constraints = a['cons']
    n_obs = a['n_obs']
    n_pset = a['n_pset']
    n_test = a['n_test']
    n_total = a['n_total']
    q = a['q']
    c = a['c']

    alpha = np.zeros(a['q'])
    s = np.zeros(a['q'])
    ratio_cons = np.zeros(c)
    
    # First we update the factors corresponding to the observed data

    # We compute an "old" distribution 

    # Data structures for objective npset nobs cavities (a, b).
    m_pset = np.zeros((q, n_pset, n_obs))
    m_obs = np.zeros((q, n_pset, n_obs))
    v_pset = np.zeros((q, n_pset, n_obs))
    v_obs = np.zeros((q, n_pset, n_obs))
    v_cov = np.zeros((q, n_pset, n_obs))

    # Data structures for constraint npset nobs cavities (c_a, c_b).

    c_m = np.zeros((c, n_pset, n_obs))
    c_v = np.zeros((c, n_pset, n_obs))

    # We do the update of the external factors.            

    c_m_external = np.zeros((c, n_pset))
    c_v_external = np.zeros((c, n_pset))
    alpha_external = np.zeros(c)
    
    # First we update the external (easy) factors that only affect the constraints

    n_task = 0
    for cons in all_constraints:
        c_m_external[ n_task, : ] = a['m_cons'][ cons ][ n_obs : n_obs + n_pset ]
        c_v_external[ n_task, : ] = np.diag(a['V_cons'][ cons ])[ n_obs : n_obs + n_pset ]
        n_task += 1

    c_vTilde_external = a['ehfhat'][ :, : ].T
    c_mTilde_external = a['fhfhat'][ :, : ].T

    c_vOld_external = 1.0 / (1.0 / c_v_external - c_vTilde_external)
    c_mOld_external = c_vOld_external * (c_m_external / c_v_external - c_mTilde_external)

    if np.any(c_vOld_external < 0):
        raise npla.linalg.LinAlgError("Negative value in the sqrt!")
    
    alpha_external = c_mOld_external / np.sqrt(c_vOld_external)
    logZ = logcdf_robust(alpha_external)
    
    ratio = np.exp(sps.norm.logpdf(alpha_external) - logZ)
    dlogZdmfOld_external = ratio / np.sqrt(c_vOld_external) 
    dlogZdmfOld2_external = - ratio * (alpha_external + ratio) / c_vOld_external
    
     # We find the parameters of the updated factors
		
    ehfhat_act = (- dlogZdmfOld2_external / (1.0 + dlogZdmfOld2_external * c_vOld_external)).T
    fhfhat_act = ((dlogZdmfOld_external - c_mOld_external * dlogZdmfOld2_external) / (1.0 + dlogZdmfOld2_external * c_vOld_external)).T
    
    if no_negative_variances_nor_nands == True:
        neg = np.where(ehfhat_act < 0)

        ehfhat_act[ neg ] = 0.0
        fhfhat_act[ neg ] = 0.0
 
    a['ehfhat'][ :, : ] = ehfhat_act * damping + (1 - damping) * a['ehfhat'][ :, : ] 
    a['fhfhat'][ :, : ] = fhfhat_act * damping + (1 - damping) * a['fhfhat'][ :, : ]

    # Done!
  
    n_task = 0 
    for obj in all_tasks:
        m_obs[ n_task, :, : ] = np.tile(a['m'][ obj ][ 0 : n_obs ], n_pset).reshape((n_pset, n_obs))
        m_pset[ n_task, :, : ] = np.tile(a['m'][ obj ][ n_obs : n_obs + n_pset ], n_obs).reshape((n_obs, n_pset)).T
        v_cov[ n_task, :, : ] = a['V'][ obj ][ n_obs : n_obs + n_pset, 0 : n_obs ]
        v_obs[ n_task, :, : ] = np.tile(np.diag(a['V'][ obj ])[ 0 : n_obs ], n_pset).reshape((n_pset, n_obs))
        v_pset[ n_task, :, : ] = np.tile(np.diag(a['V'][ obj ])[ n_obs : n_obs + n_pset ], n_obs).reshape((n_obs, n_pset)).T
        n_task += 1
        
    n_task = 0
    
    for cons in all_constraints:
        c_m[ n_task, :, : ] = a['m_cons'][ cons ][ 0 : n_obs ]
        c_v[ n_task, :, : ] = np.diag(a['V_cons'][ cons ])[ 0 : n_obs ]
        n_task += 1
   
    #ECGM: I think that this remains being the same. 
    vTilde_obs = a['ahfhat'][ :, :, :, 0, 0 ].T
    vTilde_pset = a['ahfhat'][ :, :, :, 1, 1 ].T
    covTilde = a['ahfhat'][ :, :, :, 0, 1 ].T
    mTilde_obs = a['bhfhat'][ :, :, :, 0, ].T
    mTilde_pset = a['bhfhat'][ :, :, :, 1, ].T
    
    c_vTilde = a['a_c_hfhat'][ :, :, : ].T
    c_mTilde = a['b_c_hfhat'][ :, :, : ].T

    # Obtaining cavities.

    inv_v_obs, inv_v_pset, inv_v_cov = two_by_two_symmetric_matrix_inverse(v_obs, v_pset, v_cov)
    inv_c_v = 1.0 / c_v
    
    inv_vOld_obs = inv_v_obs - vTilde_obs
    inv_vOld_pset = inv_v_pset - vTilde_pset
    inv_vOld_cov =  inv_v_cov - covTilde
    inv_c_vOld = inv_c_v - c_vTilde

    vOld_obs, vOld_pset, vOld_cov = two_by_two_symmetric_matrix_inverse(inv_vOld_obs, inv_vOld_pset, inv_vOld_cov)
    c_vOld = 1.0 / inv_c_vOld

    mOld_obs, mOld_pset  = two_by_two_symmetric_matrix_product_vector(inv_v_obs, inv_v_pset, inv_v_cov, m_obs, m_pset)
    mOld_obs = mOld_obs - mTilde_obs
    mOld_pset = mOld_pset - mTilde_pset
    mOld_obs, mOld_pset  = two_by_two_symmetric_matrix_product_vector(vOld_obs, vOld_pset, vOld_cov, mOld_obs, mOld_pset)
    
    c_mOld = c_vOld * (c_m / c_v - c_mTilde)
                
    #Computing factors.

    s = vOld_pset + vOld_obs - 2 * vOld_cov
    s_cons = c_vOld
    
    if np.any(vOld_pset < 0):
        raise npla.linalg.LinAlgError("Negative variance in the sqrt!")
        
    if np.any(vOld_obs < 0):
        raise npla.linalg.LinAlgError("Negative variance in the sqrt!")
 
    if np.any(c_vOld < 0):
        raise npla.linalg.LinAlgError("Negative value in the sqrt!")

    alpha_cons = c_mOld / np.sqrt(c_vOld)

    scale = 1.0 - 1e-4
    while np.any(s / (vOld_pset + vOld_obs) < 1e-6):
        scale = scale**2
        s = vOld_pset + vOld_obs - 2 * vOld_cov * scale

    alpha = (mOld_obs - mOld_pset) / np.sqrt(s) * sgn

    log_phi = logcdf_robust(alpha)
    log_phi_cons = logcdf_robust(alpha_cons)

    logZ_orig = log_1_minus_exp_x(np.sum(log_phi, axis = 0))

    if np.any(np.logical_not(logZ_orig == -np.inf)):
        sel = (logZ_orig == -np.inf)
        logZ_orig[ sel ] = logcdf_robust(-np.min(alpha[ :, sel ], axis = 0))

    logZ_term1 = np.sum(log_phi_cons, axis = 0) + logZ_orig 
    logZ_term2 = log_1_minus_exp_x(np.sum(log_phi_cons, axis = 0))

    if np.any(np.logical_not(logZ_term2 == -np.inf)):
        sel = (logZ_term2 == -np.inf)
        logZ_term2[ sel ] = logcdf_robust(-np.min(alpha_cons[ :, sel ], axis = 0))

    # TODO: This should be done robustly

    max_value = np.maximum(logZ_term1, logZ_term2)
    
    logZ = np.tile(np.log(np.exp(logZ_term1 - max_value) + np.exp(logZ_term2 - max_value)) + \
        max_value, q).reshape((n_pset, q, n_obs)).swapaxes(0, 1)
    logZ_cons = np.tile(np.log(np.exp(logZ_term1 - max_value) + np.exp(logZ_term2 - max_value)) + \
        max_value, c).reshape((n_pset, c, n_obs)).swapaxes(0, 1)

    log_phi_sum = np.tile(np.sum(log_phi, axis = 0), q).reshape((n_pset, q, n_obs)).swapaxes(0, 1)
    log_phi_sum_cons = np.tile(np.sum(log_phi_cons, axis = 0), q).reshape((n_pset, q, n_obs)).swapaxes(0, 1)

    ratio = - np.exp(sps.norm.logpdf(alpha) - logZ + log_phi_sum - logcdf_robust(alpha) + log_phi_sum_cons)

    logZ_orig_cons = np.tile(logZ_orig, c).reshape((n_pset, c, n_obs)).swapaxes(0, 1)
    log_phi_sum_cons = np.tile(np.sum(log_phi_cons, axis = 0), c).reshape((n_pset, c, n_obs)).swapaxes(0, 1)

    ratio_cons = np.exp(sps.norm.logpdf(alpha_cons) - logZ_cons + logZ_orig_cons + log_phi_sum_cons - logcdf_robust(alpha_cons)) - \
        np.exp(sps.norm.logpdf(alpha_cons) - logZ_cons + log_phi_sum_cons - logcdf_robust(alpha_cons))
    
    # Derivatives, non vector form.

    dlogZdmfOld_obs = ratio / np.sqrt(s) * sgn
    dlogZdmfOld_pset = ratio / np.sqrt(s) * -1.0 * sgn

    dlogZdmfOld_obs2 = - ratio / s * (alpha + ratio) 
    dlogZdmfOld_pset2 = - ratio / s * (alpha + ratio) 
    dlogZdmfOld_cov2 = - ratio / s * (alpha + ratio) * -1.0
	
    dlogZdmcOld = ratio_cons / np.sqrt(s_cons)
    dlogZdmcOld2 = - ratio_cons / s_cons * (alpha_cons + ratio_cons)

    a_VfOld_times_dlogZdmfOld2 = vOld_obs * dlogZdmfOld_obs2 + vOld_cov * dlogZdmfOld_cov2 + 1.0
    b_VfOld_times_dlogZdmfOld2 = vOld_obs * dlogZdmfOld_cov2 + vOld_cov * dlogZdmfOld_pset2 
    c_VfOld_times_dlogZdmfOld2 = vOld_cov * dlogZdmfOld_obs2 + vOld_pset * dlogZdmfOld_cov2 
    d_VfOld_times_dlogZdmfOld2 = vOld_cov * dlogZdmfOld_cov2 + vOld_pset * dlogZdmfOld_pset2 + 1.0 

    a_inv, b_inv, c_inv, d_inv = two_by_two_matrix_inverse(a_VfOld_times_dlogZdmfOld2, b_VfOld_times_dlogZdmfOld2, \
        c_VfOld_times_dlogZdmfOld2, d_VfOld_times_dlogZdmfOld2)

    vTilde_obs_new = - (dlogZdmfOld_obs2 * a_inv + dlogZdmfOld_cov2 * c_inv)
    vTilde_pset_new = - (dlogZdmfOld_cov2 * b_inv + dlogZdmfOld_pset2 * d_inv)
    vTilde_cov_new = - (dlogZdmfOld_obs2 * b_inv + dlogZdmfOld_cov2 * d_inv)

    v_1, v_2 = two_by_two_symmetric_matrix_product_vector(dlogZdmfOld_obs2, \
        dlogZdmfOld_pset2, dlogZdmfOld_cov2, mOld_obs, mOld_pset)

    v_1 = dlogZdmfOld_obs - v_1 
    v_2 = dlogZdmfOld_pset - v_2 
    mTilde_obs_new = v_1 * a_inv + v_2 * c_inv
    mTilde_pset_new = v_1 * b_inv + v_2 * d_inv

    vTilde_cons =  - dlogZdmcOld2 / (1.0 + dlogZdmcOld2 * c_vOld)
    mTilde_cons = (dlogZdmcOld - c_mOld * dlogZdmcOld2) / (1.0 + dlogZdmcOld2 * c_vOld)

    if no_negative_variances_nor_nands == True:

        finite = np.logical_and(np.logical_and(np.logical_and(np.logical_and(np.isfinite(vTilde_obs_new), np.isfinite(vTilde_pset_new)), \
            np.isfinite(vTilde_cov_new)), np.isfinite(mTilde_obs_new)), np.isfinite(mTilde_pset_new))
        
        c_finite = np.logical_and(np.isfinite(vTilde_cons), np.isfinite(mTilde_cons))

        neg1 = np.where(np.logical_or(np.logical_not(finite), vTilde_obs_new < 0))
        neg2 = np.where(np.logical_or(np.logical_not(finite), vTilde_pset_new < 0))
        c_neg = np.where(np.logical_or(np.logical_not(c_finite), vTilde_cons < 0))

        vTilde_obs_new[ neg1 ] = 0.0
        vTilde_obs_new[ neg2 ] = 0.0
        vTilde_pset_new[ neg1 ] = 0.0
        vTilde_pset_new[ neg2 ] = 0.0
        vTilde_cov_new[ neg1 ] = 0.0
        vTilde_cov_new[ neg2 ] = 0.0
        mTilde_obs_new[ neg1 ] = 0.0
        mTilde_obs_new[ neg2 ] = 0.0
        mTilde_pset_new[ neg1 ] = 0.0
        mTilde_pset_new[ neg2 ] = 0.0
        vTilde_cons[ c_neg ] = 0.0
        mTilde_cons[ c_neg ] = 0.0

    # We do the actual update

    a_c_hfHatNew = vTilde_cons
    b_c_hfHatNew = mTilde_cons

    a['ahfhat'][ :, :, :, 0, 0 ] = vTilde_obs_new.T * damping + (1 - damping) * a['ahfhat'][ :, :, :, 0, 0 ] 
    a['ahfhat'][ :, :, :, 1, 1 ] = vTilde_pset_new.T * damping + (1 - damping) * a['ahfhat'][ :, :, :, 1, 1 ] 
    a['ahfhat'][ :, :, :, 0, 1 ] = vTilde_cov_new.T * damping + (1 - damping) * a['ahfhat'][ :, :, :, 0, 1 ] 
    a['ahfhat'][ :, :, :, 1, 0 ] = vTilde_cov_new.T * damping + (1 - damping) * a['ahfhat'][ :, :, :, 1, 0 ] 
    a['bhfhat'][ :, :, :, 0 ] = mTilde_obs_new.T * damping + (1 - damping) * a['bhfhat'][ :, :, :, 0 ] 
    a['bhfhat'][ :, :, :, 1 ] = mTilde_pset_new.T * damping + (1 - damping) * a['bhfhat'][ :, :, :, 1 ]
    a['a_c_hfhat'][ :, :, : ] = a_c_hfHatNew.T * damping + (1 - damping) * a['a_c_hfhat'][ :, :, : ] 
    a['b_c_hfhat'][ :, :, : ] = b_c_hfHatNew.T * damping + (1 - damping) * a['b_c_hfhat'][ :, :, : ]
    
    # Second we update the factors corresponding to the pareto set

    # We compute an "old" distribution 

    m_pset1 = np.zeros((q, n_pset, n_pset))
    m_pset2 = np.zeros((q, n_pset, n_pset))
    v_pset1 = np.zeros((q, n_pset, n_pset))
    v_pset2 = np.zeros((q, n_pset, n_pset))
    v_cov = np.zeros((q, n_pset, n_pset))
    
    c_m_pset = np.zeros((c, n_pset, n_pset))
    c_v_pset = np.zeros((c, n_pset, n_pset))

    #Changes done as now the observations lie first and pareto set points second.
    n_task = 0
    for obj in all_tasks:
        m_pset1[ n_task, :, : ] = np.tile(a['m'][ obj ][ n_obs : n_obs + n_pset ], n_pset).reshape((n_pset, n_pset))
        m_pset2[ n_task, :, : ] = np.tile(a['m'][ obj ][ n_obs : n_obs + n_pset ], n_pset).reshape((n_pset, n_pset)).T
        v_cov[ n_task, :, : ] = a['V'][ obj ][ n_obs : n_obs + n_pset, n_obs : n_obs + n_pset ]
        v_cov[ n_task, :, : ] = v_cov[ n_task, :, : ] - np.diag(np.diag(v_cov[ n_task, :, : ])) 
        v_pset1[ n_task, :, : ] = np.tile(np.diag(a['V'][ obj ])[ n_obs : n_obs + n_pset ], n_pset).reshape((n_pset, n_pset))
        v_pset2[ n_task, :, : ] = np.tile(np.diag(a['V'][ obj ])[ n_obs : n_obs + n_pset ], n_pset).reshape((n_pset, n_pset)).T
        n_task += 1

    n_task = 0
    for cons in all_constraints:
        c_m_pset[ n_task, :, : ] = a['m_cons'][ cons ][ n_obs : n_obs + n_pset ]
        c_v_pset[ n_task, :, : ] = np.diag(a['V_cons'][ cons ])[n_obs : n_obs + n_pset]
        n_task += 1
    
    vTilde_pset1 = a['chfhat'][ :, :, :, 0, 0 ].T
    vTilde_pset2 = a['chfhat'][ :, :, :, 1, 1 ].T
    covTilde = a['chfhat'][ :, :, :, 0, 1 ].T
    mTilde_pset1 = a['dhfhat'][ :, :, :, 0 ].T
    mTilde_pset2 = a['dhfhat'][ :, :, :, 1 ].T
    
    c_vTilde_pset = a['c_c_hfhat'][ :, :, : ].T
    c_mTilde_pset = a['d_c_hfhat'][ :, :, : ].T

    inv_v_pset1, inv_v_pset2, inv_v_cov = two_by_two_symmetric_matrix_inverse(v_pset1, v_pset2, v_cov)

    inv_vOld_pset1 = inv_v_pset1 - vTilde_pset1
    inv_vOld_pset2 = inv_v_pset2 - vTilde_pset2
    inv_vOld_cov =  inv_v_cov - covTilde

    vOld_pset1, vOld_pset2, vOld_cov = two_by_two_symmetric_matrix_inverse(inv_vOld_pset1, inv_vOld_pset2, inv_vOld_cov)

    mOld_pset1, mOld_pset2  = two_by_two_symmetric_matrix_product_vector(inv_v_pset1, inv_v_pset2, inv_v_cov, m_pset1, m_pset2)
    mOld_pset1 = mOld_pset1 - mTilde_pset1
    mOld_pset2 = mOld_pset2 - mTilde_pset2
    mOld_pset1, mOld_pset2  = two_by_two_symmetric_matrix_product_vector(vOld_pset1, vOld_pset2, vOld_cov, mOld_pset1, mOld_pset2)

    s = vOld_pset1 + vOld_pset2 - 2 * vOld_cov
	
    if np.any(vOld_pset1  < 0):
        raise npla.linalg.LinAlgError("Negative value in the sqrt!")

    if np.any(vOld_pset2  < 0):
        raise npla.linalg.LinAlgError("Negative value in the sqrt!")
        
    scale = 1.0 - 1e-4
    while np.any(s / (vOld_pset1 + vOld_pset2) < 1e-6):
        print "Reducing scale to guarantee positivity!"
        scale = scale**2
        s = vOld_pset1 + vOld_pset2 - 2 * vOld_cov * scale

    alpha = (mOld_pset1 - mOld_pset2) / np.sqrt(s) * sgn    
    
    log_phi = logcdf_robust(alpha)
    logZ = log_1_minus_exp_x(np.sum(log_phi, axis = 0))

    if np.any(np.logical_not(logZ == -np.inf)):
        sel = (logZ == -np.inf)
        logZ[ sel ] = logcdf_robust(-np.min(alpha[ :, sel ], axis = 0))

    log_phi_sum = np.tile(np.sum(log_phi, axis = 0), q).reshape((n_pset, q, n_pset)).swapaxes(0, 1)

    ratio = - np.exp(sps.norm.logpdf(alpha) - logZ + log_phi_sum - logcdf_robust(alpha))

    # Derivatives, non vector form.
    
    dlogZdmfOld_pset1 = ratio / np.sqrt(s) * sgn
    dlogZdmfOld_pset2 = ratio / np.sqrt(s) * -1.0 * sgn

    dlogZdmfOld_pset12 = - ratio / s * (alpha + ratio) 
    dlogZdmfOld_pset22 = - ratio / s * (alpha + ratio) 
    dlogZdmfOld_cov2 = - ratio / s * (alpha + ratio) * -1.0
	
    a_VfOld_times_dlogZdmfOld2 = vOld_pset1 * dlogZdmfOld_pset12 + vOld_cov * dlogZdmfOld_cov2 + 1.0
    b_VfOld_times_dlogZdmfOld2 = vOld_pset1 * dlogZdmfOld_cov2 + vOld_cov * dlogZdmfOld_pset22 
    c_VfOld_times_dlogZdmfOld2 = vOld_cov * dlogZdmfOld_pset12 + vOld_pset2 * dlogZdmfOld_cov2 
    d_VfOld_times_dlogZdmfOld2 = vOld_cov * dlogZdmfOld_cov2 + vOld_pset2 * dlogZdmfOld_pset22 + 1.0 

    a_inv, b_inv, c_inv, d_inv = two_by_two_matrix_inverse(a_VfOld_times_dlogZdmfOld2, b_VfOld_times_dlogZdmfOld2, \
        c_VfOld_times_dlogZdmfOld2, d_VfOld_times_dlogZdmfOld2)

    vTilde_pset1_new = - (dlogZdmfOld_pset12 * a_inv + dlogZdmfOld_cov2 * c_inv)
    vTilde_pset2_new = - (dlogZdmfOld_cov2 * b_inv + dlogZdmfOld_pset22 * d_inv)
    vTilde_cov_new = - (dlogZdmfOld_pset12 * b_inv + dlogZdmfOld_cov2 * d_inv)

    v_1, v_2 = two_by_two_symmetric_matrix_product_vector(dlogZdmfOld_pset12, \
        dlogZdmfOld_pset22, dlogZdmfOld_cov2, mOld_pset1, mOld_pset2)

    v_1 = dlogZdmfOld_pset1 - v_1 
    v_2 = dlogZdmfOld_pset2 - v_2 
    mTilde_pset1_new = v_1 * a_inv + v_2 * c_inv
    mTilde_pset2_new = v_1 * b_inv + v_2 * d_inv

    n_task = 0
    for obj in all_tasks:
        vTilde_pset1_new[ n_task, :, : ] -= np.diag(np.diag(vTilde_pset1_new[ n_task, :, :, ]))
        vTilde_pset2_new[ n_task, :, : ] -= np.diag(np.diag(vTilde_pset2_new[ n_task, :, :, ]))
        vTilde_cov_new[ n_task, :, : ] -= np.diag(np.diag(vTilde_cov_new[ n_task, :, : ]))
        mTilde_pset1_new[ n_task, :, : ] -= np.diag(np.diag(mTilde_pset1_new[ n_task, :, : ]))
        mTilde_pset2_new[ n_task, :, : ] -= np.diag(np.diag(mTilde_pset2_new[ n_task, :, : ]))
        n_task += 1

    if no_negative_variances_nor_nands == True:
        finite = np.logical_and(np.logical_and(np.logical_and(np.logical_and(np.isfinite(vTilde_pset1_new), np.isfinite(vTilde_pset2_new)), \
            np.isfinite(vTilde_cov_new)), np.isfinite(mTilde_pset1_new)), np.isfinite(mTilde_pset2_new))
        
        neg1 = np.where(np.logical_or(np.logical_not(finite), vTilde_pset1_new < 0))
        neg2 = np.where(np.logical_or(np.logical_not(finite), vTilde_pset2_new < 0))

        vTilde_pset1_new[ neg1 ] = 0.0
        vTilde_pset1_new[ neg2 ] = 0.0
        vTilde_pset2_new[ neg1 ] = 0.0
        vTilde_pset2_new[ neg2 ] = 0.0
        vTilde_cov_new[ neg1 ] = 0.0
        vTilde_cov_new[ neg2 ] = 0.0
        mTilde_pset1_new[ neg1 ] = 0.0
        mTilde_pset1_new[ neg2 ] = 0.0
        mTilde_pset2_new[ neg1 ] = 0.0
        mTilde_pset2_new[ neg2 ] = 0.0

    # We do the actual update

    #i not equal to j condition.

    a['chfhat'][ :, :, :, 0, 0 ] = vTilde_pset1_new.T * damping + (1 - damping) * a['chfhat'][ :, :, :, 0, 0 ] 
    a['chfhat'][ :, :, :, 1, 1 ] = vTilde_pset2_new.T * damping + (1 - damping) * a['chfhat'][ :, :, :, 1, 1 ] 
    a['chfhat'][ :, :, :, 0, 1 ] = vTilde_cov_new.T * damping + (1 - damping) * a['chfhat'][ :, :, :, 0, 1 ] 
    a['chfhat'][ :, :, :, 1, 0 ] = vTilde_cov_new.T * damping + (1 - damping) * a['chfhat'][ :, :, :, 1, 0 ] 
    a['dhfhat'][ :, :, :, 0 ] = mTilde_pset1_new.T * damping + (1 - damping) * a['dhfhat'][ :, :, :, 0 ] 
    a['dhfhat'][ :, :, :, 1 ] = mTilde_pset2_new.T * damping + (1 - damping) * a['dhfhat'][ :, :, :, 1 ]

    return a

#############################################################################################################################
# updateFactors_fast_daniel_no_constraints_pareto_set_robust: Computes EP approximations for factors that do not depend on the candidate/s.
#
# INPUT: 
# a: Data structure with CPDs, PDs and EP factors. Unconditioned.
# damping: Damping factor.
# minimize: If FALSE, it maximizes.
# no_negative_variances_nor_nands: Boolean flag that control robustness.
# OUTPUT:
# a: Data structure with CPDs, PDs and EP factors. Conditioned.
#############################################################################################################################
def update_full_Factors_only_test_factors(a, damping, minimize=True, no_negative_variances_nor_nands = False, no_negatives = True):
    # used to switch between minimizing and maximizing
    sgn = -1.0 if minimize else 1.0

    # We update the h factors
    all_tasks = a['objs']
    all_constraints = a['cons']
    n_obs = a['n_obs']
    n_pset = a['n_pset']
    n_test = a['n_test']
    n_total = a['n_total']
    q = a['q']
    c = a['c']

    alpha = np.zeros(a['q'])
    s = np.zeros(a['q'])
    ratio_cons = np.zeros(c)
    
    # First we update the factors corresponding to the observed data

    # We compute an "old" distribution 

    # Data structures for objective npset ntest cavities (a, b).

    m_pset = np.zeros((q, n_pset, n_test))
    m_test = np.zeros((q, n_pset, n_test))
    v_pset = np.zeros((q, n_pset, n_test))
    v_test = np.zeros((q, n_pset, n_test))
    v_cov = np.zeros((q, n_pset, n_test))

    # Data structures for constraint npset nobs cavities (c_a, c_b).

    c_m = np.zeros((c, n_pset, n_test))
    c_v = np.zeros((c, n_pset, n_test))
    
    # Update marginals: a['m'] , a['V']
    n_task = 0 
    for obj in all_tasks:
        m_test[ n_task, :, : ] = np.tile(a['m'][ obj ][ n_obs + n_pset : n_total ], n_pset).reshape((n_pset, n_test))
        m_pset[ n_task, :, : ] = np.tile(a['m'][ obj ][ n_obs : n_obs + n_pset ], n_test).reshape((n_test, n_pset)).T
        v_cov[ n_task, :, : ] = a['V'][ obj ][ n_obs : n_obs + n_pset, n_obs + n_pset : n_total ]
        v_test[ n_task, :, : ] = np.tile(np.diag(a['V'][ obj ])[ n_obs + n_pset : n_total ], n_pset).reshape((n_pset, n_test))
        v_pset[ n_task, :, : ] = np.tile(np.diag(a['V'][ obj ])[ n_obs : n_obs + n_pset ], n_test).reshape((n_test, n_pset)).T
        n_task += 1
        
    n_task = 0
    
    for cons in all_constraints:
        c_m[ n_task, :, : ] = a['m_cons'][ cons ][ n_obs + n_pset : n_total ]
        c_v[ n_task, :, : ] = np.diag(a['V_cons'][ cons ])[ n_obs + n_pset : n_total ]
        n_task += 1
   
    vTilde_test = a['ghfhat'][ :, :, :, 0, 0 ].T
    vTilde_pset = a['ghfhat'][ :, :, :, 1, 1 ].T
    vTilde_cov = a['ghfhat'][ :, :, :, 0, 1 ].T
    mTilde_test = a['hhfhat'][ :, :, :, 0 ].T
    mTilde_pset = a['hhfhat'][ :, :, :, 1 ].T

    vTilde_test_cons = a['g_c_hfhat'][:, :, :].T
    mTilde_test_cons = a['h_c_hfhat'][:, :, :].T

    # Obtaining cavities.

    inv_v_test, inv_v_pset, inv_v_cov = two_by_two_symmetric_matrix_inverse(v_test, v_pset, v_cov)
    inv_c_v = 1.0 / c_v
    
    inv_vOld_test = inv_v_test - vTilde_test
    inv_vOld_pset = inv_v_pset - vTilde_pset
    inv_vOld_cov =  inv_v_cov - vTilde_cov
    inv_c_vOld = inv_c_v - vTilde_test_cons

    vOld_test, vOld_pset, vOld_cov = two_by_two_symmetric_matrix_inverse(inv_vOld_test, inv_vOld_pset, inv_vOld_cov)
    c_vOld = 1.0 / inv_c_vOld

    mOld_test, mOld_pset = two_by_two_symmetric_matrix_product_vector(inv_v_test, inv_v_pset, inv_v_cov, m_test, m_pset)
    mOld_test = mOld_test - mTilde_test
    mOld_pset = mOld_pset - mTilde_pset
    mOld_test, mOld_pset  = two_by_two_symmetric_matrix_product_vector(vOld_test, vOld_pset, vOld_cov, mOld_test, mOld_pset)
    
    c_mOld = c_vOld * (c_m / c_v - mTilde_test_cons)
                
    # Computing factors.

    s = vOld_pset + vOld_test - 2 * vOld_cov
    s_cons = c_vOld
    
    if np.any(vOld_pset < 0):
        raise npla.linalg.LinAlgError("Negative variance in the sqrt!")
        
    if np.any(vOld_test < 0):
        raise npla.linalg.LinAlgError("Negative variance in the sqrt!")
 
    if np.any(c_vOld < 0):
        raise npla.linalg.LinAlgError("Negative value in the sqrt!")

    alpha_cons = c_mOld / np.sqrt(c_vOld)

    scale = 1.0 - 1e-4
    while np.any(s / (vOld_pset + vOld_test) < 1e-6):
        scale = scale**2
        s = vOld_pset + vOld_test - 2 * vOld_cov * scale

    alpha = (mOld_test - mOld_pset) / np.sqrt(s) * sgn

    log_phi = logcdf_robust(alpha)
    log_phi_cons = logcdf_robust(alpha_cons)

    logZ_orig = log_1_minus_exp_x(np.sum(log_phi, axis = 0))

    if np.any(np.logical_not(logZ_orig == -np.inf)):
        sel = (logZ_orig == -np.inf)
        logZ_orig[ sel ] = logcdf_robust(-np.min(alpha[ :, sel ], axis = 0))

    logZ_term1 = np.sum(log_phi_cons, axis = 0) + logZ_orig 
    logZ_term2 = log_1_minus_exp_x(np.sum(log_phi_cons, axis = 0))

    if np.any(np.logical_not(logZ_term2 == -np.inf)):
        sel = (logZ_term2 == -np.inf)
        logZ_term2[ sel ] = logcdf_robust(-np.min(alpha_cons[ :, sel ], axis = 0))

    # TODO: This should be done robustly

    max_value = np.maximum(logZ_term1, logZ_term2)
    
    logZ = np.tile(np.log(np.exp(logZ_term1 - max_value) + np.exp(logZ_term2 - max_value)) + \
        max_value, q).reshape((n_pset, q, n_test)).swapaxes(0, 1)
    logZ_cons = np.tile(np.log(np.exp(logZ_term1 - max_value) + np.exp(logZ_term2 - max_value)) + \
        max_value, c).reshape((n_pset, c, n_test)).swapaxes(0, 1)

    log_phi_sum = np.tile(np.sum(log_phi, axis = 0), q).reshape((n_pset, q, n_test)).swapaxes(0, 1)
    log_phi_sum_cons = np.tile(np.sum(log_phi_cons, axis = 0), q).reshape((n_pset, q, n_test)).swapaxes(0, 1)

    ratio = - np.exp(sps.norm.logpdf(alpha) - logZ + log_phi_sum - logcdf_robust(alpha) + log_phi_sum_cons)

    logZ_orig_cons = np.tile(logZ_orig, c).reshape((n_pset, c, n_test)).swapaxes(0, 1)
    log_phi_sum_cons = np.tile(np.sum(log_phi_cons, axis = 0), c).reshape((n_pset, c, n_test)).swapaxes(0, 1)

    ratio_cons = np.exp(sps.norm.logpdf(alpha_cons) - logZ_cons + logZ_orig_cons + log_phi_sum_cons - logcdf_robust(alpha_cons)) - \
        np.exp(sps.norm.logpdf(alpha_cons) - logZ_cons + log_phi_sum_cons - logcdf_robust(alpha_cons))
    
    # Derivatives, non vector form.

    dlogZdmfOld_test = ratio / np.sqrt(s) * sgn
    dlogZdmfOld_pset = ratio / np.sqrt(s) * -1.0 * sgn

    dlogZdmfOld_test2 = - ratio / s * (alpha + ratio) 
    dlogZdmfOld_pset2 = - ratio / s * (alpha + ratio) 
    dlogZdmfOld_cov2 = - ratio / s * (alpha + ratio) * -1.0
	
    dlogZdmcOld = ratio_cons / np.sqrt(s_cons)
    dlogZdmcOld2 = - ratio_cons / s_cons * (alpha_cons + ratio_cons)

    a_VfOld_times_dlogZdmfOld2 = vOld_test * dlogZdmfOld_test2 + vOld_cov * dlogZdmfOld_cov2 + 1.0
    b_VfOld_times_dlogZdmfOld2 = vOld_test * dlogZdmfOld_cov2 + vOld_cov * dlogZdmfOld_pset2 
    c_VfOld_times_dlogZdmfOld2 = vOld_cov * dlogZdmfOld_test2 + vOld_pset * dlogZdmfOld_cov2 
    d_VfOld_times_dlogZdmfOld2 = vOld_cov * dlogZdmfOld_cov2 + vOld_pset * dlogZdmfOld_pset2 + 1.0 

    a_inv, b_inv, c_inv, d_inv = two_by_two_matrix_inverse(a_VfOld_times_dlogZdmfOld2, b_VfOld_times_dlogZdmfOld2, \
        c_VfOld_times_dlogZdmfOld2, d_VfOld_times_dlogZdmfOld2)

    vTilde_test_new = - (dlogZdmfOld_test2 * a_inv + dlogZdmfOld_cov2 * c_inv)
    vTilde_pset_new = - (dlogZdmfOld_cov2 * b_inv + dlogZdmfOld_pset2 * d_inv)
    vTilde_cov_new = - (dlogZdmfOld_test2 * b_inv + dlogZdmfOld_cov2 * d_inv)

    v_1, v_2 = two_by_two_symmetric_matrix_product_vector(dlogZdmfOld_test2, \
        dlogZdmfOld_pset2, dlogZdmfOld_cov2, mOld_test, mOld_pset)

    v_1 = dlogZdmfOld_test - v_1 
    v_2 = dlogZdmfOld_pset - v_2 
    mTilde_test_new = v_1 * a_inv + v_2 * c_inv
    mTilde_pset_new = v_1 * b_inv + v_2 * d_inv

    vTilde_cons =  - dlogZdmcOld2 / (1.0 + dlogZdmcOld2 * c_vOld)
    mTilde_cons = (dlogZdmcOld - c_mOld * dlogZdmcOld2) / (1.0 + dlogZdmcOld2 * c_vOld)

    if no_negative_variances_nor_nands == True:

        finite = np.logical_and(np.logical_and(np.logical_and(np.logical_and(np.isfinite(vTilde_test_new), np.isfinite(vTilde_pset_new)), \
            np.isfinite(vTilde_cov_new)), np.isfinite(mTilde_test_new)), np.isfinite(mTilde_pset_new))
        
        c_finite = np.logical_and(np.isfinite(vTilde_cons), np.isfinite(mTilde_cons))

        neg1 = np.where(np.logical_or(np.logical_not(finite), vTilde_test_new < 0))
        neg2 = np.where(np.logical_or(np.logical_not(finite), vTilde_pset_new < 0))
        c_neg = np.where(np.logical_or(np.logical_not(c_finite), vTilde_cons < 0))

        vTilde_test_new[ neg1 ] = 0.0
        vTilde_test_new[ neg2 ] = 0.0
        vTilde_pset_new[ neg1 ] = 0.0
        vTilde_pset_new[ neg2 ] = 0.0
        vTilde_cov_new[ neg1 ] = 0.0
        vTilde_cov_new[ neg2 ] = 0.0
        mTilde_test_new[ neg1 ] = 0.0
        mTilde_test_new[ neg2 ] = 0.0
        mTilde_pset_new[ neg1 ] = 0.0
        mTilde_pset_new[ neg2 ] = 0.0
        vTilde_cons[ c_neg ] = 0.0
        mTilde_cons[ c_neg ] = 0.0

    # We do the actual update

    g_c_hfHatNew = vTilde_cons
    h_c_hfHatNew = mTilde_cons

    a['ghfhat'][ :, :, :, 0, 0 ] = vTilde_test_new.T * damping + (1 - damping) * a['ghfhat'][ :, :, :, 0, 0 ] 
    a['ghfhat'][ :, :, :, 1, 1 ] = vTilde_pset_new.T * damping + (1 - damping) * a['ghfhat'][ :, :, :, 1, 1 ] 
    a['ghfhat'][ :, :, :, 0, 1 ] = vTilde_cov_new.T * damping + (1 - damping) * a['ghfhat'][ :, :, :, 0, 1 ] 
    a['ghfhat'][ :, :, :, 1, 0 ] = vTilde_cov_new.T * damping + (1 - damping) * a['ghfhat'][ :, :, :, 1, 0 ] 
    a['hhfhat'][ :, :, :, 0 ] = mTilde_test_new.T * damping + (1 - damping) * a['hhfhat'][ :, :, :, 0 ] 
    a['hhfhat'][ :, :, :, 1 ] = mTilde_pset_new.T * damping + (1 - damping) * a['hhfhat'][ :, :, :, 1 ]
    a['g_c_hfhat'][ :, :, : ] = g_c_hfHatNew.T * damping + (1 - damping) * a['g_c_hfhat'][ :, :, : ] 
    a['h_c_hfhat'][ :, :, : ] = h_c_hfHatNew.T * damping + (1 - damping) * a['h_c_hfhat'][ :, :, : ]

    return a



def gp_prediction_given_chol_K(X, Xtest, chol_star, cholV, m, model, jitter):
# computes the predictive distributions. but the chol of the kernel matrix and the
# chol of the test matrix are already provided. 
    
    Kstar = model.noiseless_kernel.cross_cov(X, Xtest)
    mf = np.dot(Kstar.T, spla.cho_solve((chol_star, False), m))
    aux = spla.cho_solve((chol_star, False), Kstar)
    # vf = model.params['amp2'].value * (1.0 + jitter) - \
    #     np.sum(spla.solve_triangular(chol_star.T, Kstar, lower=True)**2, axis=0) + \
    #     np.sum(np.dot(cholV, aux)**2, axis=0)
    vf = model.params['amp2'].value - \
        np.sum(spla.solve_triangular(chol_star.T, Kstar, lower=True)**2, axis=0) + \
        np.sum(np.dot(cholV, aux)**2, axis=0) + \
        jitter

    if np.any(vf < 0.0):
        raise Exception("Encountered negative variance: %f" % np.min(vf))

    return Kstar, mf, vf

# Method that approximates the predictive distribution at a particular location.

def predictEP_multiple_iter(obj_models, a, pareto_set, Xtest, damping = 1, n_iters = 5, no_negatives = True, minimize=True):

	# used to switch between minimizing and maximizing

	sgn = -1.0 if minimize else 1.0

	objs = a['objs']
	all_tasks = objs

	n_obs = a['n_obs']
	n_pset = a['n_pset']
	n_total = a['n_total']
	n_test = Xtest.shape[ 0 ]
	q = a['q']

	Kstar = dict()
	mf = dict()
	mP = dict()
	mPset = dict()
	vf = dict()
	vP = dict()
	cov = dict()
	vPset = dict()

	# This is used for the comutation of the variance of the predictive distribution

	ahfHatNew = dict()
	bhfHatNew = dict()

	mfOld = dict()
	VfOld = dict()
	ahfHatNew = dict()
	bhfHatNew = dict()

	for obj in all_tasks:
		mfOld[ obj ] = np.zeros((n_pset, 2))
		VfOld[ obj ] = np.zeros((n_pset, 2, 2))
		ahfHatNew[ obj ] = np.zeros((n_pset, 2, 2))
		bhfHatNew[ obj ] = np.zeros((n_pset, 2))

	# First data includes the pareto set. Then, the test point

	Xtest_ext = np.vstack((a['X'][ 0 : n_pset, : ], Xtest))

	for obj in all_tasks:

		# We compute the means and variances of each point (test and pareto set)
	
		Kstar[ obj ], mP[ obj ], vP[ obj ] = gp_prediction_given_chol_K(a['X'], Xtest_ext, 
			a['cholKstarstar'][ obj ], a['cholV'][ obj ], a['m'][ obj ], all_tasks[ obj ], a['jitter'][obj])
		vf[ obj ] = vP[ obj ][ n_pset : (n_pset + n_test) ]
		vPset[ obj ] = vP[ obj ][ 0 : n_pset ]
		mPset[ obj ] = mP[ obj ][ 0 : n_pset ]
		mf[ obj ] = mP[ obj ][ n_pset : (n_pset + n_test) ]

		# Now we compute the covariances between the test data and the pareto set

		Kstarstar = obj_models[ obj ].noiseless_kernel.cross_cov(Xtest_ext[ 0 : n_pset, : ], Xtest_ext[ n_pset : (n_pset + n_test), : ])
		aux1 = spla.solve_triangular(a['cholKstarstar'][ obj ].T, Kstar[ obj ][ : , 0 : n_pset  ], lower=True) 
		aux2 = spla.solve_triangular(a['cholKstarstar'][ obj ].T, Kstar[ obj ][ :, n_pset : (n_pset + n_test) ], lower=True)
		aux11 = np.dot(a['cholV'][ obj ], spla.solve_triangular(a['cholKstarstar'][ obj ], aux1, lower=False))
		aux12 = np.dot(a['cholV'][ obj ], spla.solve_triangular(a['cholKstarstar'][ obj ], aux2, lower=False))
		cov[ obj ] = Kstarstar - np.dot(aux1.transpose(), aux2) + np.dot(aux11.transpose(), aux12)

	# scale things for stability

#	for obj in all_tasks:
#		scale = (1.0 - 1e-4) * np.ones(cov[ obj ].shape)
#		vf_tmp = np.repeat(vf[ obj ], cov[ obj ].shape[ 0 ]).reshape(cov[ obj ].shape[ ::-1 ]).transpose() 
#		vpset_tmp = np.repeat(vPset[ obj ], cov[ obj ].shape[ 1 ]).reshape(cov[ obj ].shape) 
#		index = vf_tmp + vpset_tmp -  2.0 * scale * cov[ obj ] < 1e-10
#
#		while np.any(index):
#			scale[ index ] = scale[ index ]**2
#			index = vf_tmp + vpset_tmp -  2.0 * scale * cov[ obj ] < 1e-10
#
#   		cov[ obj ] = scale * cov[ obj ]

	# We update the predictive distribution to take into account that it has to be dominated by the paretoset
	# For this we use a single parallel update of the factors

	# We set the approximate factors to be uniform

	mTilde_pset = np.zeros((q, n_pset, n_test))
	mTilde_test = np.zeros((q, n_pset, n_test))
	vTilde_pset = np.zeros((q, n_pset, n_test))
	vTilde_test = np.zeros((q, n_pset, n_test))
	vTilde_cov = np.zeros((q, n_pset, n_test))

	# We compute a "new" distribution 

	mOld_pset = np.zeros((q, n_pset, n_test))
	mOld_test = np.zeros((q, n_pset, n_test))
	vOld_pset = np.zeros((q, n_pset, n_test))
	vOld_test = np.zeros((q, n_pset, n_test))
	covOld = np.zeros((q, n_pset, n_test))

	mNew_pset = np.zeros((q, n_pset, n_test))
	mNew_test = np.zeros((q, n_pset, n_test))
	vNew_pset = np.zeros((q, n_pset, n_test))
	vNew_test = np.zeros((q, n_pset, n_test))
	vNew_cov = np.zeros((q, n_pset, n_test))
	covOrig = np.zeros((q, n_pset, n_test))

	vfNew = dict()
	mfNew = dict()

	n_task = 0
	for obj in all_tasks:
		mNew_pset[ n_task, :, : ] = np.repeat(mPset[ obj ], n_test).reshape(((n_pset, n_test)))
		mNew_test[ n_task, :, : ] = np.repeat(mf[ obj ], n_pset).reshape(((n_test, n_pset))).transpose()
		vNew_pset[ n_task, :, : ] = np.repeat(vPset[ obj ], n_test).reshape(((n_pset, n_test)))
		vNew_test[ n_task, :, : ] = np.repeat(vf[ obj ], n_pset).reshape(((n_test, n_pset))).transpose()
		vNew_cov[ n_task, :, : ] = cov[ obj ]
		covOrig[ n_task, :, : ] = cov[ obj ]
		n_task += 1

	# We compute the predictive distribution over the points in the pareto set

	vOld_full_pset = dict()

	for obj in all_tasks:
		Kstarstar = obj_models[ obj ].noiseless_kernel.cross_cov(a['X'][ 0 : n_pset, : ], a['X'][ 0 : n_pset, : ])
		Kstar = obj_models[ obj ].noiseless_kernel.cross_cov(a['X'], a['X'][ 0 : n_pset, : ])
		aux1 = spla.solve_triangular(a['cholKstarstar'][ obj ].T, Kstar, lower=True) 
		aux2 = spla.solve_triangular(a['cholKstarstar'][ obj ].T, Kstar, lower=True)
		aux11 = np.dot(a['cholV'][ obj ], spla.solve_triangular(a['cholKstarstar'][ obj ], aux1, lower=False))
		aux12 = np.dot(a['cholV'][ obj ], spla.solve_triangular(a['cholKstarstar'][ obj ], aux2, lower=False))
		vOld_full_pset[ obj ] = Kstarstar - np.dot(aux1.transpose(), aux2) + np.dot(aux11.transpose(), aux12) + \
			np.eye(n_pset) * a['jitter'][obj]

	n_task = 0
	for obj in all_tasks:
		vfNew[ obj ] = np.zeros( n_test )
		mfNew[ obj ] = np.zeros( n_test )
	
	for k in range(n_iters):

		change = 0
		
		# We compute an old distribution by substracting the approximate factors

		det = vNew_test * vNew_pset - vNew_cov * vNew_cov
		vNew_inv_test = 1.0 / det * vNew_pset
		vNew_inv_pset = 1.0 / det * vNew_test
		vNew_inv_cov = 1.0 / det * - vNew_cov
	
		vOld_inv_test = vNew_inv_test - vTilde_test
		vOld_inv_pset = vNew_inv_pset - vTilde_pset
		vOld_inv_cov = vNew_inv_cov - vTilde_cov
	
		det = vOld_inv_test * vOld_inv_pset - vOld_inv_cov * vOld_inv_cov
		vOld_test = 1.0 / det * vOld_inv_pset
		vOld_pset = 1.0 / det * vOld_inv_test
		covOld = 1.0  / det * - vOld_inv_cov
	
		m_nat_old_test = vNew_inv_test * mNew_test + vNew_inv_cov * mNew_pset - mTilde_test
		m_nat_old_pset = vNew_inv_cov * mNew_test + vNew_inv_pset * mNew_pset - mTilde_pset
	
		mOld_test = vOld_test * m_nat_old_test + covOld * m_nat_old_pset
		mOld_pset = covOld * m_nat_old_test + vOld_pset * m_nat_old_pset

		# We comupte a new distribution
	
		s = vOld_pset + vOld_test - 2 * covOld
		alpha = (mOld_test - mOld_pset) / np.sqrt(s) * sgn
	
		if np.any(s < 0):
			raise npla.linalg.LinAlgError("Negative value in the sqrt!")
	
		log_phi = logcdf_robust(alpha)
       		logZ = np.repeat(log_1_minus_exp_x(np.sum(log_phi, axis = 0)).transpose(), q).reshape((n_test, n_pset, q)).transpose()
		log_phi_sum = np.repeat(np.sum(log_phi, axis = 0).transpose(), q).reshape((n_test, n_pset, q)).transpose()
	
		ratio = - np.exp(sps.norm.logpdf(alpha) - logZ + log_phi_sum - log_phi)
	
		dlogZdmfOld_test = ratio / np.sqrt(s) * sgn
		dlogZdmfOld_pset = ratio / np.sqrt(s) * sgn * -1.0
	
		dlogZdVfOld_test = -0.5 * ratio * alpha / s 
		dlogZdVfOld_pset = -0.5 * ratio * alpha / s 
		dlogZdVfOld_cov = -0.5 * ratio * alpha / s * -1.0
	
		# The following lines compute the updates in parallel C = dmdm - 2 dv 
		# First the first natural parameter
	
		c_11 = dlogZdmfOld_test * dlogZdmfOld_test - 2 * dlogZdVfOld_test
		c_22 = dlogZdmfOld_pset * dlogZdmfOld_pset - 2 * dlogZdVfOld_pset
		c_12 = dlogZdmfOld_pset * dlogZdmfOld_test - 2 * dlogZdVfOld_cov
		
		cp_11 = c_11 * vOld_test + c_12 * covOld
		cp_12 = c_11 * covOld + c_12 * vOld_pset
		cp_21 = c_12 * vOld_test + c_22 * covOld
		cp_22 = c_12 * covOld + c_22 * vOld_pset
	
		vNew_test = vOld_test - (vOld_test * cp_11 + covOld * cp_21)
		vNew_cov = covOld - (vOld_test * cp_12 + covOld * cp_22)
		vNew_pset = vOld_pset - (covOld * cp_12 + vOld_pset * cp_22)
	
		det = vNew_test * vNew_pset - vNew_cov * vNew_cov
		vNew_inv_test = 1.0 / det * vNew_pset
		vNew_inv_pset = 1.0 / det * vNew_test
		vNew_inv_cov = 1.0 / det * - vNew_cov
	
		det = vOld_test * vOld_pset - covOld * covOld
		vOld_inv_test = 1.0 / det * vOld_pset
		vOld_inv_pset = 1.0 / det * vOld_test
		vOld_inv_cov = 1.0 / det * - covOld
	
		# This is the approx factor
	
		vTilde_test_new = (vNew_inv_test - vOld_inv_test) 
		vTilde_pset_new = (vNew_inv_pset - vOld_inv_pset) 
		vTilde_cov_new = (vNew_inv_cov - vOld_inv_cov) 

		if no_negatives:
			neg = np.where(vTilde_test_new < 0)
			vTilde_test_new[ neg ] = 0
			vTilde_pset_new[ neg ] = 0
			vTilde_cov_new[ neg ] = 0
	
		# We avoid negative variances in the approximate factors. This avoids non PSD cov matrices
	
#		neg = np.where(vTilde_test < 0)
#		vTilde_test[ neg ] = 0
#		vTilde_pset[ neg ] = 0
#		vTilde_cov[ neg ] = 0
	
		# Now the second natural parameter = A~ (mOld + Vold dlogz_dm) + dlogz_dm
	
		v_1 = mOld_test + vOld_test * dlogZdmfOld_test + covOld * dlogZdmfOld_pset
		v_2 = mOld_pset + covOld * dlogZdmfOld_test + vOld_pset * dlogZdmfOld_pset
	
		mTilde_test_new = vTilde_test_new * v_1 + vTilde_cov_new * v_2 + dlogZdmfOld_test
		mTilde_pset_new = vTilde_cov_new * v_1 + vTilde_pset_new * v_2 + dlogZdmfOld_pset
	
		# We damp the updates

#		max_change = 0
#
#		max_change = np.max((max_change, np.max(np.abs(vTilde_test_new - vTilde_test))))
#		max_change = np.max((max_change, np.max(np.abs(vTilde_pset_new - vTilde_pset))))
#		max_change = np.max((max_change, np.max(np.abs(vTilde_cov_new - vTilde_cov))))
#		max_change = np.max((max_change, np.max(np.abs(mTilde_test_new - mTilde_test))))
#		max_change = np.max((max_change, np.max(np.abs(mTilde_pset_new - mTilde_pset))))

#		print(max_change)

		vTilde_test = vTilde_test_new * damping + (1 - damping) * vTilde_test
		vTilde_pset = vTilde_pset_new * damping + (1 - damping) * vTilde_pset
		vTilde_cov = vTilde_cov_new * damping + (1 - damping) * vTilde_cov
		mTilde_test = mTilde_test_new * damping + (1 - damping) * mTilde_test
		mTilde_pset = mTilde_pset_new * damping + (1 - damping) * mTilde_pset

		
		# After computing the first natural parameter of the approximate factors we recontruct the 
		# predictive distribution. We do the actual computation of the predictive distribution
	
		# This is the most expensive part (the reconstruction of the posterior)
	
		n_task = 0
		for obj in all_tasks:
	
			A = vOld_full_pset[ obj ]
			Ainv = matrixInverse(vOld_full_pset[ obj ])
	
			for i in range(n_test):
	
				if ((i % np.ceil(n_test / 100)) == 0):
					sys.stdout.write(".")
					sys.stdout.flush()
	
				B = covOrig[ n_task, :, i ]
				C = covOrig[ n_task, :, i ].transpose()
				D = vf[ obj ][ i ]
	
				# We invert the matrix using block inversion
					
				Anew = Ainv + np.outer(np.dot(Ainv, B), np.dot(C, Ainv)) * 1.0 / (D - np.sum(C * np.dot(Ainv, B)))  
				Dnew = 1.0 / (D - np.dot(np.dot(C, Ainv), B))
				Bnew = - np.dot(Ainv, B) * Dnew
				Cnew = - 1.0 / D * np.dot(C, Anew)
	
				# We add the contribution of the approximate factors
	
				V = np.vstack((np.hstack((Anew, Bnew.reshape((n_pset, 1)))), np.append(Cnew, Dnew).reshape((1, n_pset + 1))))
				m = np.dot(V, np.append(mPset[ obj ], mf[ obj ][ i ]))

				mnew = (m + np.append(mTilde_pset[ n_task, :, i ], np.sum(mTilde_test[ n_task, :, i ]))) 
	
				Anew = (Anew + np.diag(vTilde_pset[ n_task, :, i ])) 
				Bnew = (Bnew + vTilde_cov[ n_task, :, i ]) 
				Cnew = (Cnew + vTilde_cov[ n_task, :, i ])
				Dnew = (Dnew + np.sum(vTilde_test[ n_task, : , i ])) 
	
				# We perform the computation of D by inverting the V matrix after adding the params of the approx factors
	
				Anew_inv = matrixInverse(Anew)
	
				D = 1.0 / (Dnew - np.sum(Bnew * np.dot(Anew_inv, Cnew)))
				aux = np.outer(np.dot(Anew_inv, Bnew), np.dot(Cnew, Anew_inv))
				A = Anew_inv +  aux * 1.0 / (Dnew - np.sum(Cnew * np.dot(Anew_inv, Bnew)))  
				B = - np.dot(Anew_inv, Bnew) * D
				C = - 1.0 / Dnew * np.dot(Cnew, A)
	
				V = np.vstack((np.hstack((A, B.reshape((n_pset, 1)))), np.append(C, D).reshape((1, n_pset + 1))))
	
				mean = np.dot(V, mnew)
	
				mNew_pset[ n_task, : , i ] = mean[ 0 : n_pset ]
				mNew_test[ n_task, : , i ] = mean[ n_pset ]
				vNew_pset[ n_task, : , i ] = np.diag(V)[ 0 : n_pset ]
				vNew_test[ n_task, : , i ] = D
				vNew_cov[ n_task, : , i ] = V[ n_pset, 0 : n_pset ]

				change = np.max((change, np.max(np.abs(vfNew[ obj ][ i ] - D))))
				change = np.max((change, np.max(np.abs(mfNew[ obj ][ i ] - mean[ n_pset ]))))

				vfNew[ obj ][ i ] = D
				mfNew[ obj ][ i ] = mean[ n_pset ]

			n_task += 1
			print ''	

		print(change)

	for obj in all_tasks:
		if np.any(vfNew[ obj ] <= 0):
			raise Exception("vfnew is negative: %g at index %d" % (np.min(vfNew[ obj ]), np.argmin(vfNew[ obj ])))
		if np.any(np.isnan(vfNew[ obj ])):
			raise Exception("vfnew constrains nan")

	return {'mf': mfNew, 'vf':vfNew, 'mfo': mf, 'vfo': vf} 

	# don't bother computing mf and mc since they are not used in the acquisition function
	# m = mean, v = var, f = objective, c = constraint


# Method that approximates the predictive distribution at a particular location.

def predictEP_unconditioned(obj_models, con_models, a, pareto_set, Xtest):

    # used to switch between minimizing and maximizing
    objs = a['objs']
    cons = a['cons']
    all_tasks = objs
    all_constraints = cons

    n_obs = a['n_obs']
    n_pset = a['n_pset']
    n_total = a['n_total']
    n_test = Xtest.shape[ 0 ]
    q = a['q']
    c = a['c']

    Kstar = dict()
    mf = dict()
    mP = dict()
    mPset = dict()
    vf = dict()
    vP = dict()
    cov = dict()
    vPset = dict()
    mc = dict()
    vc = dict()

    # This is used for the comutation of the variance of the predictive distribution

    ahfHatNew = dict()
    bhfHatNew = dict()
    c_ahfHatNew = dict()
    c_bhfHatNew = dict()
    
    mfOld = dict()
    VfOld = dict()
    
    mcOld = dict()
    VcOld = dict()
    
    for obj in all_tasks:
        mfOld[ obj ] = np.zeros((n_pset, 2))
        VfOld[ obj ] = np.zeros((n_pset, 2, 2))
        ahfHatNew[ obj ] = np.zeros((n_pset, 2, 2))
        bhfHatNew[ obj ] = np.zeros((n_pset, 2))

    for cons in all_constraints:
        mcOld[ cons ] = np.zeros((n_pset))
        VcOld[ cons ] = np.zeros((n_pset))
        c_ahfHatNew[ cons ] = np.zeros((n_pset))
        c_bhfHatNew[ cons ] = np.zeros((n_pset))
        
    # First data includes the pareto set. Then, the test point

    Xtest_ext = np.vstack((a['X'][ 0 : n_pset, : ], Xtest))

    for obj in all_tasks:

        # We compute the means and variances of each point (test and pareto set)
	
	Kstar[ obj ], mP[ obj ], vP[ obj ] = gp_prediction_given_chol_K(a['X'], Xtest_ext, 
        	a['cholKstarstar'][ obj ], a['cholV'][ obj ], a['m'][ obj ], all_tasks[ obj ], a['jitter'][obj])
	vf[ obj ] = vP[ obj ][ n_pset : (n_pset + n_test) ]
	mf[ obj ] = mP[ obj ][ n_pset : (n_pset + n_test) ]
 
    for cons in all_constraints:

        # We compute the means and variances of each point (test and pareto set)
	
	Kstar[ cons ], mP[ cons ], vP[ cons ] = gp_prediction_given_chol_K(a['X'], Xtest_ext, 
        	a['cholKstarstar_cons'][ cons ], a['cholV_cons'][ cons ], a['m_cons'][ cons ], all_constraints[ cons ], a['jitter'][cons])
	vc[ cons ] = vP[ cons ][ n_pset : (n_pset + n_test) ]
	mc[ cons ] = mP[ cons ][ n_pset : (n_pset + n_test) ]
 
    mfNew = dict()
    vfNew = dict()

    mcNew = dict()
    vcNew = dict()
    
    for obj in all_tasks:
        vfNew[ obj ] = vf[ obj ]
        mfNew[ obj ] = mf[ obj ]

    for cons in all_constraints:
        vcNew[ cons ] = vc[ cons ]
        mcNew[ cons ] = mc[ cons ]
    
    for obj in all_tasks:
        if np.any(vfNew[ obj ] <= 0):
            raise Exception("vfnew is negative: %g at index %d" % (np.min(vfNew[ obj ]), np.argmin(vfNew[ obj ])))
        if np.any(np.isnan(vfNew[ obj ])):
            raise Exception("vfnew constrains nan")

    for cons in all_constraints:
        if np.any(vcNew[ cons ] <= 0):
            raise Exception("vcnew is negative: %g at index %d" % (np.min(vcNew[ cons ]), np.argmin(vcNew[ cons ])))
        if np.any(np.isnan(vcNew[ cons ])):
            raise Exception("vcnew constrains nan")

    return {'mf': mfNew, 'vf':vfNew, 'mfo': mf, 'vfo': vf, 'mc': mcNew, 'vc': vcNew, 'mcc': mc, 'vcc': vc} 

    # don't bother computing mf and mc since they are not used in the acquisition function
    # m = mean, v = var, f = objective, c = constraint



def compute_full_unconstrained_predictive_distribution(a, n_pset, Xtest, all_tasks, n_test, obj_models, all_constraints):
	Kstar = dict()
	mf = dict()
    	mc = dict()
    	mP = dict()
    	mP_cons = dict()
    	mPset = dict()
    	mPset_cons = dict()
    	vf = dict()
    	vc = dict()
    	vP = dict()
    	vP_cons = dict()
    	cov = dict()
    	vPset = dict()
    	vPset_cons = dict()

	# First data includes the pareto set. Then, the test point

	Xtest_ext = np.vstack((a['X'][ 0 : n_pset, : ], Xtest))

    	for obj in all_tasks:

        	# We compute the means and variances of each point (test and pareto set)

        	Kstar[ obj ], mP[ obj ], vP[ obj ] = gp_prediction_given_chol_K(a['X'], Xtest_ext,
            		a['cholKstarstar'][ obj ], a['cholV'][ obj ], a['m'][ obj ], all_tasks[ obj ], a['jitter'][obj])
        	vf[ obj ] = vP[ obj ][ n_pset : (n_pset + n_test) ]
        	vPset[ obj ] = vP[ obj ][ 0 : n_pset ]
        	mPset[ obj ] = mP[ obj ][ 0 : n_pset ]
        	mf[ obj ] = mP[ obj ][ n_pset : (n_pset + n_test) ]

        	# Now we compute the covariances between the test data and the pareto set
        	Kstarstar = obj_models[ obj ].noiseless_kernel.cross_cov(Xtest_ext[ 0 : n_pset, : ], Xtest_ext[ n_pset : (n_pset + n_test), : ])
        	aux1 = spla.solve_triangular(a['cholKstarstar'][ obj ].T, Kstar[ obj ][ : , 0 : n_pset  ], lower=True)
        	aux2 = spla.solve_triangular(a['cholKstarstar'][ obj ].T, Kstar[ obj ][ :, n_pset : (n_pset + n_test) ], lower=True)
        	aux11 = np.dot(a['cholV'][ obj ], spla.solve_triangular(a['cholKstarstar'][ obj ], aux1, lower=False))
        	aux12 = np.dot(a['cholV'][ obj ], spla.solve_triangular(a['cholKstarstar'][ obj ], aux2, lower=False))
        	cov[ obj ] = Kstarstar - np.dot(aux1.transpose(), aux2) + np.dot(aux11.transpose(), aux12)

	for cons in all_constraints:

        	# We compute the means and variances of each point (test and pareto set)

        	Kstar[ cons ], mP_cons[ cons ], vP_cons[ cons ] = gp_prediction_given_chol_K(a['X'], Xtest_ext,
            		a['cholKstarstar_cons'][ cons ], a['cholV_cons'][ cons ], a['m_cons'][ cons ], all_constraints[ cons ], a['jitter'][cons])
        	vc[ cons ] = vP_cons[ cons ][ n_pset : (n_pset + n_test) ]
        	mc[ cons ] = mP_cons[ cons ][ n_pset : (n_pset + n_test) ]

    	# scale things for stability

    	for obj in all_tasks:
        	cov[ obj ] = cov[ obj ] * 0.95

	return Kstar, mf, mc, mP, mP_cons, mPset, mPset_cons, vf, vc, vP, vP_cons, cov, vPset, vPset_cons, Xtest_ext


"""
See Miguel's paper (http://arxiv.org/pdf/1406.2541v1.pdf) section 2.1 and Appendix A

Returns a function the samples from the approximation...

if testing=True, it does not return the result but instead the random cosine for testing only

We express the kernel as an expectation. But then we approximate the expectation with a weighted sum
theta are the coefficients for this weighted sum. that is why we take the dot product of theta at the end
we also need to scale at the end so that it's an average of the random features. 

if use_woodbury_if_faster is False, it never uses the woodbury version
"""
def sample_gp_with_random_features(gp, nFeatures, testing=False, use_woodbury_if_faster=True):

    d = gp.num_dims
    N_data = gp.observed_values.size

    nu2 = gp.noise_value()

    sigma2 = gp.params['amp2'].value  # the kernel amplitude

    # We draw the random features
    if gp.options['kernel'] == "SquaredExp":
        W = npr.randn(nFeatures, d) / gp.params['ls'].value
    elif gp.options['kernel'] == "Matern52":
        m = 5.0/2.0
        W = npr.randn(nFeatures, d) / gp.params['ls'].value / np.sqrt(npr.gamma(shape=m, scale=1.0/m, size=(nFeatures,1)))
    else:
        raise Exception('This random feature sampling is for the squared exp or Matern5/2 kernels and you are using the %s' % gp.options['kernel'])
    b = npr.uniform(low=0, high=2*np.pi, size=nFeatures)[:,None]

    # Just for testing the  random features in W and b... doesn't test the weights theta

    if testing:
        return lambda x: np.sqrt(2 * sigma2 / nFeatures) * np.cos(np.dot(W, gp.noiseless_kernel.transformer.forward_pass(x).T) + b)

    randomness = npr.randn(nFeatures)

    # W has size nFeatures by d
    # tDesignMatrix has size Nfeatures by Ndata
    # woodbury has size Ndata by Ndata
    # z is a vector of length nFeatures

    if gp.has_data:
        tDesignMatrix = np.sqrt(2.0 * sigma2 / nFeatures) * np.cos(np.dot(W, \
		gp.noiseless_kernel.transformer.forward_pass(gp.observed_inputs).T) + b)

        if use_woodbury_if_faster and N_data < nFeatures:
            # you can do things in cost N^2d instead of d^3 by doing this woodbury thing

            # We obtain the posterior on the coefficients
            woodbury = np.dot(tDesignMatrix.T, tDesignMatrix) + nu2*np.eye(N_data)
            chol_woodbury = spla.cholesky(woodbury)
            # inverseWoodbury = chol2inv(chol_woodbury)
            z = np.dot(tDesignMatrix, gp.observed_values / nu2)
            # m = z - np.dot(tDesignMatrix, np.dot(inverseWoodbury, np.dot(tDesignMatrix.T, z)))
            m = z - np.dot(tDesignMatrix, spla.cho_solve((chol_woodbury, False), np.dot(tDesignMatrix.T, z))) 
            # (above) alternative to original but with cho_solve
            
            # z = np.dot(tDesignMatrix, gp.observed_values / nu2)
            # m = np.dot(np.eye(nFeatures) - \
            # np.dot(tDesignMatrix, spla.cho_solve((chol_woodbury, False), tDesignMatrix.T)), z)
            
            # woodbury has size N_data by N_data
            D, U = npla.eigh(woodbury)
            # sort the eigenvalues (not sure if this matters)
            idx = D.argsort()[::-1] # in decreasing order instead of increasing
            D = D[idx]
            U = U[:,idx]
            R = 1.0 / (np.sqrt(D) * (np.sqrt(D) + np.sqrt(nu2)))
            # R = 1.0 / (D + np.sqrt(D*nu2))

            # We sample from the posterior of the coefficients
            theta = randomness - \
    np.dot(tDesignMatrix, np.dot(U, (R * np.dot(U.T, np.dot(tDesignMatrix.T, randomness))))) + m

        else:
            # all you are doing here is sampling from the posterior of the linear model
            # that approximates the GP
            # Sigma = matrixInverse(np.dot(tDesignMatrix, tDesignMatrix.T) / nu2 + np.eye(nFeatures))
            # m = np.dot(Sigma, np.dot(tDesignMatrix, gp.observed_values / nu2))
            # theta = m + np.dot(randomness, spla.cholesky(Sigma, lower=False)).T

            # Sigma = matrixInverse(np.dot(tDesignMatrix, tDesignMatrix.T) + nu2*np.eye(nFeatures))
            # m = np.dot(Sigma, np.dot(tDesignMatrix, gp.observed_values))
            # theta = m + np.dot(randomness, spla.cholesky(Sigma*nu2, lower=False)).T

            chol_Sigma_inverse = spla.cholesky(np.dot(tDesignMatrix, tDesignMatrix.T) + nu2*np.eye(nFeatures))
            Sigma = chol2inv(chol_Sigma_inverse)
            m = spla.cho_solve((chol_Sigma_inverse, False), np.dot(tDesignMatrix, gp.observed_values))
            theta = m + np.dot(randomness, spla.cholesky(Sigma*nu2, lower=False)).T


    else:
        # We sample from the prior -- same for Matern
        theta = npr.randn(nFeatures)

    def wrapper(x, gradient): 
    # the argument "gradient" is 
    # not the usual compute_grad that computes BOTH when true
    # here it only computes the objective when true
        
        if x.ndim == 1:
            x = x[None,:]

        x = gp.noiseless_kernel.transformer.forward_pass(x)

        if not gradient:
            result = np.dot(theta.T, np.sqrt(2.0 * sigma2 / nFeatures) * np.cos(np.dot(W, x.T) + b))
            if result.size == 1:
                result = float(result) # if the answer is just a number, take it out of the numpy array wrapper
                # (failure to do so messed up NLopt and it only gives a cryptic error message)
            return result
        else:
            grad = np.dot(theta.T, -np.sqrt(2.0 * sigma2 / nFeatures) * np.sin(np.dot(W, x.T) + b) * W)
	    return gp.noiseless_kernel.transformer.backward_pass(grad)
    
    return wrapper

"""
Given some approximations to the GP sample, find a subset of the pareto set
wrapper_functions should be a dict with keys 'objective' and optionally 'constraints'
"""
# find MINIMUM if minimize=True, else find a maximum

def global_optimization_of_GP_approximation(funs, num_dims, inputs, minimize=True):

	moo = MOOP_basis_functions(funs['objectives'], num_dims, funs['constraints'])

	# XXX Currently it is only implemented the grid approach

	assert USE_GRID_ONLY == True

	if USE_GRID_ONLY == True:

#		moo.solve_using_grid(grid = sobol_grid.generate(num_dims, num_dims * GRID_SIZE))

                # We add the observations made

#		grid = np.vstack((sobol_grid.generate(num_dims, num_dims * GRID_SIZE), inputs))

		grid = np.vstack((np.random.uniform(size = ((num_dims * GRID_SIZE, num_dims))), inputs))

                # This is to speed up the optimization and to avoid evaluating the constraints many times
                # on infeasible points

                grid = find_feasible_grid(funs['constraints'], grid)

                if grid is not None:

                    for objective in funs['objectives']:

                        result_to_add = global_optimization_of_GP_approximation_single_obj_multiple_cons( \
                            {'objective': objective, 'constraints': funs['constraints']}, num_dims, grid)

                        if result_to_add is not None:
                            if np.min(cdist(grid, result_to_add)) > 1e-6:
                                grid = np.vstack((grid, result_to_add))
                else:

                    # The problem has no solution! This is to send something to solve_using_grid

                    grid = np.zeros((1, num_dims))

		moo.solve_using_grid(grid)

# 		DHL : Removed because in a constrained setting it does not make sense

#		for i in range(len(funs['objectives'])):
#			result = find_global_optimum_GP_sample(funs['objectives'][ i ], num_dims, grid, minimize)
#			moo.append_to_population(result)

	else:

		assert False

		# TODO

		assert NSGA2_POP > len(funs['objectives']) + 1

		moo.solve_using_grid(grid = sobol_grid.generate(num_dims, num_dims * GRID_SIZE))

		for i in range(len(funs['objectives'])):
			result = find_global_optimum_GP_sample(funs['objectives'][ i ], num_dims, grid, minimize)
			moo.append_to_population(result)

		pareto_set = moo.compute_pareto_front_and_set_summary(NSGA2_POP)['pareto_set']

		moo.initialize_population(np.maximum(NSGA2_POP - pareto_set.shape[ 0 ], 0))

		for i in range(pareto_set.shape[ 0 ]):
			moo.append_to_population(pareto_set[ i, : ])

		moo.evolve_population_only(NSGA2_EPOCHS)

		for i in range(pareto_set.shape[ 0 ]):
			moo.append_to_population(pareto_set[ i, : ])

	result = moo.compute_pareto_front_and_set_summary(PARETO_SET_SIZE)

	return result['pareto_set']


"""
Given some approximations to the GP sample, find its minimum
We do that by first evaluating it on a grid, taking the best, and using that to
initialize an optimization. If nothing on the grid satisfies the constraint, then
we return None

wrapper_functions should be a dict with keys 'objective' and optionally 'constraints'
"""
# find MINIMUM if minimize=True, else find a maximum

def find_feasible_grid(constraints, grid):

    con_evals = np.ones(grid.shape[0]).astype('bool')

    for con_fun in constraints:
        con_evals = np.logical_and(con_evals, con_fun(grid, gradient = False) >= 0)

    if not np.any(con_evals):
        return None
    else:
        return grid[ con_evals, : ]

def global_optimization_of_GP_approximation_single_obj_multiple_cons(funs, num_dims, feasible_grid, minimize=True):
    
    assert num_dims == feasible_grid.shape[ 1 ]

    num_con = len(funs['constraints'])

    if feasible_grid is None:
        return None

    obj_evals = funs['objective'](feasible_grid, gradient = False)

    if minimize:
        best_guess_index = np.argmin(obj_evals)
        best_guess_value = np.min(obj_evals)
    else:
        best_guess_index = np.argmax(obj_evals)
        best_guess_value = np.max(obj_evals)

    x_initial = feasible_grid[ best_guess_index, : ]

    fun_counter = defaultdict(int)

    assert minimize # todo - can fix later

    f       = lambda x: float(funs['objective'](x, gradient=False))
    f_prime = lambda x: funs['objective'](x, gradient=True).flatten()

    constraint_tol = 1e-6 # my tolerance

    # with SLSQP in scipy, the constraints are written as c(x) >= 0

    # We substract a small value from the constraints to guarantee that we find something feasible

    def g(x):
        g_func = np.zeros(num_con)
        for i,constraint_wrapper in enumerate(funs['constraints']):
            g_func[i] = constraint_wrapper(x, gradient=False)
        return g_func

    def g_prime(x):
        g_grad_func = np.zeros((num_con, num_dims))
        for i,constraint_wrapper in enumerate(funs['constraints']):
            g_grad_func[i,:] = constraint_wrapper(x, gradient=True)
        return g_grad_func

    bounds = [(0.0,1.0)]*num_dims

    opt_x = spo.fmin_slsqp(f, x_initial.copy(), bounds=bounds, disp=0, fprime=f_prime, f_ieqcons=g, fprime_ieqcons=g_prime)

    # make sure bounds are respected

    opt_x[opt_x > 1.0] = 1.0
    opt_x[opt_x < 0.0] = 0.0

    if f(opt_x) < best_guess_value and np.all(g(opt_x)>= 0):
       return opt_x[None]
    else:
        logging.debug('SLSQP failed when optimizing x*')

        # We try to solve the problem again but more carefully (we substract 1e-6 to the constraints)

        def g(x):
            g_func = np.zeros(num_con)
            for i,constraint_wrapper in enumerate(funs['constraints']):
                g_func[i] = constraint_wrapper(x, gradient=False) - 1e-6
            return g_func

        opt_x = spo.fmin_slsqp(f, x_initial.copy(), bounds=bounds, disp=0, fprime=f_prime, f_ieqcons=g, fprime_ieqcons=g_prime)

        opt_x[opt_x > 1.0] = 1.0
        opt_x[opt_x < 0.0] = 0.0

        if f(opt_x) < best_guess_value and np.all(g(opt_x)>= - 1e-6):
            return opt_x[None]
        else:
            logging.debug('SLSQP failed two times when optimizing x*')
            return x_initial[None]

# This functions finds the global optimum of each objective, which could be useful to
# initialize the population in NSGA2

def find_global_optimum_GP_sample(fun, num_dims, grid, minimize = True):

	assert num_dims == grid.shape[ 1 ]

	# First, evaluate on a grid 

	obj_evals = fun(grid, gradient = False)

	if minimize:
		best_guess_index = np.argmin(obj_evals)
		best_guess_value = np.min(obj_evals)
	else:
		best_guess_index = np.argmax(obj_evals)
		best_guess_value = np.max(obj_evals)

	x_initial = grid[ best_guess_index ]

	def f(x):
		if x.ndim == 1:
			x = x[None,:]

		a = fun(x, gradient = False)
		a_grad = fun(x, gradient = True)

		return (a, a_grad)
                
	bounds = [ (0, 1) ] * num_dims
	x_opt, y_opt, opt_info = spo.fmin_l_bfgs_b(f, x_initial, bounds=bounds, disp=0, approx_grad = False)
	# make sure bounds are respected

	x_opt[ x_opt > 1.0 ] = 1.0
	x_opt[ x_opt < 0.0 ] = 0.0

	return x_opt

def compute_unconstrained_variances_and_init_acq_fun(obj_models_dict, cand, con_models):
            unconstrainedVariances = dict()
            constrainedVariances = dict()
            acq = dict()

            for obj in obj_models_dict:
                unconstrainedVariances[ obj ] = obj_models_dict[ obj ].predict(cand, full_cov=True)[ 1 ]
                np.fill_diagonal(unconstrainedVariances[ obj ], np.diagonal(unconstrainedVariances[ obj ]) + obj_models_dict[ obj ].noise_value())

            for cons in con_models:
                unconstrainedVariances[ cons ] = con_models[ cons ].predict(cand, full_cov=True)[ 1 ]
                np.fill_diagonal(unconstrainedVariances[ cons ], np.diagonal(unconstrainedVariances[ cons ]) + con_models[ cons ].noise_value())

            for t in unconstrainedVariances:
                acq[t] = 0

            return acq, unconstrainedVariances, constrainedVariances

#Include test marginals.
def update_full_marginals(self, a):
                n_obs = a['n_obs']
                n_total = a['n_total']
                n_pset = a['n_pset']
                n_test = a['n_test']
                objectives = a['objs']
                constraints = a['cons']
                all_tasks = objectives
                all_constraints = constraints
                n_objs = len(all_tasks)
                n_cons = len(all_constraints)
                ntask = 0

                # Updating constraint distribution marginals.
                #vTilde_back = defaultdict(lambda: np.zeros((n_total, n_total)))
                #vTilde_cons_back = defaultdict(lambda: np.zeros((n_total, n_total)))
                #import pdb; pdb.set_trace();
                for cons in all_constraints:

                        # Summing all the factors into the diagonal. Reescribir.

                        vTilde_cons = np.zeros((n_total,n_total))
                        vTilde_cons[ np.eye(n_total).astype('bool') ] = np.append(np.append(np.sum(a['a_c_hfhat'][ :, : , ntask ], axis = 1), \
                                np.sum(a['c_c_hfhat'][ :, : , ntask ], axis = 1) + a['ehfhat'][ :, ntask ]), \
                                np.sum(a['g_c_hfhat'][ :, : , ntask ], axis = 1))

                        mTilde_cons = np.append(np.append(np.sum(a['b_c_hfhat'][ :, : , ntask ], axis = 1), \
                                np.sum(a['d_c_hfhat'][ :, : , ntask ], axis = 1) + a['fhfhat'][ :, ntask ]), \
                                 np.sum(a['h_c_hfhat'][ :, : , ntask ], axis = 1))

                        # Natural parameter conversion and update of the marginal variance matrices.

                        a['Vinv_cons'][cons] = a['VpredInv_cons'][cons] + vTilde_cons
                        a['V_cons'][cons] = matrixInverse(a['VpredInv_cons'][cons] + vTilde_cons)

                        # Natural parameter conversion and update of the marginal mean vector.

                        a['m_nat_cons'][cons] = np.dot(a['VpredInv_cons'][cons], a['mPred_cons'][cons]) + mTilde_cons
                        a['m_cons'][cons] = np.dot(a['V_cons'][cons], a['m_nat_cons'][ cons ])

                        ntask = ntask + 1
                ntask = 0
                for obj in all_tasks:

                        vTilde = np.zeros((n_total,n_total))

                        vTilde[ np.eye(n_total).astype('bool') ] = np.append(np.append(np.sum(a['ahfhat'][ :, : , ntask, 0, 0 ], axis = 1), \
                                np.sum(a['ahfhat'][ :, : , ntask, 1, 1 ], axis = 0) + np.sum(a['chfhat'][ :, : , ntask, 0, 0 ], axis = 1) + \
                                np.sum(a['chfhat'][ :, : , ntask, 1, 1 ], axis = 0) + np.sum(a['ghfhat'][ :, : , ntask, 1, 1 ], axis = 0)), \
                                np.sum(a['ghfhat'][ :, : , ntask, 0, 0 ], axis = 1))

                        vTilde[ n_obs : n_obs + n_pset, n_obs : n_obs + n_pset ] = vTilde[ n_obs : n_obs + n_pset, n_obs : n_obs + n_pset ] + \
                                a['chfhat'][ :, : , ntask, 0, 1 ] + a['chfhat'][ :, : , ntask, 1, 0 ].T

                        vTilde[ 0 : n_obs, n_obs : n_obs + n_pset ] = a['ahfhat'][ :, :, ntask, 0, 1]
                        vTilde[ n_obs : n_obs + n_pset, 0 : n_obs ] =  a['ahfhat'][ :, :, ntask, 0, 1].transpose()

                        vTilde[ n_obs + n_pset : n_total, n_obs : n_obs + n_pset ] = a['ghfhat'][ :, :, ntask, 0, 1]
                        vTilde[ n_obs : n_obs + n_pset, n_obs + n_pset : n_total ] =  a['ghfhat'][ :, :, ntask, 0, 1].transpose()

                        a['Vinv'][obj] = a['VpredInv'][obj] + vTilde
                        a['V'][obj] = matrixInverse(a['VpredInv'][obj] + vTilde)

                        mTilde = np.append(np.append(np.sum(a['bhfhat'][ :, : , ntask, 0 ], axis = 1),
                                np.sum(a['bhfhat'][ :, : , ntask, 1 ], axis = 0) + np.sum(a['hhfhat'][ :, : , ntask, 1 ], axis = 0) +\
                                np.sum(a['dhfhat'][ :, : , ntask, 0 ], axis = 1) + np.sum(a['dhfhat'][ :, : , ntask, 1 ], axis = 0)), \
                                np.sum(a['hhfhat'][ :, : , ntask, 0 ], axis = 1))

                        a['m_nat'][obj] = np.dot(a['VpredInv'][obj], a['mPred'][obj]) + mTilde
                        a['m'][obj] = np.dot(a['V'][obj], a['m_nat'][ obj ])

                        ntask = ntask + 1

                return a

def get_test_predictive_distributions(self, a):
                n_obs = a['n_obs']
                n_pset = a['n_pset']
                n_test = a['n_test']
                n_total = a['n_total']
                q = len(a['objs'])
                c = len(a['cons'])
                predictive_distributions = {
                        'mf' : defaultdict(lambda: np.zeros(n_test)),
                        'vf' : defaultdict(lambda: np.zeros((n_test, n_test))),
                        'mc' : defaultdict(lambda: np.zeros(n_test)),
                        'vc' : defaultdict(lambda: np.zeros((n_test, n_test))),
                }
                for obj in a['objs'].keys():
                        predictive_distributions['mf'][ obj ] = a['m'][ obj ][ n_obs + n_pset : n_total ]
                        predictive_distributions['vf'][ obj ] = a['V'][ obj ][ n_obs + n_pset : n_total , n_obs + n_pset : n_total ]
                for cons in a['cons'].keys():
                        predictive_distributions['mc'][ cons ] = a['m_cons'][ cons ][ n_obs + n_pset : n_total ]
                        predictive_distributions['vc'][ cons ] = a['V_cons'][ cons ][ n_obs + n_pset : n_total , n_obs + n_pset : n_total ]


                return predictive_distributions, a

def compute_PPESMOC_approximation(self, predictionEP, obj_models_dict, con_models, unconstrainedVariances, constrainedVariances, acq):

                predictionEP_obj = predictionEP[ 'vf' ]
                predictionEP_cons = predictionEP[ 'vc' ]

                # DHL changed fill_diag, because that was updating the a structure and screwing things up later on

                for obj in obj_models_dict:
                    predictionEP_obj[ obj ] = predictionEP_obj[ obj ] + np.eye(predictionEP_obj[ obj ].shape[ 0 ]) * obj_models_dict[ obj ].noise_value()
                    constrainedVariances[ obj ] = predictionEP_obj[ obj ]

                for cons in con_models:
                    predictionEP_cons[ cons ] = predictionEP_cons[ cons ] + np.eye(predictionEP_cons[ obj ].shape[ 0 ]) * con_models[ cons ].noise_value()
                    constrainedVariances[ cons ] = predictionEP_cons[ cons ]

                # We only care about the variances because the means do not affect the entropy
                # The summation of the acq of the tasks (t) is done in a higher method. Do no do it here.

                for t in unconstrainedVariances:

                    # DHL replaced np.log(np.linalg.det()) to avoid precision errors

                    value = 0.5 * np.linalg.slogdet(unconstrainedVariances[t])[ 1 ] - 0.5 * np.linalg.slogdet(constrainedVariances[t])[ 1 ]

                    # We set negative values of the acquisition function to zero  because the 
                    # entropy cannot be increased when conditioning

                    value = np.max(value, 0)

                    acq[t] += value

                return acq

class PPESMOC(AbstractAcquisitionFunction):

	def __init__(self, num_dims, verbose=True, input_space=None, grid=None, opt = None):

		global NSGA2_POP
		global NSGA2_EPOCHS
		global PARETO_SET_SIZE
		global NUM_RANDOM_FEATURES
		global GRID_SIZE
		global USE_GRID_ONLY

		# we want to cache these. we use a dict indexed by the state integer

		self.cached_EP_solution = dict()
		self.cached_pareto_set = dict()
		self.cached_acquisition_on_grid = dict()
		self.has_gradients = True
		self.num_dims = num_dims
		self.input_space = input_space
		self.reuse_EP_solution = dict()
	
		self.options = PESM_OPTION_DEFAULTS.copy()

		if opt is not None: #For tests, opt can be None.
			self.options.update(opt)

		PARETO_SET_SIZE = self.options['pesm_pareto_set_size']
		NUM_RANDOM_FEATURES = self.options['pesm_num_random_features']

		NSGA2_POP = self.options['pesm_nsga2_pop'] 
		NSGA2_EPOCHS = self.options['pesm_nsga2_epochs'] 

		GRID_SIZE = self.options['pesm_grid_size'] 
		USE_GRID_ONLY = self.options['pesm_use_grid_only_to_solve_problem']

                self.max_ep_iterations = 250
                #self.max_ep_iterations = 5

	def mock_test_message(self):
		print "Testing access to class"

	# obj_models is a GP
	# con_models is a dict of named constraints and their GPs


	#Put in this method all necessary preconditions for the computation of PPESMOC.
	def assert_preconditions(self, models, compute_grad):
		#Caching must be set to true.
                for model in models:

                        # if model.pending is not None:
                        #     raise NotImplementedError("PES not implemented for pending stuff? Not sure. Should just impute the mean...")

                        if not model.options['caching']:
                                logging.error("Warning: caching is off while using PES!")
		
		assert len({model.state for model in models}) == 1, "Models are not all at the same state"
                assert compute_grad
	
	################################################################################################
	# sample_pareto_sets: Function that samples the configured samples per hyper of Pareto Sets.
	# 
	# INPUT:
	# models: GPs with the objectives.
	# con_models_dict: GPs with the constraints.
	# key: ID of the GP hyperparameter configuration.
	#
	# OUTPUT:
	# pareto_set: Dictionary of (int(self.options['pesm_samples_per_hyper']) pareto sets.
	#################################################################################################
	def sample_pareto_sets(self, models, con_models_dict, key):
		if not key in self.cached_pareto_set:

                        pareto_set = dict()

                        for i in range(int(self.options['pesm_samples_per_hyper'])):
                                pareto_set[ str(i) ] = sample_solution(self.num_dims, models, con_models_dict.values())

                        self.cached_pareto_set[ key ] = pareto_set
                else:
                        pareto_set = self.cached_pareto_set[ key ]
		
		return pareto_set

	###############################################################################
	# evaluate_acquisition_function_and_compute_EP_non_dependant_on_candidate_factors: It computes the EP factors that depend
	# on the candidate and it also evaluates the PPESMOC acquisition function now that it has all the information.
	#
	# INPUT:
	#
        # key: ID of the GP hyperparameter configuration
        # obj_model_dict: GPs with the objectives.
	# cand: Candidate points.
	# epSolution: EP factors that do not depend on the candidate.
	# pareto_set: Dictionary of Pareto Sets.
        # minimize: If FALSE, it maximizes.
        # con_models_dict: GPs with the constraints.
	#
	# OUTPUT:
	#
	# acq_dict: Dictionary with as many acquisition functions as tasks exist. 
	###############################################################################
	def evaluate_acquisition_function_and_compute_EP_factors_dependant_on_candidate(self, key, obj_model_dict, cand, epSolution, \
									pareto_set, minimize, con_models_dict):
		# Use the EP solution to compute the acquisition function 
                # This is a total hack to speed up things in the decoupled setting! We use the fact that 
                # we always evaluate first the acquisiton on the  grid to find a good point for the 
                # optimization. Then, we optimize (individual evaluations). The grid is always the same.
                if not key in self.cached_acquisition_on_grid:
                    acq_dict = self.evaluate_acquisition_function_given_EP_solution(obj_model_dict, cand, epSolution, pareto_set, \
                        minimize=minimize, opt = self.options, con_models=con_models_dict)
                    self.cached_acquisition_on_grid[ key ] = acq_dict
                else:

                    # If we are computing the acquisiton on the grid we reuse the stored result. Otherwise we recompute the
                    # acquisition                
                    if cand.shape[ 0 ] == 1:
                            acq_dict = self.evaluate_acquisition_function_given_EP_solution(obj_model_dict, cand, epSolution, pareto_set, \
                            minimize=minimize, opt = self.options, con_models=con_models_dict)
                    else:

                            acq_dict = self.cached_acquisition_on_grid[ key ]

		return acq_dict


	def _verify_models_same_state(self, models):
		for model in models:
                        if not model.options['caching']:
                                logging.error("Warning: caching is off while using PPESMOC!")

                # make sure all models are at the same state
                assert len({model.state for model in models}) == 1, "Models are not all at the same state"

	def verify_models_same_state_and_grad(self, obj_models, con_models, compute_grad):
		self._verify_models_same_state(obj_models)
		self._verify_models_same_state(con_models)
		#assert compute_grad #Not necessary for plots.

        #Function used to debug the results of a function.
        def print_args_debug(function_name, file_name, return_variables):
            log = open(file_name, "a")
            log.write('============================\n')
            log.write('Results of ' + function_name + '\n')
            for key, value in return_variables.items():
                log.write("%s == %s" %(key, value))
            log.write('============================\n')
            log.close()
            
	###############################################################################
	# acquistion: Function where the PPESMOC acquisition function is computed.
	#
	# INPUT:
	# obj_model_dict and con_model_dict: GPs with the objectives and constraints.
	# cand: Candidates where the acquisition function must be computed.
	# current_best: Current best point of the acquisition function.
	# compute_grad: Flag to compute gradients, if True, gradients must be computed.
	# minimize: If false, it maximizes the Gps.
	#
	# OUTPUT:
	# total_acq: Acquisition function.
	###############################################################################

	def acquisition(self, obj_model_dict, con_model_dict, cand, current_best, compute_grad, minimize=True, tasks=None, tasks_values=None,
                            test=True):
		obj_models = obj_model_dict.values() #Some redefinitions.
		models = obj_models

		self.verify_models_same_state_and_grad(obj_models, con_model_dict.values(), compute_grad)
		
		# We check if we have already computed the EP approximation. If so, we reuse the result obtained
		# and the pareto set samples.
                obj_model_dict_copy_gradients = obj_model_dict.copy() 
                con_model_dict_copy_gradients = con_model_dict.copy() 

		key = tuple([obj_model_dict[ obj ].state for obj in obj_model_dict])

		pareto_set = self.sample_pareto_sets(models, con_model_dict, key) #We sample a dictionary of Pareto Sets.
                pareto_set_copy_gradients = pareto_set.copy()
               
		self.sampled_pareto_set = pareto_set
		# We approximate the factors that do not depend on the candidate by using EP.
                if cand.shape[0] > 1:   #This is to show in the grid.
                    cand = cand.reshape((cand.shape[0], cand.shape[1], 1))
                    #print "Computing PPESMOC acquisition function for plotting"
                    acq_dict = np.array([self.compute_ep(pareto_set, obj_model_dict, minimize, con_model_dict, key, point, self.max_ep_iterations)[0] for point in cand])
                    total_acq = np.zeros((acq_dict.shape[0]))
                    for i in range(acq_dict.shape[0]):
                        for task in tasks:
                            total_acq[i] += acq_dict[i][task]

                    return total_acq
                else:
                    cand_test = cand.copy()
		    cand = cand.reshape((self.options['batch_size'], self.num_dims)) 

		    #print "Computing PPESMOC acquisition function"

		    #Here is where the PPESMOC acqusition function is computed (aka substraction of variance predictive distributions const. and unconst.)
		    #The omegas are the EP factors.
		    acq_dict, list_a = self.compute_ep(pareto_set, obj_model_dict, minimize, con_model_dict, key, cand, self.max_ep_iterations)

        	    # by default, sum the PESC contribution for all tasks
               	    if tasks is None:
		        tasks = acq_dict.keys()

		    # Compute the total acquisition function for the tasks of interests

		    total_acq = 0.0
		    for task in tasks:
	    	        total_acq += acq_dict[ task ]
                    #write_log('total_acq PPESMOC', self.options['log_route_acq_funs'], {'acq' : total_acq})            

		    #Compute the gradients of the acquisition function w.r.t the test points (cand).
		    #EP factors are traspassed, by the way, I could have traspased also all the computed matrices of PPESMOC,
		    #that would make less redundant code, more efficiency and less bugs.... I should have done it this way...
		    #Consider to do this if bugs are unbeatable.

                    #                ini = self.entropy_constrained_wrt_test_points_for_task(True, obj_model_dict[ obj_model_dict.keys()[ 1 ] ], obj_model_dict.keys()[ 1 ], cand, list_a[ str(0) ].copy(), inc = 1e-3)
                    #                fini = self.entropy_constrained_wrt_test_points_for_task(True, obj_model_dict[ obj_model_dict.keys()[ 1 ] ], obj_model_dict.keys()[ 1 ], cand, list_a[ str(0) ].copy(), inc = -1e-3)
                    #                
                    #                print((ini - fini) / (2 * 1e-3))
                    #
                    #                grad = self.compute_gradients_entropy_constrained_wrt_test_points_for_task(True, obj_model_dict[ obj_model_dict.keys()[ 1 ] ], obj_model_dict.keys()[ 1 ], cand, list_a[ str(0) ])
                    if not compute_grad:
                        return total_acq
                    else:
                        #Aqui es donde hay que devolver los gradientes pues es donde se calculaban antes. Tienes toda la info.
                        #compute_gradients_acq_fun_xtests = jacobian(compute_acq_fun_wrt_test_points)  
                        compute_gradients_acq_fun_xtests = grad(compute_acq_fun_wrt_test_points)  

                        #TIME TEST: NUMPY FUNCTION VS AUTOGRAD FUNCTION. 
                        if(test):
                            import pdb; pdb.set_trace();
                            import time
                            from spearmint.acquisition_functions.PPESMOC_gradients_time import compute_acq_fun_wrt_test_points_time
                            eps = 1e-4
                            time_numpy = np.zeros(10)
                            time_autograd = np.zeros(10)
                            time_grad_autograd = np.zeros(10)
                            time_autograd_time = np.zeros(10)
                            time_grad_autograd_time = np.zeros(10)
                            for i in range(10):
                                start = time.time()
                                self.compute_ep(pareto_set, obj_model_dict, minimize, con_model_dict, key, cand + np.random.uniform(0, 1e-1), self.max_ep_iterations)
                                end = time.time()
                                print('Elapsed time for Numpy function: ' + str(end-start) + ' seconds.')
                                time_numpy[i] = str(end-start)
                                start = time.time()
                                test_acq = compute_acq_fun_wrt_test_points(cand + np.random.uniform(0, 1e-1), obj_model_dict,\
                                        con_model_dict, pareto_set, list_a, tasks)
                                end = time.time()
                                print('Elapsed time for Autograd function: ' + str(end-start) + ' seconds.')
                                time_autograd[i] = str(end-start)
                                start = time.time()
                                test_acq_grad = compute_gradients_acq_fun_xtests(cand + np.random.uniform(0, 1e-1), obj_model_dict, con_model_dict, pareto_set, list_a, tasks)
                                end = time.time()
                                print('Elapsed time for Autograd gradients function: ' + str(end-start) + ' seconds.')
                                time_grad_autograd[i] = str(end-start)
                                compute_gradients_acq_fun_xtests_time = grad(compute_acq_fun_wrt_test_points_time)
                                start = time.time()
                                test_acq_time = compute_acq_fun_wrt_test_points_time(cand + np.random.uniform(0, 1e-1), obj_model_dict, con_model_dict, pareto_set, list_a, tasks)
                                end = time.time()
                                print('Elapsed time for Autograd time enhanced function: ' + str(end-start) + ' seconds.')
                                time_autograd_time[i] = str(end-start)
                                start = time.time()
                                test_acq_grad_time = compute_gradients_acq_fun_xtests_time(cand + np.random.uniform(0, 1e-1), obj_model_dict, con_model_dict, pareto_set, list_a, tasks)
                                end = time.time()
                                print('Elapsed time for Autograd gradients time enhanced function: ' + str(end-start) + ' seconds.')
                                time_grad_autograd_time[i] = str(end-start)
                                #assert np.abs(test_acq_time-test_acq) < 1e-3
                                #assert np.all(np.abs(test_acq_grad-test_acq_grad_time) < 1e-1)
                            print('Mean numpy time: ' + str(np.mean(time_numpy)) + ' seconds.')
                            print('Std. numpy time: ' + str(np.std(time_numpy)) + ' seconds.')
                            print('Mean autograd time: ' + str(np.mean(time_autograd)) + ' seconds.')
                            print('Std. autograd time: ' + str(np.std(time_autograd)) + ' seconds.')
                            print('Mean grad_autograd time: ' + str(np.mean(time_grad_autograd)) + ' seconds.')
                            print('Std. grad_autograd time: ' + str(np.std(time_grad_autograd)) + ' seconds.')
                            print('Mean autograd_time time: ' + str(np.mean(time_autograd_time)) + ' seconds.')
                            print('Std. autograd_time time: ' + str(np.std(time_autograd_time)) + ' seconds.')
                            print('Mean grad_autograd_time time: ' + str(np.mean(time_grad_autograd_time)) + ' seconds.')
                            print('Std. grad_autograd_time time: ' + str(np.std(time_grad_autograd_time)) + ' seconds.')
                            import pdb; pdb.set_trace();

                            import pdb; pdb.set_trace();

                        """
                        warnings.showwarning = warn_with_traceback
                        warnings.simplefilter("always")
                        warnings.simplefilter("error")
                        warnings.simplefilter("ignore", DeprecationWarning)
                        total_acq_autograd = compute_acq_fun_wrt_test_points(cand, obj_model_dict_copy_gradients, con_model_dict_copy_gradients, \
                            pareto_set_copy_gradients, list_a, tasks, self.options['log_route_acq_funs'], write_log) #Debug.
                        print total_acq_autograd
                        #Coincide casi totalmente!
                        """
                        #Test.
                        #autograd_acq = compute_acq_fun_wrt_test_points(cand, obj_model_dict, con_model_dict, pareto_set, list_a, tasks,
                                                                        #self.options['log_route_acq_funs'], write_log)
                        acq_grad = compute_gradients_acq_fun_xtests(cand, obj_model_dict, con_model_dict, pareto_set, list_a, tasks)#,
                                                                #self.options['log_route_acq_funs'], write_log) 
                        #Testing autograd acquisition function. It must give the same value than PPESMOC.
                        #assert total_acq == total_acq_autograd Gives a diference of 1e-12. Can be OK.

                        #Testing autograd gradients.    
                        #import pdb; pdb.set_trace();
                        #self.test_full_acquisition_gradients(pareto_set, obj_model_dict, minimize, con_model_dict, key, tasks, cand, \
                        #self.options['log_route_acq_funs'], write_log, compute_gradients_acq_fun_xtests, compute_acq_fun_wrt_test_points)
    
	                return total_acq, acq_grad

        # This function is only for testing that the gradient is OK

	def entropy_constrained_wrt_test_points_for_task(self, is_objective, task_model, task, cand, a, inc = 0.0):
            
            n_obs = a['n_obs']
            n_pset = a['n_pset']
            n_test = a['n_test']
            n_total = a['n_total']
 
            eps = cand * 0.0
            eps[0,1] = inc

            KooPlusInoise = task_model.kernel.cov(task_model.inputs)
            Kot = task_model.noiseless_kernel.cross_cov(task_model.inputs, cand)
            Kot_new = task_model.noiseless_kernel.cross_cov(task_model.inputs, cand + eps)
            Ktt = task_model.noiseless_kernel.cov(cand)
            Ktt_new = task_model.noiseless_kernel.cov(cand + eps)
            Kxt = task_model.noiseless_kernel.cross_cov(a['X'][ 0 : (n_total - n_test), ], cand)
            Kxt_new = task_model.noiseless_kernel.cross_cov(a['X'][ 0 : (n_total - n_test), ], cand + eps)
            Kox = task_model.noiseless_kernel.cross_cov(task_model.inputs, a['X'][ 0 : (n_total - n_test), ])
            Kxx = task_model.noiseless_kernel.cov(a['X'][ 0 : (n_total - n_test), ])

            Byy = Ktt - np.dot(np.dot(Kot.T, matrixInverse(KooPlusInoise)), Kot)
            Byy_new = Ktt_new - np.dot(np.dot(Kot_new.T, matrixInverse(KooPlusInoise)), Kot_new)
            Bxy = Kxt - np.dot(np.dot(Kox.T, matrixInverse(KooPlusInoise)), Kot)
            Bxy_new = Kxt_new - np.dot(np.dot(Kox.T, matrixInverse(KooPlusInoise)), Kot_new)
            Bxx = Kxx - np.dot(np.dot(Kox.T, matrixInverse(KooPlusInoise)), Kox)

            if is_objective:
                Theta_xx = (a['Vinv'][ task ] - a['VpredInv'][ task ])[ 0 : (n_total - n_test), 0 : (n_total - n_test) ]
                Theta_xy = (a['Vinv'][ task ] - a['VpredInv'][ task ])[ 0 : (n_total - n_test), (n_total - n_test) : n_total ]
                Theta_yy = (a['Vinv'][ task ] - a['VpredInv'][ task ])[ (n_total - n_test) : n_total, (n_total - n_test) : n_total ]
            else:
                Theta_xx = (a['Vinv_cons'][ task ] - a['VpredInv_cons'][ task ])[ 0 : (n_total - n_test), 0 : (n_total - n_test) ]
                Theta_xy = (a['Vinv_cons'][ task ] - a['VpredInv_cons'][ task ])[ 0 : (n_total - n_test), (n_total - n_test) : n_total ]
                Theta_yy = (a['Vinv_cons'][ task ] - a['VpredInv_cons'][ task ])[ (n_total - n_test) : n_total, \
                    (n_total - n_test) : n_total ]

            Cxx = matrixInverse(Bxx - np.dot(np.dot(Bxy, matrixInverse(Byy)), Bxy.T))
            Cxx_new = matrixInverse(Bxx - np.dot(np.dot(Bxy_new, matrixInverse(Byy_new)), Bxy_new.T))
            Cyy_new = matrixInverse(Byy_new - np.dot(np.dot(Bxy_new.T, matrixInverse(Bxx)), Bxy_new))
            Cyy = matrixInverse(Byy - np.dot(np.dot(Bxy.T, matrixInverse(Bxx)), Bxy))
            Cxy = - np.dot(np.dot(matrixInverse(Bxx), Bxy), Cyy)
            Cxy_new = - np.dot(np.dot(matrixInverse(Bxx), Bxy_new), Cyy_new)
            
            Dxx = Cxx + Theta_xx
            Dxx_new = Cxx_new + Theta_xx
            Dyy = Cyy + Theta_yy
            Dyy_new = Cyy_new + Theta_yy
            Dxy = Cxy + Theta_xy
            Dxy_new = Cxy_new + Theta_xy

            Eyy = matrixInverse(Dyy_new - np.dot(Dxy_new.T, np.dot(matrixInverse(Dxx_new), Dxy_new)))

            if 'noise' not in task_model.params:
                task_model.params['noise'] = 0.0

            return 0.5 * np.linalg.slogdet(Eyy + np.eye(cand.shape[ 0 ]) * task_model.params['noise'].value)[ 1 ]

        # This function computes the gradients for a particular task of the constrained entropy

	def compute_gradients_entropy_constrained_wrt_test_points_for_task(self, is_objective, task_model, task, cand, a):

            n_obs = a['n_obs']
            n_pset = a['n_pset']
            n_test = a['n_test']
            n_total = a['n_total']

            other = a['X'][ 0 : (n_total - n_test), : ]

            # Only works for Matern2 kernels since the gradients are optimzied for this kernel. 
            # Can be easily modified to include SquaredExp.

            assert task_model.options['kernel'] == "Matern52"

            
            if 'noise' not in task_model.params:
                noise_level = 0.0
            else:
                noise_level = task_model.params['noise'].value

            if is_objective:
                Vinv = matrixInverse(a['V'][ task ][ n_obs + n_pset : n_total, n_obs + n_pset : n_total  ] \
                    + np.eye(cand.shape[ 0 ]) * noise_level) 
            else:
                Vinv = matrixInverse(a['V_cons'][ task ][ n_obs + n_pset : n_total, n_obs + n_pset : n_total ] \
                    + np.eye(cand.shape[ 0 ]) * noise_level)

            from scipy.spatial.distance import cdist

            r2 = cdist(cand / task_model.params['ls'].value, cand / task_model.params['ls'].value, 'sqeuclidean')
            r = np.sqrt(r2)
            grad_r2 = (5.0/6.0) * np.exp(-np.sqrt(5.0) * r) * (1 + np.sqrt(5.0)*r) * task_model.params['amp2'].value * -1.0

            # The gradient of the squared distance has structure which we use here: d r2 / d_xij = 
            # 2 * (x_ij * t(d_i) %*% 1 - x*j d_i^T) + 2 * (transpose_of_prevous_thing)
            # Importantly, the data has to be scaled by the lengthscale
            # We use that Tr(A * (B o cc^T)) = Tr((A o B) * cc^T), where o is the hadamard product

            # Some matrices we may need

            KooPlusInoise = task_model.kernel.cov(task_model.inputs)
            KooPlusInoiseInv = matrixInverse(KooPlusInoise)
            Kot = task_model.noiseless_kernel.cross_cov(task_model.inputs, cand)
            Ktt = task_model.noiseless_kernel.cov(cand)
            Kxt = task_model.noiseless_kernel.cross_cov(a['X'][ 0 : (n_total - n_test), ], cand)
            Kox = task_model.noiseless_kernel.cross_cov(task_model.inputs, a['X'][ 0 : (n_total - n_test), ])
            Kxx = task_model.noiseless_kernel.cov(a['X'][ 0 : (n_total - n_test), ])

#            Byy = Ktt - np.dot(np.dot(Kot.T, KooPlusInoiseInv), Kot)
#            ByyInv = matrixInverse(Byy)
#            Bxy = Kxt - np.dot(np.dot(Kox.T, KooPlusInoiseInv), Kot)
#            Bxx = Kxx - np.dot(np.dot(Kox.T, KooPlusInoiseInv), Kox)
#            BxxInv = matrixInverse(Bxx)


            if is_objective:
                Theta_xx = (a['Vinv'][ task ] - a['VpredInv'][ task ])[ 0 : (n_total - n_test), 0 : (n_total - n_test) ]
                Theta_xy = (a['Vinv'][ task ] - a['VpredInv'][ task ])[ 0 : (n_total - n_test), (n_total - n_test) : n_total ]
                Theta_yy = (a['Vinv'][ task ] - a['VpredInv'][ task ])[ (n_total - n_test) : n_total, (n_total - n_test) : n_total ]

                # Robust Computations

                Byy = a['Vpred'][ task ][ (n_total - n_test) : n_total,(n_total - n_test) : n_total ]
                ByyInv = matrixInverse(Byy)
                Bxy = a['Vpred'][ task ][ 0 : (n_total - n_test), (n_total - n_test) : n_total ]
                Bxx = a['Vpred'][ task ][ 0 : (n_total - n_test), 0 : (n_total - n_test) ]
                Cxx = a['VpredInv'][ task ][ 0 : (n_total - n_test), 0 : (n_total - n_test) ]
                Cyy = a['VpredInv'][ task ][ (n_total - n_test) : n_total, (n_total - n_test) : n_total ]
                Cxy = a['VpredInv'][ task ][  0 : (n_total - n_test), (n_total - n_test) : n_total ]
                Dxx = a['Vinv'][ task ][ 0 : (n_total - n_test), 0 : (n_total - n_test) ]
                Dyy = a['Vinv'][ task ][ (n_total - n_test) : n_total, (n_total - n_test) : n_total ]
                Dxy = a['Vinv'][ task ][  0 : (n_total - n_test), (n_total - n_test) : n_total ]
                Eyy = a['V'][ task ][ (n_total - n_test) : n_total, (n_total - n_test) : n_total ]
                BxxInv = matrixInverse(Bxx)

            else:

                Theta_xx = (a['Vinv_cons'][ task ] - a['VpredInv_cons'][ task ])[ 0 : (n_total - n_test), 0 : (n_total - n_test) ]
                Theta_xy = (a['Vinv_cons'][ task ] - a['VpredInv_cons'][ task ])[ 0 : (n_total - n_test), (n_total - n_test) : n_total ]
                Theta_yy = (a['Vinv_cons'][ task ] - a['VpredInv_cons'][ task ])[ (n_total - n_test) : n_total, \
                    (n_total - n_test) : n_total ]

                # Robust Computations

                Byy = a['Vpred_cons'][ task ][ (n_total - n_test) : n_total,(n_total - n_test) : n_total ]
                ByyInv = matrixInverse(Byy)
                Bxy = a['Vpred_cons'][ task ][ 0 : (n_total - n_test), (n_total - n_test) : n_total ]
                Bxx = a['Vpred_cons'][ task ][ 0 : (n_total - n_test), 0 : (n_total - n_test) ]
                Cxx = a['VpredInv_cons'][ task ][ 0 : (n_total - n_test), 0 : (n_total - n_test) ]
                Cyy = a['VpredInv_cons'][ task ][ (n_total - n_test) : n_total, (n_total - n_test) : n_total ]
                Cxy = a['VpredInv_cons'][ task ][  0 : (n_total - n_test), (n_total - n_test) : n_total ]
                Dxx = a['Vinv_cons'][ task ][ 0 : (n_total - n_test), 0 : (n_total - n_test) ]
                Dyy = a['Vinv_cons'][ task ][ (n_total - n_test) : n_total, (n_total - n_test) : n_total ]
                Dxy = a['Vinv_cons'][ task ][  0 : (n_total - n_test), (n_total - n_test) : n_total ]
                Eyy = a['V_cons'][ task ][ (n_total - n_test) : n_total, (n_total - n_test) : n_total ]
                BxxInv = matrixInverse(Bxx)


#           Cxx = matrixInverse(Bxx - np.dot(np.dot(Bxy, ByyInv), Bxy.T))
#           Cyy = matrixInverse(Byy - np.dot(np.dot(Bxy.T, matrixInverse(Bxx)), Bxy))

            BxxInvBxy = np.dot(matrixInverse(Bxx), Bxy)

#            Cxy = - np.dot(BxxInvBxy, Cyy)
            
#            Dxx = Cxx + Theta_xx
#            Dyy = Cyy + Theta_yy
#            Dxy = Cxy + Theta_xy
        
            DxxInv = matrixInverse(Dxx)
            DxxInvDxy = np.dot(DxxInv, Dxy)

#            try:
#                Eyy = matrixInverse(Dyy - np.dot(Dxy.T, DxxInvDxy))
#            except:
#                import pdb; pdb.set_trace()

#            import pdb; pdb.set_trace()

            EyyCyy = np.dot(Eyy, Cyy)
            CyyEyyVinv = np.dot(EyyCyy.T, Vinv)
            CyyEyyVinvEyyCyy = np.dot(CyyEyyVinv, EyyCyy)

            M = CyyEyyVinvEyyCyy * grad_r2 

            # Contribution of Dyy done, now the contribution of DyxDxxInvDxy. We use that Dyx = Cyx + Cte
            # where Cyx = Cxy.T = - (BxxInv Bxy Cyy).T = - Cyy Byx BxxInv and reuse the gradients of Dyy

            DyxDxxInvBxxInvBxy = np.dot(DxxInvDxy.T, BxxInvBxy)
            DyxDxxInvBxxInvBxyCyy = np.dot(DyxDxxInvBxxInvBxy, Cyy)
            EyyDyxDxxInvBxxInvBxyCyy = np.dot(Eyy, DyxDxxInvBxxInvBxyCyy)
            CyyEyyVinvEyyDyxDxxInvBxxInvBxyCyy = np.dot(CyyEyyVinv, EyyDyxDxxInvBxxInvBxyCyy)
            CyyEyyVinvEyyDyxDxxInvBxxInv = np.dot(CyyEyyVinv, np.dot(Eyy, np.dot(DxxInvDxy.T, BxxInv)))

            M = M + 2.0 * CyyEyyVinvEyyDyxDxxInvBxxInvBxyCyy * grad_r2

            # Contribution of Dyx done, now the contribution of DxxInv

            EyyDyxDxxInvCxx = np.dot(np.dot(Eyy, DxxInvDxy.T), Cxx)
            EyyDyxDxxInvCxxBxy = np.dot(EyyDyxDxxInvCxx, Bxy)
            EyyDyxDxxInvCxxBxyByyInv = np.dot(EyyDyxDxxInvCxxBxy, ByyInv)
            ByyInvByxCxxInvDxxInvDxyEyyVinvEyyDyxDxxInvCxxInvBxyByyInv = np.dot(np.dot(EyyDyxDxxInvCxxBxyByyInv.T, Vinv), EyyDyxDxxInvCxxBxyByyInv)

            M = M + ByyInvByxCxxInvDxxInvDxyEyyVinvEyyDyxDxxInvCxxInvBxyByyInv * grad_r2

            colSums = np.dot(np.ones(M.shape[ 0 ]), M).reshape((cand.shape[ 0 ], 1))
            rowSums = np.dot(np.ones(M.shape[ 0 ]), M.T).reshape((cand.shape[ 0 ], 1))

           # We start with the gradient of Kx_test x_test

            gradient = 2.0 * (colSums * cand / (task_model.params['ls'].value**2) - np.dot(M, cand / task_model.params['ls'].value**2)) + \
                2.0 * (rowSums * cand / (task_model.params['ls'].value**2) - np.dot(M.T, cand / task_model.params['ls'].value**2))

            # Next, the gradient of Kox_test 

            import scipy.linalg as spla

            Q = spla.cho_solve((spla.cholesky(task_model.kernel.cov(task_model.inputs)), False), \
                task_model.noiseless_kernel.cross_cov(task_model.inputs, cand))

            Qother = spla.cho_solve((spla.cholesky(task_model.kernel.cov(task_model.inputs)), False), \
                task_model.noiseless_kernel.cross_cov(task_model.inputs, other))

            r2 = cdist(task_model.inputs / task_model.params['ls'].value, cand / task_model.params['ls'].value, 'sqeuclidean')
            r = np.sqrt(r2)
            grad_r2 = (5.0/6.0) * np.exp(-np.sqrt(5.0) * r) * (1 + np.sqrt(5.0)*r) * task_model.params['amp2'].value * -1.0

            BxxInvBxy = np.dot(matrixInverse(Bxx), Bxy)
            QotherBxxInvBxy = np.dot(Qother, BxxInvBxy)

            M = (np.dot(-2.0 * Q + 2.0 * QotherBxxInvBxy, CyyEyyVinvEyyCyy) * grad_r2)

            # Contribution of Dyy done, now the contribution of DyxDxxInvDxy. We use that Dyx = Cyx + Cte
            # where Cyx = Cxy.T = - (BxxInv Bxy Cyy).T = - Cyy Byx BxxInv and reuse the gradients of Dyy

            M = M + 2.0 * ((np.dot(-1.0 * Q, CyyEyyVinvEyyDyxDxxInvBxxInvBxyCyy) * grad_r2) +  \
                (np.dot(-1.0 * Q, CyyEyyVinvEyyDyxDxxInvBxxInvBxyCyy.T) * grad_r2))

            M = M + 2.0 * ((np.dot(QotherBxxInvBxy, CyyEyyVinvEyyDyxDxxInvBxxInvBxyCyy) * grad_r2) +  \
                (np.dot(QotherBxxInvBxy, CyyEyyVinvEyyDyxDxxInvBxxInvBxyCyy.T) * grad_r2))

            M = M + 2.0 * np.dot(Qother, CyyEyyVinvEyyDyxDxxInvBxxInv.T) * grad_r2

            # Contribution of Dyx done, now the contribution of DxxInv

            M = M + ((np.dot(-1.0 * Q, ByyInvByxCxxInvDxxInvDxyEyyVinvEyyDyxDxxInvCxxInvBxyByyInv) * grad_r2) +  \
                (np.dot(-1.0 * Q, ByyInvByxCxxInvDxxInvDxyEyyVinvEyyDyxDxxInvCxxInvBxyByyInv.T) * grad_r2))

            CxxDxxInvDxyEyyVinvEyyDyxDxxInvCxx = np.dot(EyyDyxDxxInvCxx.T, np.dot(Vinv, EyyDyxDxxInvCxx))
            CxxDxxInvDxyEyyVinvEyyDyxDxxInvCxxBxyByyInv = np.dot(np.dot(Bxy, ByyInv).T, CxxDxxInvDxyEyyVinvEyyDyxDxxInvCxx).T

            M = M + 2.0 * np.dot(Qother, CxxDxxInvDxyEyyVinvEyyDyxDxxInvCxxBxyByyInv) * grad_r2

            # The gradient of the squared distance has structure which we use here: d r2 / d_xij = 
            # 2 * (x_ij * t(d_i) %*% 1 - x_obs*j d_i^T) 
            # Importantly, the data has to be scaled by the lengthscale
            # We use that Tr(A * (B o cc^T)) = Tr((A o B) * cc^T), where o is the hadamard product

            colSums = np.dot(np.ones(M.shape[ 0 ]), M).reshape((cand.shape[ 0 ], 1))
            gradient2 = 2.0 * (colSums * cand / (task_model.params['ls'].value**2) - np.dot(M.T, \
                 task_model.inputs / task_model.params['ls'].value**2))

            # Now the second term of the gradient of Kx_test x (Kxx + Isigma^2) Kx x_test

            gradient = gradient + gradient2 

            # Next, the gradient of Koandpareto,x_test 

            r2 = cdist(other / task_model.params['ls'].value, cand / task_model.params['ls'].value, 'sqeuclidean')
            r = np.sqrt(r2)
            grad_r2 = (5.0/6.0) * np.exp(-np.sqrt(5.0) * r) * (1 + np.sqrt(5.0)*r) * task_model.params['amp2'].value * -1.0

            M = (-2.0 * np.dot(BxxInvBxy, CyyEyyVinvEyyCyy) * grad_r2)

            # Contribution of Dyy done, now the contribution of DyxDxxInvDxy. We use that Dyx = Cyx + Cte
            # where Cyx = Cxy.T = - (BxxInv Bxy Cyy).T = - Cyy Byx BxxInv and reuse the gradients of Dyy

            M = M + 2.0 * ((-1.0 * np.dot(BxxInvBxy, CyyEyyVinvEyyDyxDxxInvBxxInvBxyCyy) * grad_r2) + \
                (-1.0 * np.dot(BxxInvBxy, CyyEyyVinvEyyDyxDxxInvBxxInvBxyCyy.T) * grad_r2))

            M = M - 2.0 * CyyEyyVinvEyyDyxDxxInvBxxInv.T * grad_r2

            # Contribution of Dyx done, now the contribution of DxxInv

            M = M - 2.0 * CxxDxxInvDxyEyyVinvEyyDyxDxxInvCxxBxyByyInv * grad_r2

            # The gradient of the squared distance has structure which we use here: d r2 / d_xij = 
            # 2 * (x_ij * t(d_i) %*% 1 - x_obs*j d_i^T) 
            # Importantly, the data has to be scaled by the lengthscale
            # We use that Tr(A * (B o cc^T)) = Tr((A o B) * cc^T), where o is the hadamard product

            colSums = np.dot(np.ones(M.shape[ 0 ]), M).reshape((cand.shape[ 0 ], 1))
            gradient3 = 2.0 * (colSums * cand / (task_model.params['ls'].value**2) - np.dot(M.T, \
                other / task_model.params['ls'].value**2))

            # Now the second term of the gradient of Kx_test x (Kxx + Isigma^2) Kx x_test
            
            gradient = gradient + gradient3

            return 0.5 * gradient

        # This function computes the gradients for a particular task of the unconstrained entropy

	def compute_gradients_entropy_unconstrained_wrt_test_points_for_task(self, is_objective, task_model, task, cand, a):

                n_obs = a['n_obs']
                n_pset = a['n_pset']
                n_test = a['n_test']
                n_total = a['n_total']

                # Only works for Matern2 kernels since the gradients are optimzied for this kernel. 
                # Can be easily modified to include SquaredExp.

                assert task_model.options['kernel'] == "Matern52"

                if 'noise' not in task_model.params:
                    noise_level = 0.0
                else:
                    noise_level = task_model.params['noise'].value

                if is_objective:
                    Vinv = matrixInverse(a['Vpred'][ task ][ n_obs + n_pset : n_total, n_obs + n_pset : n_total  ] \
                        + np.eye(cand.shape[ 0 ]) * noise_level) 
                else:
                    Vinv = matrixInverse(a['Vpred_cons'][ task ][ n_obs + n_pset : n_total, n_obs + n_pset : n_total ] \
                        + np.eye(cand.shape[ 0 ]) * noise_level)

                from scipy.spatial.distance import cdist

                r2 = cdist(cand / task_model.params['ls'].value, cand / task_model.params['ls'].value, 'sqeuclidean')
                r = np.sqrt(r2)
                grad_r2 = (5.0/6.0) * np.exp(-np.sqrt(5.0) * r) * (1 + np.sqrt(5.0)*r) * task_model.params['amp2'].value * -1.0

                # The gradient of the squared distance has structure which we use here: d r2 / d_xij = 
                # 2 * (x_ij * t(d_i) %*% 1 - x*j d_i^T) + 2 * (transpose_of_prevous_thing)
                # Importantly, the data has to be scaled by the lengthscale
                # We use that Tr(A * (B o cc^T)) = Tr((A o B) * cc^T), where o is the hadamard product

                M = Vinv * grad_r2 

                colSums = np.dot(np.ones(M.shape[ 0 ]), M).reshape((cand.shape[ 0 ], 1))

                # We start with the gradient of Kx_test x_test

                gradient = 4.0 * (colSums * cand / (task_model.params['ls'].value**2) - np.dot(M, cand / task_model.params['ls'].value**2))

                #import pdb; pdb.set_trace();
                # Now the gradient of Kx_test x (Kxx + Isigma^2) Kx x_test

                import scipy.linalg as spla

                Q = spla.cho_solve((spla.cholesky(task_model.kernel.cov(task_model.inputs)), False), \
                    task_model.noiseless_kernel.cross_cov(task_model.inputs, cand))

                r2 = cdist(task_model.inputs / task_model.params['ls'].value, cand / task_model.params['ls'].value, 'sqeuclidean')
                r = np.sqrt(r2)
                grad_r2 = (5.0/6.0) * np.exp(-np.sqrt(5.0) * r) * (1 + np.sqrt(5.0)*r) * task_model.params['amp2'].value * -1.0

                M = (-1.0 * np.dot(Vinv, Q.T) * grad_r2.T).T

                # The gradient of the squared distance has structure which we use here: d r2 / d_xij = 
                # 2 * (x_ij * t(d_i) %*% 1 - x_obs*j d_i^T) 
                # Importantly, the data has to be scaled by the lengthscale
                # We use that Tr(A * (B o cc^T)) = Tr((A o B) * cc^T), where o is the hadamard product

                colSums = np.dot(np.ones(M.shape[ 0 ]), M).reshape((cand.shape[ 0 ], 1))
                gradient2 = 2.0 * (colSums * cand / (task_model.params['ls'].value**2) - np.dot(M.T, \
                    task_model.inputs / task_model.params['ls'].value**2))

                # Now the second term of the gradient of Kx_test x (Kxx + Isigma^2) Kx x_test
            
                gradient3 = gradient2.copy()

                return 0.5 * (gradient + gradient2 + gradient3)

	def compute_gradients_wrt_test_points(self, obj_model_dict, con_model_dict, pareto_set, tasks, cand, list_a):

                grads = dict()

                # We add all the gradients

                for i in range(len(list_a)):

	            for obj in obj_model_dict:

                        gradient_tmp = self.compute_gradients_entropy_unconstrained_wrt_test_points_for_task( \
                            True, obj_model_dict[ obj ], obj, cand, list_a[ str(i) ])

                        gradient_tmp_2 = self.compute_gradients_entropy_constrained_wrt_test_points_for_task( \
                            True, obj_model_dict[ obj ], obj, cand, list_a[ str(i) ])

                        if i == 0:
                            grads[ obj ] = gradient_tmp / len(list_a) - gradient_tmp_2 / len(list_a)
                        else:
                            grads[ obj ] += gradient_tmp / len(list_a) - gradient_tmp_2 / len(list_a)

	            for con in con_model_dict:

                        gradient_tmp = self.compute_gradients_entropy_unconstrained_wrt_test_points_for_task( \
                            False, con_model_dict[ con ], con, cand, list_a[ str(i) ])
            
                        gradient_tmp_2 = self.compute_gradients_entropy_constrained_wrt_test_points_for_task( \
                            False, con_model_dict[ con ], con, cand, list_a[ str(i) ])
 
                        if i == 0:
                            grads[ con ] = gradient_tmp / len(list_a) - gradient_tmp_2 / len(list_a)
                        else:
                            grads[ con ] += gradient_tmp / len(list_a) - gradient_tmp_2 / len(list_a)


                # We only sum the gradients correspnding to the tasks in tasks

                final_gradients = np.zeros(cand.shape)
                for task in tasks:
                    final_gradients += grads[ task ]

		return final_gradients

	def compute_ep(self, pareto_set, obj_model_dict, minimize, con_models_dict, key, cand, max_ep_iterations = 250):
		if cand.ndim == 1:
                	cand = cand[None]

		acq, unconstrainedVariances, constrainedVariances = self.compute_unconstrained_variances_and_init_acq_fun \
										(obj_model_dict, cand, con_models_dict)
                write_log('compute_unconstrained_variances_and_init_acq_fun PPESMOC', self.options['log_route_acq_funs'], \
                                {'acq' : acq, 'unconstrainedVariances' : unconstrainedVariances, 'constrainedVariances' : constrainedVariances})

		acq_dict, list_a = self.compute_and_evaluate_acquisition_function_all_factors(pareto_set, obj_model_dict, minimize, con_models_dict, \
					key, cand, self.options, acq, unconstrainedVariances, constrainedVariances, max_ep_iterations)

		return acq_dict, list_a 
	
	def compute_and_evaluate_acquisition_function_all_factors(self, pareto_set, obj_model_dict, minimize, con_models_dict, key, cand, \
									opt, acq, unconstrainedVariances, constrainedVariances, max_ep_iterations = 250):
		list_a = dict()
               	for i in range(int(self.options['pesm_samples_per_hyper'])):
                	if pareto_set[ str(i) ] is None:
				continue;
                        else:
                        	acq, list_a[ str(i) ] = self.ppesmoc_computation_for_pareto_set_sample(obj_model_dict, pareto_set[ str(i) ], \
								minimize, con_models_dict, self.input_space, cand, \
                                                		unconstrainedVariances, constrainedVariances, acq, opt, key, i, max_ep_iterations)
                                write_log('compute_PPESMOC_approximation PPESMOC', self.options['log_route_acq_funs'], {'acq' : acq})

		#Validate this index. I think that it enters in conflict with the previous index.
		#Anyway, it would be easy to see, because it is only a change in scale.

		for t in unconstrainedVariances:
       		        acq[t] /= float(self.options['pesm_samples_per_hyper'])

            	for t in acq:
                	if np.any(np.isnan(acq[t])):
	                    raise Exception("Acquisition function containts NaN for task %s" % t)
                write_log('BB acqs PPESMOC', self.options['log_route_acq_funs'], {'acq' : acq})
		return acq, list_a

	def compute_previous_operations_before_ep_algorithm(self, obj_model_dict, con_models_dict, pareto_set, cand):
			all_tasks = obj_model_dict.copy()
                        all_constraints= con_models_dict.copy()
                        # The old order of this set is pareto-obs. The new order must be obs-pareto-candidate.
                        # Why do I have to pass both obj and con model dicts and all tasks and constraints?
                        #X, n_obs, n_pset, n_test, n_total = build_set_of_points_that_conditions_GPs(obj_model_dict, con_model_dict, \
                        #       all_tasks, all_constraints, pareto_set, cand)
                        X, n_obs, n_pset, n_test, n_total = build_set_of_points_that_conditions_GPs(obj_model_dict, con_models_dict, \
                                pareto_set, cand)
                        write_log('build_set_of_points_that_conditions_GPs PPESMOC', self.options['log_route_acq_funs'], \
                                {'X' : X, 'n_obs' : n_obs, 'n_pset' : n_pset, 'n_test' : n_test, 'n_total' : n_total})
                        q = len(all_tasks)
                        c = len(all_constraints)

                        #Computation of predictive unconditional distributions of the GPs.
                        mPred, Vpred, cholVpred, VpredInv, cholKstarstar, mPred_cons, Vpred_cons, cholVpred_cons, VpredInv_cons, cholKstarstar_cons = \
                                build_unconditioned_predictive_distributions(all_tasks, all_constraints, X)
                        write_log('build_unconditioned_predictive_distributions PPESMOC', self.options['log_route_acq_funs'], \
                            {'mPred' : mPred, 'Vpred' : Vpred, 'VpredInv' : VpredInv, 'mPred_cons' : mPred_cons, \
                            'Vpred_cons' : Vpred_cons, 'VpredInv_cons' : VpredInv_cons})
                        jitter = get_jitter(obj_model_dict, con_models_dict)

                        # We create the posterior approximation. This needs info for both the factors dependant and not dependant on X.
                        # If we have computed the EP algorithm earlier
                        return create_data_structure_for_EP_computations_and_posterior_approximation(obj_model_dict, con_models_dict, n_obs, \
				n_pset, n_test, n_total, q, c, mPred, Vpred, VpredInv, cholKstarstar, mPred_cons, Vpred_cons, VpredInv_cons, \
				cholKstarstar_cons, jitter, X)

	def ppesmoc_computation_for_pareto_set_sample(self, obj_model_dict, pareto_set, minimize, con_models_dict, input_space, cand, \
								unconstrainedVariances, constrainedVariances, acq, opt, key, ps_index, max_ep_iterations = 250):
            dict_par_points = None
            if key in self.reuse_EP_solution:
                dict_par_points = self.reuse_EP_solution[ key ]

            a = self.compute_previous_operations_before_ep_algorithm(obj_model_dict, con_models_dict, pareto_set, cand)

            # We reuse the approximate factors if there is a previous EP solution
            if key in self.reuse_EP_solution:

                a_old = dict_par_points[ ps_index ]
                a['ahfhat'] = a_old['ahfhat'].copy()
                a['bhfhat'] = a_old['bhfhat'].copy() 
                a['chfhat'] = a_old['chfhat'].copy()   
                a['dhfhat'] = a_old['dhfhat'].copy() 
                a['ehfhat'] = a_old['ehfhat'].copy()   
                a['fhfhat'] = a_old['fhfhat'].copy()   
                a['ghfhat'] = a_old['ghfhat'].copy() * 0.0 # We set the previous parameters for test points to zero
                a['hhfhat'] =  a_old['hhfhat'].copy() * 0.0 # We set the previous parameters for test points to zero
                a['a_c_hfhat'] = a_old['a_c_hfhat'].copy()   
                a['b_c_hfhat'] = a_old['b_c_hfhat'].copy()   
                a['c_c_hfhat'] = a_old['c_c_hfhat'].copy()   
                a['d_c_hfhat'] = a_old['d_c_hfhat'].copy()   
                a['g_c_hfhat'] = a_old['g_c_hfhat'].copy() * 0.0 # We set the previous parameters for test points to zero
                a['h_c_hfhat'] = a_old['h_c_hfhat'].copy() * 0.0  # We set the previous parameters for test points to zero

                # We try to recompute the posterior. If that fails we start from scratch
                try:
                    import time
                    start = time.time()
                    a = self.update_full_marginals(copy.deepcopy(a))
                    end = time.time()
                    print('update marginals 1: ' + str(end-start) + ' seconds.')
                except:
                    print("Failed computation from EP previous solution")
                    a = self.compute_previous_operations_before_ep_algorithm(obj_model_dict, con_models_dict, pareto_set, cand)

            if opt['pesm_not_constrain_predictions'] == True:
                raise Exception("To have sense, parallel PESMOC needs to constraint predictions")
            else:
                predictionEP, a = self.ppesmoc_EP_computation(a, minimize, obj_model_dict, con_models_dict, key, ps_index, max_ep_iterations)
                """
                write_log('get_test_predictive_distributions', self.options['log_route_acq_funs'], \
                        {'mfs' : predictionEP['mf'], \
                         'vfs' : predictionEP['vf'], \
                         'mcs' : predictionEP['mc'], \
                         'vcs' : predictionEP['vc'], \
                         'unconstrainedVariances' : unconstrainedVariances})
                """

            return self.compute_PPESMOC_approximation(predictionEP, obj_model_dict, con_models_dict, \
                                                unconstrainedVariances, constrainedVariances, acq), a

        # This method is only for testing the gradients of the full acquisition function

        def test_full_acquisition_gradients(self, pareto_set, obj_model_dict, minimize, con_model_dict, key, tasks, cand, log, fun_log, fun_grad_autograd, func_autograd):

            for i in range(cand.shape[0]*cand.shape[1]):
                cand = np.random.random(cand.shape)

                # We run everything to guarantee that there is an EP solution already computed
                acq_dict, list_a = self.compute_ep(pareto_set, obj_model_dict, minimize, con_model_dict, key, cand)

                # No we will test the gradients of the acquisition function. We evaluate the aquisition without running EP
   
                # This only computes the unconstrainedVariances

                eps = cand * 0
                EPSILON = 1e-4
                if i==0:
                    eps[0,0] = EPSILON
                elif i==1:
                    eps[0,1] = EPSILON
                elif i==2:
                    eps[1,0] = EPSILON
                elif i==3:
                    eps[1,1] = EPSILON
                elif i==4:
                    eps[2,0] = EPSILON
                else:
                    eps[2,1] = EPSILON

                #Debug the result of the following functions in the log.
                acq_dict, list_a = self.compute_ep(pareto_set, obj_model_dict, minimize, con_model_dict, key, cand + eps, max_ep_iterations = 0)

                # by default, sum the PESC contribution for all tasks

                if tasks is None:
                    tasks = acq_dict.keys()
    
                # Compute the total acquisition function for the tasks of interests

                total_acq = 0.0
                for task in tasks:
                    total_acq += acq_dict[ task ]

                total_acq_ini = total_acq

                acq_dict, list_a = self.compute_ep(pareto_set, obj_model_dict, minimize, con_model_dict, key, cand - eps, max_ep_iterations = 0)

                # by default, sum the PESC contribution for all tasks

                if tasks is None:
                    tasks = acq_dict.keys()

                # Compute the total acquisition function for the tasks of interests

                total_acq = 0.0
                for task in tasks:
                    total_acq += acq_dict[ task ]

                total_acq_fini = total_acq

                #grad must be equal to self.compute_gradients_wrt_test_points(obj_model_dict, con_model_dict, pareto_set, tasks, cand, list_a)
                #grad = (total_acq_ini - total_acq_fini) / (2 * 1e-3)

                #f'(a) = (f(a+h)-f(a)) / h.
                """
                acq_dict_plus, list_a = self.compute_ep(pareto_set, obj_model_dict, minimize, con_model_dict, key, cand + eps, max_ep_iterations = 0)
                if tasks is None:
                    tasks = acq_dict_plus.keys()

                # Compute the total acquisition function for the tasks of interests

                total_acq_plus = 0.0
                for task in tasks:
                    total_acq_plus += acq_dict_plus[ task ]

                acq_dict_base, _ = self.compute_ep(pareto_set, obj_model_dict, minimize, con_model_dict, key, cand, max_ep_iterations = 0)
                if tasks is None:
                    tasks = acq_dict_base.keys()

                # Compute the total acquisition function for the tasks of interests

                total_acq_base = 0.0
                for task in tasks:
                    total_acq_base += acq_dict_base[ task ]

                #h = eps
                # fun_grad_autograd, func_autograd
                total_acq_plus = fun_grad(cand, obj_model_dict, con_model_dict, pareto_set, list_a, tasks, log, fun_log)
                """
                #Funciones de autograd.
                #total_acq_plus = func_autograd(cand.copy() + eps, obj_model_dict, con_model_dict, pareto_set, list_a, tasks, log, fun_log)
                #total_acq_ini
                #total_acq_base = func_autograd(cand.copy() - eps, obj_model_dict, con_model_dict, pareto_set, list_a, tasks, log, fun_log)
                #total_acq_fini
                #total_acq_plus = self.compute_ep(pareto_set, obj_model_dict, minimize, con_model_dict, key, cand + eps, max_ep_iterations = 0)
                #total_acq_base = self.compute_ep(pareto_set, obj_model_dict, minimize, con_model_dict, key, cand, max_ep_iterations = 0)
                #Definicion de derivada.
                grad = (total_acq_ini - total_acq_fini) / (2.0*EPSILON)
                #grad = (total_acq_plus - total_acq_base) / (2.0*EPSILON)
                autograds = fun_grad_autograd(cand.copy(), obj_model_dict, con_model_dict, pareto_set, list_a, tasks)#, log, fun_log)
                if i==0:
                    autograds = autograds[0,0]
                elif i==1:
                    autograds = autograds[0,1]
                elif i==2:
                    autograds = autograds[1,0]
                elif i==3:
                    autograds = autograds[1,1]
                elif i==4:
                    autograds = autograds[2,0]
                else:
                    autograds = autograds[2,1]
                import pdb; pdb.set_trace();
                assert np.abs(grad - autograds) < 1e-5

            return


	# Mix between perform_EP_algorithm(...) and predictEP_multiple_iter_optim_robust(...) methods.
	# All the information about the factors, points and distributions lies in a.
	def ppesmoc_EP_computation(self, a, minimize, all_tasks, all_constraints, key, ps_index, max_ep_iterations = 250):

		convergence = False
    		damping     = 0.5
    		iteration   = 0
                import time
                start = time.time()        
    		a = self.update_full_marginals(copy.deepcopy(a))
                end = time.time()
                print('update marginals 1: ' + str(end-start) + ' seconds.')
                """
                write_log('update_full_marginals', self.options['log_route_acq_funs'], \
                        {'tVinv_cons' : a['Vinv_cons'], \
                         'V_cons' : a['V_cons'], \
                         'm_nat_cons' : a['m_nat_cons'], \
                         'm_cons' : a['m_cons'], \
                         'Vinv' : a['Vinv'], \
                         'V' : a['V'], \
                         'm_nat' : a['m_nat'], \
                         'm' : a['m']
                        })
                """
    		aOld = copy.deepcopy(a)
		aOriginal = copy.deepcopy(a)

                # If we have a cached EP solution we run EP until convergence

                if not (key in self.reuse_EP_solution):
		    while not convergence and iteration < max_ep_iterations:
        		aNew, a = self.compute_full_updates_and_reduce_damping_if_fail(a, damping, minimize, aOld)
		        aOld = copy.deepcopy(a)
		        a = copy.deepcopy(aNew)
		        convergence = compute_convergence_criterion(a, convergence, all_tasks, iteration, aOld, damping, all_constraints)
		        damping   *= 0.99
		        iteration += 1

                # We save the EP solution after refining all non-test factors

		if key not in self.reuse_EP_solution:
                    self.reuse_EP_solution[key] = dict()
                    self.reuse_EP_solution[key][ps_index] = copy.deepcopy(a)

                # We process the factors for the test points only (just one time)
                start = time.time()
                a = update_full_Factors_only_test_factors(copy.deepcopy(a), 0.1, \
                                minimize = minimize, no_negative_variances_nor_nands = True)
                end = time.time()
                print('update full factors: ' + str(end-start) + ' seconds.')
                """
                write_log('update_full_Factors_only_test_factors', self.options['log_route_acq_funs'], \
                    {'a[ghfhat][ :, :, :, 0, 0 ]': a['ghfhat'][ :, :, :, 0, 0 ], \
                     'a[ghfhat][ :, :, :, 1, 1 ]': a['ghfhat'][ :, :, :, 1, 1 ], \
                     'a[ghfhat][ :, :, :, 0, 1 ]': a['ghfhat'][ :, :, :, 0, 1 ], \
                     'a[ghfhat][ :, :, :, 1, 0 ]': a['ghfhat'][ :, :, :, 1, 0 ], \
                     'a[hhfhat][ :, :, :, 0 ]': a['hhfhat'][ :, :, :, 0 ], \
                     'a[hhfhat][ :, :, :, 1 ]': a['hhfhat'][ :, :, :, 1 ], \
                     'a[g_c_hfhat][ :, :, : ]': a['g_c_hfhat'][ :, :, : ], \
                     'a[h_c_hfhat][ :, :, : ]': a['h_c_hfhat'][ :, :, : ], \
                     'a[m]': a['m'], 'a[V]': a['V'], 'a[m_cons]': a['m_cons'], 'a[V_cons]': a['V_cons']})
                """
                start = time.time()
                a = self.update_full_marginals(copy.deepcopy(a))
                end = time.time()
                print('update full marginals 2: ' + str(end-start) + ' seconds.')
                """
                write_log('update_full_marginals', self.options['log_route_acq_funs'], \
                        {'Vinv_cons' : a['Vinv_cons'], \
                         'V_cons' : a['V_cons'], \
                         'm_nat_cons' : a['m_nat_cons'], \
                         'm_cons' : a['m_cons'], \
                         'Vinv' : a['Vinv'], \
                         'V' : a['V'], \
                         'm_nat' : a['m_nat'], \
                         'm' : a['m']
                        })        
                """
		return self.get_test_predictive_distributions(a)

	def get_test_predictive_distributions(self, a):
		n_obs = a['n_obs']
		n_pset = a['n_pset']
		n_test = a['n_test']
		n_total = a['n_total']
		q = len(a['objs'])
		c = len(a['cons'])
		predictive_distributions = {
			'mf' : defaultdict(lambda: np.zeros(n_test)),
			'vf' : defaultdict(lambda: np.zeros((n_test, n_test))),
			'mc' : defaultdict(lambda: np.zeros(n_test)),
			'vc' : defaultdict(lambda: np.zeros((n_test, n_test))),
		}
		for obj in a['objs'].keys():
			predictive_distributions['mf'][ obj ] = a['m'][ obj ][ n_obs + n_pset : n_total ]
			predictive_distributions['vf'][ obj ] = a['V'][ obj ][ n_obs + n_pset : n_total , n_obs + n_pset : n_total ]
		for cons in a['cons'].keys():
			predictive_distributions['mc'][ cons ] = a['m_cons'][ cons ][ n_obs + n_pset : n_total ]
			predictive_distributions['vc'][ cons ] = a['V_cons'][ cons ][ n_obs + n_pset : n_total , n_obs + n_pset : n_total ]
		

		return predictive_distributions, a

	#Include test marginals.
	def update_full_marginals(self, a):
		n_obs = a['n_obs']
	        n_total = a['n_total']
	        n_pset = a['n_pset']
		n_test = a['n_test']
	        objectives = a['objs']
	        constraints = a['cons']
	        all_tasks = objectives
	        all_constraints = constraints
		n_objs = len(all_tasks)
		n_cons = len(all_constraints)
        	ntask = 0

	        # Updating constraint distribution marginals.
		#vTilde_back = defaultdict(lambda: np.zeros((n_total, n_total)))
		#vTilde_cons_back = defaultdict(lambda: np.zeros((n_total, n_total)))
                #import pdb; pdb.set_trace();
		for cons in all_constraints:

	                # Summing all the factors into the diagonal. Reescribir.

        	        vTilde_cons = np.zeros((n_total,n_total))
	                vTilde_cons[ np.eye(n_total).astype('bool') ] = np.append(np.append(np.sum(a['a_c_hfhat'][ :, : , ntask ], axis = 1), \
        	                np.sum(a['c_c_hfhat'][ :, : , ntask ], axis = 1) + a['ehfhat'][ :, ntask ]), \
				np.sum(a['g_c_hfhat'][ :, : , ntask ], axis = 1))

        	        mTilde_cons = np.append(np.append(np.sum(a['b_c_hfhat'][ :, : , ntask ], axis = 1), \
                	        np.sum(a['d_c_hfhat'][ :, : , ntask ], axis = 1) + a['fhfhat'][ :, ntask ]), \
				 np.sum(a['h_c_hfhat'][ :, : , ntask ], axis = 1))

	                # Natural parameter conversion and update of the marginal variance matrices.

        	        a['Vinv_cons'][cons] = a['VpredInv_cons'][cons] + vTilde_cons
	                a['V_cons'][cons] = matrixInverse(a['VpredInv_cons'][cons] + vTilde_cons)

        	        # Natural parameter conversion and update of the marginal mean vector.

                	a['m_nat_cons'][cons] = np.dot(a['VpredInv_cons'][cons], a['mPred_cons'][cons]) + mTilde_cons
	                a['m_cons'][cons] = np.dot(a['V_cons'][cons], a['m_nat_cons'][ cons ])

	                ntask = ntask + 1
		ntask = 0
	        for obj in all_tasks:

        	        vTilde = np.zeros((n_total,n_total))

                	vTilde[ np.eye(n_total).astype('bool') ] = np.append(np.append(np.sum(a['ahfhat'][ :, : , ntask, 0, 0 ], axis = 1), \
	                        np.sum(a['ahfhat'][ :, : , ntask, 1, 1 ], axis = 0) + np.sum(a['chfhat'][ :, : , ntask, 0, 0 ], axis = 1) + \
	                        np.sum(a['chfhat'][ :, : , ntask, 1, 1 ], axis = 0) + np.sum(a['ghfhat'][ :, : , ntask, 1, 1 ], axis = 0)), \
				np.sum(a['ghfhat'][ :, : , ntask, 0, 0 ], axis = 1))

        	        vTilde[ n_obs : n_obs + n_pset, n_obs : n_obs + n_pset ] = vTilde[ n_obs : n_obs + n_pset, n_obs : n_obs + n_pset ] + \
	                        a['chfhat'][ :, : , ntask, 0, 1 ] + a['chfhat'][ :, : , ntask, 1, 0 ].T

        	        vTilde[ 0 : n_obs, n_obs : n_obs + n_pset ] = a['ahfhat'][ :, :, ntask, 0, 1]
	                vTilde[ n_obs : n_obs + n_pset, 0 : n_obs ] =  a['ahfhat'][ :, :, ntask, 0, 1].transpose()

			vTilde[ n_obs + n_pset : n_total, n_obs : n_obs + n_pset ] = a['ghfhat'][ :, :, ntask, 0, 1]
	                vTilde[ n_obs : n_obs + n_pset, n_obs + n_pset : n_total ] =  a['ghfhat'][ :, :, ntask, 0, 1].transpose()

        	        a['Vinv'][obj] = a['VpredInv'][obj] + vTilde
	                a['V'][obj] = matrixInverse(a['VpredInv'][obj] + vTilde)

                	mTilde = np.append(np.append(np.sum(a['bhfhat'][ :, : , ntask, 0 ], axis = 1),
                        	np.sum(a['bhfhat'][ :, : , ntask, 1 ], axis = 0) + np.sum(a['hhfhat'][ :, : , ntask, 1 ], axis = 0) +\
	                        np.sum(a['dhfhat'][ :, : , ntask, 0 ], axis = 1) + np.sum(a['dhfhat'][ :, : , ntask, 1 ], axis = 0)), \
				np.sum(a['hhfhat'][ :, : , ntask, 0 ], axis = 1))

	                a['m_nat'][obj] = np.dot(a['VpredInv'][obj], a['mPred'][obj]) + mTilde
        	        a['m'][obj] = np.dot(a['V'][obj], a['m_nat'][ obj ])

                	ntask = ntask + 1

		return a

	def compute_full_updates_and_reduce_damping_if_fail(self, a, damping, minimize, aOld):
		update_correct = False
	        damping_inner = damping
	        fail = False
	        second_update = False

	        while update_correct == False:

	            error = False

	            try:

	                aNew = update_full_Factors_no_test_factors(copy.deepcopy(a), damping_inner, \
                            minimize = minimize, no_negative_variances_nor_nands = False)
        	        aNew = self.update_full_marginals(copy.deepcopy(aNew))

	            except npla.linalg.LinAlgError as e:
	                error = True

	            if error == False:
	                if fail == True and second_update == False:
        	                a = aNew.copy()
	                        second_update = True
        	        else:
	                        update_correct = True
        	    else:

	                a = copy.deepcopy(aOld)
        	        damping_inner = damping_inner * 0.5
	                fail = True
	                second_update = False
        	        print 'Reducing damping factor to guarantee EP update! Damping: %f' % (damping_inner)

                        # If damping inner is < 1e-5 we start from scratch:

                        if damping_inner < 1e-5:
                            a['ahfhat'] *= 0
                            a['bhfhat'] *= 0 
                            a['chfhat'] *= 0   
                            a['dhfhat'] *= 0 
                            a['ehfhat'] *= 0   
                            a['fhfhat'] *= 0   
                            a['ghfhat'] *= 0   
                            a['hhfhat'] *= 0 
                            a['g_c_hfhat'] *= 0   
                            a['h_c_hfhat'] *= 0   
                            a['a_c_hfhat'] *= 0   
                            a['b_c_hfhat'] *= 0   
                            a['c_c_hfhat'] *= 0   
                            a['d_c_hfhat'] *= 0   

	        return aNew, a
		
	######################################################################################################################
	# compute_unconstrained_variances_and_init_acq_fun: Computes the unconstrained variances necessary for PPESMOC and inits
	# the data structure that hold the acquistion function.
	#
	# INPUT:
	# obj_models_dict: GPs of the objectives.
	# con_models: GPs of the constraints.
	# cand: Candidate/s point/s.
	#
	# OUTPUT:
	# acq: Data structure that will hold PPESMOC.
	# unconstrainedVariances: Covariance matrices that result by the prediction of the candidate point plus noise.
	# constrainedVariances: Data structure that will hold the constrainedVariances. 
	######################################################################################################################
	def compute_unconstrained_variances_and_init_acq_fun(self, obj_models_dict, cand, con_models):
	    unconstrainedVariances = dict()
	    constrainedVariances = dict()
	    acq = dict()

	    for obj in obj_models_dict:
	        unconstrainedVariances[ obj ] = obj_models_dict[ obj ].predict(cand, full_cov=True)[ 1 ]
		np.fill_diagonal(unconstrainedVariances[ obj ], np.diagonal(unconstrainedVariances[ obj ]) + obj_models_dict[ obj ].noise_value())

	    for cons in con_models:
	        unconstrainedVariances[ cons ] = con_models[ cons ].predict(cand, full_cov=True)[ 1 ]
		np.fill_diagonal(unconstrainedVariances[ cons ], np.diagonal(unconstrainedVariances[ cons ]) + con_models[ cons ].noise_value())

	    for t in unconstrainedVariances:
	        acq[t] = 0

	    return acq, unconstrainedVariances, constrainedVariances

	######################################################################################################################
	# evaluate_acquisition_function_for_sample: Computes non dependant PPESMOC factors and then evaluates PPESMOC for 
	# a Pareto set sample.
	#
	# INPUT:
	# i: Index of the Pareto Set and epSolution sample.
	# epSolution: Data structure with CPDs, PDs and EP factors not dependant on the candidate points.
	# obj_models_dict: GPs of the objectives.
	# con_models: GPs of the constraints.
	# pareto_set: Dictionary of pareto sets.
	# cand: Candidate/s point/s.
	# minimize: If FALSE, it maximizes.
	# unconstrainedVariances: Covariance matrices that result by the prediction of the candidate point plus noise.
	# constrainedVariances: Data structure that will hold the constrainedVariances.
	# acq: Data structure that will hold PPESMOC.
	# OUTPUT:
	# acq: PPESMOC acquisition function approximated by EP factors.
	######################################################################################################################
	def evaluate_acquisition_function_for_sample(self, i, epSolution, obj_models_dict, con_models, pareto_set, cand, minimize,\
						unconstrainedVariances, constrainedVariances, acq, opt):
            
        	# We check if we have to constrain the predictions or not

	        if opt['pesm_not_constrain_predictions'] == True:
	            predictionEP = predictEP_unconditioned(obj_models_dict, con_models, epSolution[ str(i) ], pareto_set[ str(i) ], cand)
        	else:
	            predictionEP = predictEP_multiple_iter_optim_robust(obj_models_dict, con_models, epSolution[ str(i) ], pareto_set[ str(i) ], cand,  \
                        n_iters = 1, damping = .1, no_negatives = True, minimize = minimize)
	
		return self.compute_PPESMOC_approximation(predictionEP, obj_models_dict, con_models, \
						unconstrainedVariances, constrainedVariances, acq)

	#TODO TODO TODO:
	#This will need to be changed according to the PPESMOC notes, also, predictions will have another shape.
	#Are now constrainedVariances the diag of the actual ones or unconstrainedVariances must be the matrix? 
	#PPESMOC Notes: It is said that we need the matrices, so it is case 2: unconstrainedVariances must be the matrix.
	#How do I extract these matrices? look at build_unconditioned_predictive_distributions, predict on cand.
	#DONE.
	#After having extracted that quantity, equation (2) must be implemented here.
	#It is equation (5) minus the same equation for the CPD computed by adding and refining EP factors that
	#condition the PDs.
	#With that PPESMOC will be done and now it remains to code the gradient of PPESMOC w.r.t points.

	def compute_PPESMOC_approximation(self, predictionEP, obj_models_dict, con_models, unconstrainedVariances, constrainedVariances, acq):

                predictionEP_obj = predictionEP[ 'vf' ]
                predictionEP_cons = predictionEP[ 'vc' ]

                # DHL changed fill_diag, because that was updating the a structure and screwing things up later on

                for obj in obj_models_dict:
                    predictionEP_obj[ obj ] = predictionEP_obj[ obj ] + np.eye(predictionEP_obj[ obj ].shape[ 0 ]) * obj_models_dict[ obj ].noise_value()
                    constrainedVariances[ obj ] = predictionEP_obj[ obj ]

                for cons in con_models:
		    predictionEP_cons[ cons ] = predictionEP_cons[ cons ] + np.eye(predictionEP_cons[ obj ].shape[ 0 ]) * con_models[ cons ].noise_value()
                    constrainedVariances[ cons ] = predictionEP_cons[ cons ]

                # We only care about the variances because the means do not affect the entropy
		# The summation of the acq of the tasks (t) is done in a higher method. Do no do it here.

                for t in unconstrainedVariances:

                    # DHL replaced np.log(np.linalg.det()) to avoid precision errors

		    value = 0.5 * np.linalg.slogdet(unconstrainedVariances[t])[ 1 ] - 0.5 * np.linalg.slogdet(constrainedVariances[t])[ 1 ]

                    # We set negative values of the acquisition function to zero  because the 
                    # entropy cannot be increased when conditioning

		    value = np.max(value, 0)

                    acq[t] += value

                return acq


	######################################################################################################################
	# evaluate_acquisition_function_given_EP_solution: Given the CPD by non dependant on the candidate factors, it 
	# conditions the CPD to the dependant factors and it evaluates the PPESMOC acquisition function, computing the 
	# unconstrained variances and the constrained variances and applying the PPESMOC expression.
	#
	# INPUT:
	# obj_models_dict: GPs of the objectives.
	# cand: Candidate set of points.
	# epSolution: Data structure with CPDs, PDs and EP factors not dependant on the candidate points.
	# pareto_set: Dictionary of pareto_sets.
	# minimize: If FALSE, it maximizes.
	# opt: Options of the BO experiment.
	# con_models: GPs of the constraints.
	# OUTPUT:
	# acq: PPESMOC acquisition function approximated by EP factors by tasks.
	######################################################################################################################
	def evaluate_acquisition_function_given_EP_solution(self, obj_models_dict, cand, epSolution, pareto_set, minimize=True, opt = None, con_models = {}, acq = {}, unconstrainedVariances = {}, constrainedVariances = {}):

	    # We then evaluate the constrained variances
	    # As it is passed by parameter and returned, self.evaluate_acquisition_function_for_sample sums the previous acq to the current one.
	    for i in range(len(epSolution)):
		if epSolution[ str(i) ] is None:
                        continue
		acq = self.evaluate_acquisition_function_for_sample(i, epSolution, obj_models_dict, con_models, pareto_set, cand, minimize,\
                                                unconstrainedVariances, constrainedVariances, acq, opt)

	    for t in unconstrainedVariances:
	        acq[t] /= len(epSolution)

	    for t in acq:
	        if np.any(np.isnan(acq[t])):
	            raise Exception("Acquisition function containts NaN for task %s" % t)

	    return acq

def test_random_features_sampling():

    D = 2
    N = 12

    np.random.seed(2)
    
    inputs  = npr.rand(N,D)
    # W       = npr.randn(D,1)
    # vals    = np.dot(inputs**2, W).flatten() + np.sqrt(1e-3)*npr.randn(N)
    # vals = npr.randn(N)   
    vals = np.sin(np.sum(inputs,axis=1)*7.0)*np.sum(inputs,axis=1)+npr.randn(N)*0.1

    options = dict()
    options['likelihood'] = 'noiseless'

    beta_opt = dict()
    beta_opt['BetaWarp'] = {}
    ignore = dict()
    ignore['IgnoreDims'] = {'to_ignore': [ 1 ]}
    options['transformations'] = [ beta_opt, ignore ]
    options['transformations'] = [ ignore ]
    options['stability_jitter'] = 1e-10
    options['kernel'] = 'SquaredExp'
    options['fit_mean'] = False

#    gp = GP(D, kernel='SquaredExp', likelihood='noiseless', fit_mean = False, stability_jitter=1e-10)
    gp = GP(D, **options)
#    gp.fit(inputs, vals, fit_hypers=False)
    gp.fit(inputs, vals, fit_hypers=True)
    gp.set_state(9)

    print 'ls=%s' % str(gp.params['ls'].value)
    print 'noise=%f' % float(gp.noise_value())
    print 'amp2=%f' % float(gp.params['amp2'].value)

    """
    Test the function sample_gp_with_random_features by taking the dot product
    of the random cosine functions and comparing them to the kernel...
    Right, because these are like the finite feature space, whereas the kernel is
    like an infinite feature space. So as the number of features grows the result
    should approach the kernel
    """
    num_test_inputs = 20
    test_input_1 = 5*npr.randn(num_test_inputs,D)
    test_input_2 = 5*npr.randn(num_test_inputs,D)
    # print test_input_1
    # print test_input_2
#    K = gp.scaled_input_kernel.cross_cov(test_input_1, test_input_2)
    K = gp.noiseless_kernel.cross_cov(test_input_1, test_input_2)

    print 'Error between the real coveraiance matrix and the approximated covariance matrix'
    nmax = 5
    for log_nFeatures in np.arange(0,nmax+1):
        tst_fun = sample_gp_with_random_features(gp, nFeatures=10**log_nFeatures, testing=True)
        this_should_be_like_K = np.dot(tst_fun(test_input_1).T, tst_fun(test_input_2))
        # print '%f, %f' % (K, this_should_be_like_K)
        print 'nFeatures = 10^%d, average absolute error = %f' % (log_nFeatures, np.mean(np.abs(K-this_should_be_like_K)))

    # The above test is good for the random features. But we should also test theta somehow. 
    print 'difference between predicted mean at the inputs and the true values (should be 0 if noiseless): %f' % np.mean(np.abs(gp.predict(inputs)[0]-vals))
    print 'Error between the predicted mean using the GP approximation, and the true values'
    for log_nFeatures in np.arange(0,nmax+1):
        wrapper = sample_gp_with_random_features(gp, nFeatures=10**log_nFeatures)
        print 'nFeatures = 10^%d, error on true values = %f' % (log_nFeatures, np.mean(np.abs(vals-wrapper(inputs, gradient=False))))
        # print 'True values: %s' % str(vals)
        # print 'Approximated values: %s' % str(wrapper(inputs, gradient=False))

    # print 'at test, sampled val = %s' % wrapper(inputs[0][None], gradient=False)
    # print 'at test, mean=%f,var=%f' % gp.predict(inputs[0][None])



    # Now test the mean and covariance at some test points?
    test = npr.randn(2, D)
    # test[1,:] = test[0,:]+npr.randn(1,D)*0.2

    m, cv = gp.predict(test, full_cov=True)
    print 'true mean = %s' % m
    print 'true cov = \n%s' % cv

    n_samples = int(1e4)
    samples = gp.sample_from_posterior_given_hypers_and_data(test, n_samples=n_samples, joint=True)
    true_mean = np.mean(samples, axis=1)
    true_cov = np.cov(samples)
    print ''
    print 'mean of %d gp samples = %s' % (n_samples, true_mean)
    print 'cov of %d gp samples = \n%s' % (n_samples, true_cov)

    import sys
    approx_samples = 0.0*samples
    for i in xrange(n_samples):
        if i % (n_samples/100) == 0:
            sys.stdout.write('%02d%% ' % (i/((n_samples/100))))
            sys.stdout.flush()
#        wrapper = sample_gp_with_random_features(gp, nFeatures=10000, use_woodbury_if_faster=True)
        wrapper = sample_gp_with_random_features(gp, nFeatures=10000)
        samples[:,i] = np.array(wrapper(test, gradient=False)).T

    approx_mean = np.mean(samples, axis=1)
    approx_cov = np.cov(samples)

    print ''
    print 'mean of %d approx samples = %s' % (n_samples, approx_mean)
    print 'cov of %d approx samples = \n%s' % (n_samples, approx_cov)

    print ''
    print 'error of true means = %s' % np.sum(np.abs(true_mean-m))
    print 'error of true covs = %s' % np.sum(np.abs(true_cov-cv))
    print 'error of approx means = %s' % np.sum(np.abs(approx_mean-m))
    print 'error of approx covs = %s' % np.sum(np.abs(approx_cov-cv))


def test_pareto_set_sampling():

    D = 1
    N = 12
    
    inputs  = npr.rand(N,D)
    # W       = npr.randn(D,1)
    # vals    = np.dot(inputs**2, W).flatten() + np.sqrt(1e-3)*npr.randn(N)
    # vals = npr.randn(N)   
    vals1 = np.sin(np.sum(inputs,axis=1)*7.0)*np.sum(inputs,axis=1)+npr.randn(N)*0.1
    vals2 = np.cos(np.sum(inputs,axis=1)*7.0)*np.sum(inputs,axis=1)+npr.randn(N)*0.1
    objective1 = GP(D)#, kernel='SquaredExp')#, likelihood='noiseless')
    objective2 = GP(D)#, kernel='SquaredExp')#, likelihood='noiseless')
    objective1.fit(inputs, vals1, fit_hypers=False)
    objective2.fit(inputs, vals2, fit_hypers=False)

    print 'ls=%s' % str(objective1.params['ls'].value)
    print 'noise=%f' % float(objective1.params['noise'].value)
    print 'amp2=%f' % float(objective1.params['amp2'].value)

    print '\n'

    print 'ls=%s' % str(objective2.params['ls'].value)
    print 'noise=%f' % float(objective2.params['noise'].value)
    print 'amp2=%f' % float(objective2.params['amp2'].value)

    objectives_dict = dict()

    objectives_dict['f1'] = objective1
    objectives_dict['f2'] = objective2
    
    pareto_set = sample_solution(1, objectives_dict.values())

    gp_samples = dict()
    gp_samples['objectives'] = [ sample_gp_with_random_features(objective_gp, NUM_RANDOM_FEATURES) for objective_gp in objectives_dict.values() ]
    funs = gp_samples['objectives']


    moo = MOOP_basis_functions(funs, 1)

    moo.evolve(100, 100)

    result = moo.compute_pareto_front_and_set_summary(20)

    size = result['pareto_set'].shape[ 0 ]
    subset = np.random.choice(range(size), min(size, PARETO_SET_SIZE), replace = False)
	
    pareto_set = result['pareto_set'][ subset, ]
    front = result['frontier'][ subset, ]

    moo.pop.plot_pareto_fronts()

    print 'plotting'

    if D == 1:
        import matplotlib.pyplot as plt
        spacing = np.linspace(0,1,1000)[:,None]

        plt.figure()
        plt.plot(inputs, vals1, color='r', marker='o', markersize=10, linestyle='None')
        plt.plot(inputs, vals2, color='g', marker='x', markersize=10, linestyle='None')
        plt.plot(pareto_set[:,0], np.ones(len(pareto_set[:,0])) * np.mean(vals1), 'b.')
        plt.plot(spacing, funs[ 0 ](spacing, False), 'r.')
        plt.plot(spacing, funs[ 1 ](spacing, False), 'g.')
        plt.show()
        plt.figure()
        plt.plot(funs[ 0 ](spacing, False), funs[ 1 ](spacing, False), 'b.', marker = 'o')
        plt.plot(front[:,0], front[:,1], 'r.', marker = 'x')
        plt.show()

# Test the predictive distribution given a pareto set

def test_conditioning():

	np.random.seed(1)

	D = 1
	N = 5
    
	inputs  = npr.rand(N,D)
	vals1 = np.sin(np.sum(inputs,axis=1)*7.0)*np.sum(inputs,axis=1)+npr.randn(N)*0.1
	vals2 = np.cos(np.sum(inputs,axis=1)*7.0)*np.sum(inputs,axis=1)+npr.randn(N)*0.1
	objective1 = GP(D)#, kernel='SquaredExp')#, likelihood='noiseless')
	objective2 = GP(D)#, kernel='SquaredExp')#, likelihood='noiseless')
	objective1.fit(inputs, vals1, fit_hypers = False)
	objective2.fit(inputs, vals2, fit_hypers = False)

	print 'ls=%s' % str(objective1.params['ls'].value)
	print 'noise=%f' % float(objective1.params['noise'].value)
	print 'amp2=%f' % float(objective1.params['amp2'].value)

	print '\n'

	print 'ls=%s' % str(objective2.params['ls'].value)
	print 'noise=%f' % float(objective2.params['noise'].value)
	print 'amp2=%f' % float(objective2.params['amp2'].value)

	objectives_dict = dict()

	objectives_dict['f1'] = objective1
	objectives_dict['f2'] = objective2

	gp_samples = dict()
	gp_samples['objectives'] = [ sample_gp_with_random_features(objective_gp, NUM_RANDOM_FEATURES) for objective_gp in objectives_dict.values() ]
	funs = gp_samples['objectives']

	moo = MOOP_basis_functions(funs, 1)

	moo.evolve(100, 100)

	result = moo.compute_pareto_front_and_set_summary(10)

	pareto_set = result['pareto_set']
	front = result['frontier']

	moo.pop.plot_pareto_fronts()

        import matplotlib.pyplot as plt
        spacing = np.linspace(0,1,1000)[:,None]

	mean1 = objective1.predict(spacing)[0]
	mean2 = objective2.predict(spacing)[0]
	var1 = objective1.predict(spacing)[1]
	var2 = objective2.predict(spacing)[1]

        plt.figure()
        plt.plot(inputs, vals1, color='r', marker='o', markersize=10, linestyle='None')
        plt.plot(inputs, vals2, color='g', marker='x', markersize=10, linestyle='None')
	plt.plot(spacing, mean1, 'r.')
	plt.plot(spacing, mean1 + np.sqrt(var1), color = 'r', marker = '.', markersize = 1)
	plt.plot(spacing, mean1 - np.sqrt(var1), color = 'r', marker = '.', markersize = 1)
        plt.plot(spacing, mean2, 'g.')
	plt.plot(spacing, mean2 + np.sqrt(var2), color = 'g', marker = '.', markersize = 1)
	plt.plot(spacing, mean2 - np.sqrt(var2), color = 'g', marker = '.', markersize = 1)
        plt.show()
        plt.figure()
        plt.plot(inputs, vals1, color='r', marker='o', markersize=10, linestyle='None')
        plt.plot(inputs, vals2, color='g', marker='x', markersize=10, linestyle='None')
	plt.plot(spacing, funs[ 0 ](spacing, False), 'r.')
        plt.plot(spacing, funs[ 1 ](spacing, False), 'g.')
        plt.plot(pareto_set[:,0], np.ones(len(pareto_set[:,0])) * 0, 'b.')
        plt.show()
        plt.figure()
        plt.plot(funs[ 0 ](spacing, False), funs[ 1 ](spacing, False), 'b.', marker = 'o')
        plt.plot(front[:,0], front[:,1], 'r.', marker = 'x')
        plt.show()

#	pareto_set = np.zeros((3, 1))
#	pareto_set[ 0, 0 ] = 0.5
#	pareto_set[ 1, 0 ] = 0.65
#	pareto_set[ 2, 0 ] = 0.85
	
	epSolution = ep(objectives_dict, pareto_set, minimize=True)

	ret = predictEP_multiple_iter_optim(objectives_dict, epSolution, pareto_set, spacing, n_iters = 1, damping = .5, no_negatives = True)
        plt.figure()
        plt.plot(inputs, vals1, color='r', marker='o', markersize=10, linestyle='None')
        plt.plot(inputs, vals2, color='g', marker='x', markersize=10, linestyle='None')
	plt.plot(spacing, ret['mf']['f1'], 'r.')
	plt.plot(spacing, ret['mf']['f1'] + np.sqrt(ret['vf']['f1']), color = 'r', marker = '.', markersize = 1)
	plt.plot(spacing, ret['mf']['f1'] - np.sqrt(ret['vf']['f1']), color = 'r', marker = '.', markersize = 1)
	plt.plot(spacing, ret['mf']['f2'], 'g.')
	plt.plot(spacing, ret['mf']['f2'] + np.sqrt(ret['vf']['f2']), color = 'g', marker = '.', markersize = 1)
	plt.plot(spacing, ret['mf']['f2'] - np.sqrt(ret['vf']['f2']), color = 'g', marker = '.', markersize = 1)
        plt.plot(pareto_set[:,0], np.ones(len(pareto_set[:,0])) * 0, 'r.')
        plt.show()

	ret = predictEP_adf(objectives_dict, epSolution, pareto_set, spacing)
        plt.figure()
        plt.plot(inputs, vals1, color='r', marker='o', markersize=10, linestyle='None')
        plt.plot(inputs, vals2, color='g', marker='x', markersize=10, linestyle='None')
	plt.plot(spacing, ret['mf']['f1'], 'r.')
	plt.plot(spacing, ret['mf']['f1'] + np.sqrt(ret['vf']['f1']), color = 'r', marker = '.', markersize = 1)
	plt.plot(spacing, ret['mf']['f1'] - np.sqrt(ret['vf']['f1']), color = 'r', marker = '.', markersize = 1)
	plt.plot(spacing, ret['mf']['f2'], 'g.')
	plt.plot(spacing, ret['mf']['f2'] + np.sqrt(ret['vf']['f2']), color = 'g', marker = '.', markersize = 1)
	plt.plot(spacing, ret['mf']['f2'] - np.sqrt(ret['vf']['f2']), color = 'g', marker = '.', markersize = 1)
        plt.plot(pareto_set[:,0], np.ones(len(pareto_set[:,0])) * 0, 'g.')
        plt.show()

	ret = predictEP_unconditioned(objectives_dict, epSolution, pareto_set, spacing)
        plt.figure()
        plt.plot(inputs, vals1, color='r', marker='o', markersize=10, linestyle='None')
        plt.plot(inputs, vals2, color='g', marker='x', markersize=10, linestyle='None')
	plt.plot(spacing, ret['mf']['f1'], 'r.')
	plt.plot(spacing, ret['mf']['f1'] + np.sqrt(ret['vf']['f1']), color = 'r', marker = '.', markersize = 1)
	plt.plot(spacing, ret['mf']['f1'] - np.sqrt(ret['vf']['f1']), color = 'r', marker = '.', markersize = 1)
	plt.plot(spacing, ret['mf']['f2'], 'g.')
	plt.plot(spacing, ret['mf']['f2'] + np.sqrt(ret['vf']['f2']), color = 'g', marker = '.', markersize = 1)
	plt.plot(spacing, ret['mf']['f2'] - np.sqrt(ret['vf']['f2']), color = 'g', marker = '.', markersize = 1)
        plt.plot(pareto_set[:,0], np.ones(len(pareto_set[:,0])) * 0, 'b.')
        plt.show()

	#import pdb; pdb.set_trace()


# Test the predictive distribution given a pareto set

def test_predictive():

	np.random.seed(1)

	D = 1
	N = 10
    
	inputs  = npr.rand(N,D)
	vals1 = np.sin(np.sum(inputs,axis=1)*7.0)*np.sum(inputs,axis=1)+npr.randn(N)*0.1
	vals2 = np.cos(np.sum(inputs,axis=1)*7.0)*np.sum(inputs,axis=1)+npr.randn(N)*0.1
	objective1 = GP(D)#, kernel='SquaredExp')#, likelihood='noiseless')
	objective2 = GP(D)#, kernel='SquaredExp')#, likelihood='noiseless')
	objective1.fit(inputs, vals1, fit_hypers = False)
	objective2.fit(inputs, vals2, fit_hypers = False)

	print 'ls=%s' % str(objective1.params['ls'].value)
	print 'noise=%f' % float(objective1.params['noise'].value)
	print 'amp2=%f' % float(objective1.params['amp2'].value)

	print '\n'

	print 'ls=%s' % str(objective2.params['ls'].value)
	print 'noise=%f' % float(objective2.params['noise'].value)
	print 'amp2=%f' % float(objective2.params['amp2'].value)

	objectives_dict = dict()

	objectives_dict['f1'] = objective1
	objectives_dict['f2'] = objective2

	gp_samples = dict()
	gp_samples['objectives'] = [ sample_gp_with_random_features(objective_gp, NUM_RANDOM_FEATURES) for objective_gp in objectives_dict.values() ]
	funs = gp_samples['objectives']

	moo = MOOP_basis_functions(funs, 1)

	moo.evolve(100, 100)

	result = moo.compute_pareto_front_and_set_summary(3)

	pareto_set = result['pareto_set']
	front = result['frontier']

	moo.pop.plot_pareto_fronts()

        import matplotlib.pyplot as plt
        spacing = np.linspace(0,1,1000)[:,None]

	mean1 = objective1.predict(spacing)[0]
	mean2 = objective2.predict(spacing)[0]
	var1 = objective1.predict(spacing)[1]
	var2 = objective2.predict(spacing)[1]

        plt.figure()
        plt.plot(inputs, vals1, color='r', marker='o', markersize=10, linestyle='None')
        plt.plot(inputs, vals2, color='g', marker='x', markersize=10, linestyle='None')
	plt.plot(spacing, mean1, 'r.')
	plt.plot(spacing, mean1 + np.sqrt(var1), color = 'r', marker = '.', markersize = 1)
	plt.plot(spacing, mean1 - np.sqrt(var1), color = 'r', marker = '.', markersize = 1)
        plt.plot(spacing, mean2, 'g.')
	plt.plot(spacing, mean2 + np.sqrt(var2), color = 'g', marker = '.', markersize = 1)
	plt.plot(spacing, mean2 - np.sqrt(var2), color = 'g', marker = '.', markersize = 1)
        plt.show()
        plt.figure()
        plt.plot(inputs, vals1, color='r', marker='o', markersize=10, linestyle='None')
        plt.plot(inputs, vals2, color='g', marker='x', markersize=10, linestyle='None')
	plt.plot(spacing, funs[ 0 ](spacing, False), 'r.')
        plt.plot(spacing, funs[ 1 ](spacing, False), 'g.')
        plt.plot(pareto_set[:,0], np.ones(len(pareto_set[:,0])) * 0, 'b.')
        plt.show()
        plt.figure()
        plt.plot(funs[ 0 ](spacing, False), funs[ 1 ](spacing, False), 'b.', marker = 'o')
        plt.plot(front[:,0], front[:,1], 'r.', marker = 'x')
        plt.show()

	pareto_set = np.zeros((3, 1))
	pareto_set[ 0, 0 ] = 0.5
	pareto_set[ 1, 0 ] = 0.65
	pareto_set[ 2, 0 ] = 0.85
	
	epSolution = ep(objectives_dict, pareto_set, minimize=True)

	ret = predictEP_multiple_iter_optim(objectives_dict, epSolution, pareto_set, spacing, n_iters = 1, damping = .5, no_negatives = True)
        plt.figure()
        plt.plot(inputs, vals1, color='r', marker='o', markersize=10, linestyle='None')
        plt.plot(inputs, vals2, color='g', marker='x', markersize=10, linestyle='None')
	plt.plot(spacing, ret['mf']['f1'], 'r.')
	plt.plot(spacing, ret['mf']['f1'] + np.sqrt(ret['vf']['f1']), color = 'r', marker = '.', markersize = 1)
	plt.plot(spacing, ret['mf']['f1'] - np.sqrt(ret['vf']['f1']), color = 'r', marker = '.', markersize = 1)
	plt.plot(spacing, ret['mf']['f2'], 'g.')
	plt.plot(spacing, ret['mf']['f2'] + np.sqrt(ret['vf']['f2']), color = 'g', marker = '.', markersize = 1)
	plt.plot(spacing, ret['mf']['f2'] - np.sqrt(ret['vf']['f2']), color = 'g', marker = '.', markersize = 1)
        plt.plot(pareto_set[:,0], np.ones(len(pareto_set[:,0])) * 0, 'r.')
        plt.show()

	ret = predictEP_adf(objectives_dict, epSolution, pareto_set, spacing)
        plt.figure()
        plt.plot(inputs, vals1, color='r', marker='o', markersize=10, linestyle='None')
        plt.plot(inputs, vals2, color='g', marker='x', markersize=10, linestyle='None')
	plt.plot(spacing, ret['mf']['f1'], 'r.')
	plt.plot(spacing, ret['mf']['f1'] + np.sqrt(ret['vf']['f1']), color = 'r', marker = '.', markersize = 1)
	plt.plot(spacing, ret['mf']['f1'] - np.sqrt(ret['vf']['f1']), color = 'r', marker = '.', markersize = 1)
	plt.plot(spacing, ret['mf']['f2'], 'g.')
	plt.plot(spacing, ret['mf']['f2'] + np.sqrt(ret['vf']['f2']), color = 'g', marker = '.', markersize = 1)
	plt.plot(spacing, ret['mf']['f2'] - np.sqrt(ret['vf']['f2']), color = 'g', marker = '.', markersize = 1)
        plt.plot(pareto_set[:,0], np.ones(len(pareto_set[:,0])) * 0, 'g.')
        plt.show()

	ret = predictEP_unconditioned(objectives_dict, epSolution, pareto_set, spacing)
        plt.figure()
        plt.plot(inputs, vals1, color='r', marker='o', markersize=10, linestyle='None')
        plt.plot(inputs, vals2, color='g', marker='x', markersize=10, linestyle='None')
	plt.plot(spacing, ret['mf']['f1'], 'r.')
	plt.plot(spacing, ret['mf']['f1'] + np.sqrt(ret['vf']['f1']), color = 'r', marker = '.', markersize = 1)
	plt.plot(spacing, ret['mf']['f1'] - np.sqrt(ret['vf']['f1']), color = 'r', marker = '.', markersize = 1)
	plt.plot(spacing, ret['mf']['f2'], 'g.')
	plt.plot(spacing, ret['mf']['f2'] + np.sqrt(ret['vf']['f2']), color = 'g', marker = '.', markersize = 1)
	plt.plot(spacing, ret['mf']['f2'] - np.sqrt(ret['vf']['f2']), color = 'g', marker = '.', markersize = 1)
        plt.plot(pareto_set[:,0], np.ones(len(pareto_set[:,0])) * 0, 'b.')
        plt.show()

	# We generate samples from the posterior that are compatible with the pareto points observed

	grid = np.linspace(0,1,20)[:,None]

	pareto_set_locations = np.zeros((0, 1))

	for i in range(pareto_set.shape[ 0 ]):
		to_include = grid[np.where(grid < pareto_set[ i, : ])[0]][-1]
		if to_include not in pareto_set_locations:
			pareto_set_locations = np.vstack((pareto_set_locations, to_include))

	n_total = 0

	samples_f1 = np.array([])
	samples_f2 = np.array([])

	for i in range(10000):
	
		# We sampel a GP from the posterior	
		
		sample = [ sample_gp_with_random_features(objective_gp, NUM_RANDOM_FEATURES) for objective_gp in objectives_dict.values() ]

		# We evaluate the GPs on the grid

		funs = sample

		val_f1 = funs[ 0 ](grid, False)
		val_f2 = funs[ 1 ](grid, False)

		values = np.vstack((val_f1, val_f2)).T

		selection = _cull_algorithm(values)
		optimal_locations = grid[ selection, : ]
		optimal_values = values[ selection, : ]

		all_included = True
		n_included = 0
		for j in range(pareto_set_locations.shape[ 0 ]):
			if not pareto_set_locations[ j, : ] in optimal_locations:
				all_included = False
			else:
				n_included += 1

		print(n_included)

		if all_included:

			print 'Included\n'

			if n_total == 0:
				samples_f1 = funs[ 0 ](spacing, False)
				samples_f2 = funs[ 1 ](spacing, False)
			else:
				samples_f1 = np.vstack((samples_f1, funs[ 0 ](spacing, False)))
				samples_f2 = np.vstack((samples_f2, funs[ 1 ](spacing, False)))

			n_total += 1

	pos2 = np.where(spacing > 0.84)[ 0 ][ 0 ]
	pos1 = np.where(spacing > 0.63)[ 0 ][ 0 ]
	sel = np.where(np.logical_and(samples_f1[ :, pos1 ] < samples_f2[ :, pos1 ], samples_f1[ :, pos2 ] < samples_f2[ :, pos2 ]))[ 0 ]

	plt.figure()
	plt.plot(inputs, vals1, color='r', marker='o', markersize=10, linestyle='None')
	plt.plot(inputs, vals2, color='g', marker='x', markersize=10, linestyle='None')
	plt.plot(spacing, np.mean(samples_f1[ sel, : ], axis = 0), 'r.')
	plt.plot(spacing, np.mean(samples_f1[ sel, : ], axis = 0) + np.std(samples_f1[ sel, : ], axis = 0), color = 'r', marker = '.', markersize = 1)
	plt.plot(spacing, np.mean(samples_f1[ sel, : ], axis = 0) - np.std(samples_f1[ sel, : ], axis = 0), color = 'r', marker = '.', markersize = 1)
	plt.plot(spacing, np.mean(samples_f2[ sel, : ], axis = 0), 'g.')
	plt.plot(spacing, np.mean(samples_f2[ sel, : ], axis = 0) + np.std(samples_f2[ sel, : ], axis = 0), color = 'g', marker = '.', markersize = 1)
	plt.plot(spacing, np.mean(samples_f2[ sel, : ], axis = 0) - np.std(samples_f2[ sel, : ], axis = 0), color = 'g', marker = '.', markersize = 1)
        plt.plot(pareto_set_locations, pareto_set_locations * 0, 'b.')
        plt.show()

	print(n_total)

	# We plot the approx acquisition function and the exact (over a single sample of the pareto set)

	ret = predictEP_multiple_iter_optim(objectives_dict, epSolution, pareto_set, spacing, n_iters = 10, damping = .5, no_negatives = True)
	var1_post_ap = ret['vf']['f1']
	var2_post_ap = ret['vf']['f2']
	initial_entropy = 0.5 * np.log(2 * 3.1415926 * var1 * np.exp(1)) + 0.5 * np.log(2 * 3.1415926 * var2 * np.exp(1))
	posterior_entropy_ap = 0.5 * np.log(2 * 3.1415926 * var1_post_ap * np.exp(1)) + 0.5 * np.log(2 * 3.1415926 * var2_post_ap * np.exp(1))

	posterior_entropy_ext = np.zeros(spacing.shape[ 0 ])
			
	for u in range(spacing.shape[ 0 ]):
		obs = np.vstack((samples_f1[ :, u ], samples_f2[ :, u ])).T
		posterior_entropy_ext[ u ] = entropy(obs.tolist(), k = 5, base = np.exp(1))

	plt.figure()
	plt.plot(inputs, vals2 * 0, color='black', marker='x', markersize=10, linestyle='None')
	plt.plot(spacing, initial_entropy - posterior_entropy_ext, color='red', marker='.', markersize=1)
	plt.plot(spacing, initial_entropy - posterior_entropy_ap, color='blue', marker='.', markersize=1)
        plt.show()

	import pdb; pdb.set_trace()

# TODO

# Test the predictive distribution given a pareto set

def test_acquisition_function(iteration = 0):

	np.random.seed(2)

	D = 1
	N = 7
    
	inputs  = npr.rand(N,D)
	vals1 = np.sin(np.sum(inputs,axis=1)*7.0)*np.sum(inputs,axis=1)
	vals2 = np.cos(np.sum(inputs,axis=1)*7.0)*np.sum(inputs,axis=1)
	vals3 = np.tan(np.sum(inputs,axis=1)*7.0)*np.sum(inputs,axis=1)
	objective1 = GP(D)#, kernel='SquaredExp')#, likelihood='noiseless')
	objective2 = GP(D)#, kernel='SquaredExp')#, likelihood='noiseless')
	constraint1 = GP(D)#, kernel='SquaredExp')#, likelihood='noiseless')
	objective1.fit(inputs, vals1, fit_hypers = False)
	objective2.fit(inputs, vals2, fit_hypers = False)
	constraint1.fit(inputs, vals3, fit_hypers = False)

	print 'ls=%s' % str(objective1.params['ls'].value)
	print 'noise=%f' % float(objective1.params['noise'].value)
	print 'amp2=%f' % float(objective1.params['amp2'].value)

	print '\n'

	print 'ls=%s' % str(objective2.params['ls'].value)
	print 'noise=%f' % float(objective2.params['noise'].value)
	print 'amp2=%f' % float(objective2.params['amp2'].value)

	print '\n'
	
	print 'ls=%s' % str(constraint1.params['ls'].value)
        print 'noise=%f' % float(constraint1.params['noise'].value)
        print 'amp2=%f' % float(constraint1.params['amp2'].value)

	objectives_dict = dict()

	objectives_dict['f1'] = objective1
	objectives_dict['f2'] = objective2

	constraints_dict = dict()

	constraints_dict['c1'] = constraint1

        spacing = np.linspace(0,1,1000)[:,None]

	mean1 = objective1.predict(spacing)[0]
	mean2 = objective2.predict(spacing)[0]
	var1 = objective1.predict(spacing)[1]
	var2 = objective2.predict(spacing)[1]

	mean_c1 = constraint1.predict(spacing)[0]
	var_c1 = constraint1.predict(spacing)[1]

	total_samples = 0
	k = 0

	np.random.seed(int(iteration))
	
	while total_samples < 10:

		print 'Total Samples:%d Sample:%d' % (total_samples, k)

		gp_samples = dict()
		gp_samples['objectives'] = [ sample_gp_with_random_features(objective_gp, NUM_RANDOM_FEATURES) \
			for objective_gp in objectives_dict.values() ]
		gp_samples['constraints'] = [ sample_gp_with_random_features(constraint1, NUM_RANDOM_FEATURES) ]
		funs = gp_samples['objectives']

		grid = np.linspace(0,1,50)[:,None] #Using 50 points is an approximation to 20 points without a constraint.

		val_c1 = gp_samples['constraints'][0](grid, False)
                grid = grid[np.where(val_c1>=0)] #Grid will only contain feasible points.

		val_f1 = funs[ 0 ](grid, False)
		val_f2 = funs[ 1 ](grid, False)
		
		values = np.vstack((val_f1, val_f2)).T
		selection = _cull_algorithm(values)
		pareto_set_locations = grid[ selection, : ]
		front = values[ selection, : ]

		print '\tPareto Set size Before Summary:%f' % (float(pareto_set_locations.shape[ 0 ]))

		result = _compute_pareto_front_and_set_summary_x_space(front, pareto_set_locations, 3)

		pareto_set_locations = result['pareto_set']
		front = result['frontier']

#		moo = MOOP_basis_functions(funs, 1)
#		moo.evolve(100, 100)
#		result = moo.compute_pareto_front_and_set_summary(3)
#		pareto_set = result['pareto_set']
#		front = result['frontier']

		# We generate samples from the posterior that are compatible with the pareto points observed

#		pareto_set_locations = np.zeros((0, 1))

#		for i in range(pareto_set.shape[ 0 ]):
#			to_include = grid[np.where(grid < pareto_set[ i, : ])[0]][-1]
#			if to_include not in pareto_set_locations:
#				pareto_set_locations = np.vstack((pareto_set_locations, to_include))

		print '\tPareto Set size:%f' % (float(pareto_set_locations.shape[ 0 ]))

		n_total = 0
		
		samples_f1 = np.array([])
		samples_f2 = np.array([])
		samples_c1 = np.array([])
			
		for i in range(100): # ECGM: I have got a 32% of valid points.... so, for speed's sake, i tuned this parameter to 100. ( 32 expected points, good enough for the 10 test ).
	
			# We sample a GP from the posterior
			grid = np.linspace(0,1,50)[:,None] #Using 50 points is an approximation to 20 points without a constraint.
			sample = [ sample_gp_with_random_features(objective_gp, NUM_RANDOM_FEATURES) for objective_gp in objectives_dict.values() ]
			sample_constraints = [ sample_gp_with_random_features( constraint1, NUM_RANDOM_FEATURES ) ]	
			
			# We evaluate the GPs on the grid
			funs = sample
	
			val_c1 = sample_constraints[0](grid, False)
			grid = grid[np.where(val_c1 >= 0)]
			val_f1 = funs[ 0 ](grid, False)
			val_f2 = funs[ 1 ](grid, False)
	
			values = np.vstack((val_f1, val_f2)).T
	
			selection = _cull_algorithm(values)
			optimal_locations = grid[ selection, : ]
			optimal_values = values[ selection, : ]
	
			all_included = True
			for j in range(pareto_set_locations.shape[ 0 ]):
				if not pareto_set_locations[ j, : ] in optimal_locations:
					all_included = False

			if all_included:
				if n_total == 0:
					samples_f1 = funs[ 0 ](spacing, False)
					samples_f2 = funs[ 1 ](spacing, False)
					samples_c1 = sample_constraints[0](spacing, False)
				else:
					samples_f1 = np.vstack((samples_f1, funs[ 0 ](spacing, False)))
					samples_f2 = np.vstack((samples_f2, funs[ 1 ](spacing, False)))
					samples_c1 = np.vstack((samples_c1, sample_constraints[0](spacing, False)))
	
				n_total += 1

		print(n_total)
		
		if n_total > 10:

			epSolution = ep(objectives_dict, pareto_set_locations, con_models = constraints_dict, minimize=True)

			# We plot the approx acquisition function and the exact (over a single sample of the pareto set)
	
			ret = predictEP_multiple_iter_optim_robust(objectives_dict, constraints_dict, epSolution, pareto_set_locations, spacing, n_iters = 1, 
				damping = .1, no_negatives = True)
			var1_post_ext = np.var(samples_f1, axis = 0)
			var2_post_ext = np.var(samples_f2, axis = 0)
			cons1_post_ext = np.var(samples_c1, axis = 0)
			var1_post_ap = ret['vf']['f1']
			var2_post_ap = ret['vf']['f2']
			cons1_post_ap = ret['vc']['c1']
			K = len(ret['vf'])
			C = len(ret['vc'])
			
			initial_entropy = ((K+C)/2.0) * np.log(2.0 * np.pi * np.exp(1)) + 0.5 * np.log(var1) + 0.5 * np.log(var2) + 0.5 * np.log(var_c1)
			posterior_entropy_ext = np.zeros(spacing.shape[ 0 ])
			
			for u in range(spacing.shape[ 0 ]):
				obs = np.vstack((samples_f1[ :, u ], samples_f2[ :, u ], samples_c1[ :, u ])).T
				posterior_entropy_ext[ u ] = entropy(obs.tolist(), k = 1, base = np.exp(1))
			
			#ECGM: I just use here the expression 5 of "Predictive Entropy Search for Multi-objective Optimization with constraints" conference paper.
			entropy_ap = np.log(var_c1) + np.log(var1) + np.log(var2) - (np.log(cons1_post_ap) + np.log(var1_post_ap) + np.log(var2_post_ap))
	
			if total_samples == 0:
				acq_ext = np.array(initial_entropy - posterior_entropy_ext).reshape((1, 1000))
				acq_ap = np.array(entropy_ap).reshape((1, 1000))
			else:
				acq_ext = np.vstack((acq_ext, np.array(initial_entropy - posterior_entropy_ext).reshape((1, 1000))))
				acq_ap = np.vstack((acq_ap, np.array(entropy_ap).reshape((1, 1000))))
			
			total_samples += 1

		k += 1

	# We save the results

	name_exact = './results/coupled/exact_%s' % (iteration)
	name_ap = './results/coupled/ap_%s' % (iteration)
	
	import pdb; pdb.set_trace();
	np.save(name_exact, acq_ext)
	np.save(name_ap, acq_ap)

	#import matplotlib.pyplot as plt

#	plt.figure()
#	plt.plot(inputs, vals2 * 0, color='black', marker='x', markersize=10, linestyle='None')
#	plt.plot(spacing, np.mean(acq_ext, axis = 0), color='red', marker='.', markersize=1)
#	plt.plot(spacing, np.mean(acq_ap, axis = 0),  color='blue', marker='.', markersize=1)
#       plt.show()

#	plt.figure()
#	plt.plot(inputs, vals1, color='r', marker='o', markersize=10, linestyle='Nonie')
#	plt.plot(inputs, vals2, color='g', marker='x', markersize=10, linestyle='None')
#	plt.plot(spacing, np.mean(samples_f1, axis = 0), 'r.')
#	plt.plot(spacing, np.mean(samples_f1, axis = 0) + np.std(samples_f1, axis = 0), color = 'r', marker = '.', markersize = 1)
#	plt.plot(spacing, np.mean(samples_f1, axis = 0) - np.std(samples_f1, axis = 0), color = 'r', marker = '.', markersize = 1)
#	plt.plot(spacing, np.mean(samples_f2, axis = 0), 'g.')
#	plt.plot(spacing, np.mean(samples_f2, axis = 0) + np.std(samples_f2, axis = 0), color = 'g', marker = '.', markersize = 1)
#	plt.plot(spacing, np.mean(samples_f2, axis = 0) - np.std(samples_f2, axis = 0), color = 'g', marker = '.', markersize = 1)
#	plt.plot(pareto_set_locations, pareto_set_locations * 0, 'b.')
#	plt.show()

#	plt.figure()
#	plt.plot(inputs, vals1, color='r', marker='o', markersize=10, linestyle='None')
#	plt.plot(inputs, vals2, color='g', marker='x', markersize=10, linestyle='None')
#	plt.plot(spacing, ret['mf']['f1'], 'r.')
#	plt.plot(spacing, ret['mf']['f1'] + np.sqrt(ret['vf']['f1']), color = 'r', marker = '.', markersize = 1)
#	plt.plot(spacing, ret['mf']['f1'] - np.sqrt(ret['vf']['f1']), color = 'r', marker = '.', markersize = 1)
#	plt.plot(spacing, ret['mf']['f2'], 'g.')
#	plt.plot(spacing, ret['mf']['f2'] + np.sqrt(ret['vf']['f2']), color = 'g', marker = '.', markersize = 1)
#	plt.plot(spacing, ret['mf']['f2'] - np.sqrt(ret['vf']['f2']), color = 'g', marker = '.', markersize = 1)
#       plt.plot(pareto_set_locations, pareto_set_locations * 0, 'b.')
#        plt.show()

#        plt.figure()
#        plt.plot(inputs, vals1, color='r', marker='o', markersize=10, linestyle='None')
#        plt.plot(inputs, vals2, color='g', marker='x', markersize=10, linestyle='None')
# 	 plt.plot(spacing, mean1, 'r.')
#	 plt.plot(spacing, mean1 + np.sqrt(var1), color = 'r', marker = '.', markersize = 1)
#	 plt.plot(spacing, mean1 - np.sqrt(var1), color = 'r', marker = '.', markersize = 1)
#        plt.plot(spacing, mean2, 'g.')
#	 plt.plot(spacing, mean2 + np.sqrt(var2), color = 'g', marker = '.', markersize = 1)
#	 plt.plot(spacing, mean2 - np.sqrt(var2), color = 'g', marker = '.', markersize = 1)
#        plt.show()

# Test the predictive distribution given a pareto set

def test_acquisition_function_decoupled(iteration = 0):

	np.random.seed(3)

	D = 1
	N = 7
    
	inputs1  = npr.rand(N,D)
	inputs2  = npr.rand(N,D)
	inputs3 = npr.rand(N,D)
	vals1 = np.sin(np.sum(inputs1,axis=1)*7.0)*np.sum(inputs1,axis=1)
	vals2 = np.cos(np.sum(inputs2,axis=1)*7.0)*np.sum(inputs2,axis=1)
	vals3 = np.tan(np.sum(inputs3,axis=1)*7.0)*np.sum(inputs3,axis=1)
	objective1 = GP(D)#, kernel='SquaredExp')#, likelihood='noiseless')
	objective2 = GP(D)#, kernel='SquaredExp')#, likelihood='noiseless')
	constraint1 = GP(D)#, kernel='SquaredExp')#, likelihood='noiseless')
	objective1.fit(inputs1, vals1, fit_hypers = False)
	objective2.fit(inputs2, vals2, fit_hypers = False)
	constraint1.fit(inputs3, vals3, fit_hypers = False)

	print 'ls=%s' % str(objective1.params['ls'].value)
	print 'noise=%f' % float(objective1.params['noise'].value)
	print 'amp2=%f' % float(objective1.params['amp2'].value)

	print '\n'

	print 'ls=%s' % str(objective2.params['ls'].value)
	print 'noise=%f' % float(objective2.params['noise'].value)
	print 'amp2=%f' % float(objective2.params['amp2'].value)

	print '\n'

        print 'ls=%s' % str(constraint1.params['ls'].value)
        print 'noise=%f' % float(constraint1.params['noise'].value)
        print 'amp2=%f' % float(constraint1.params['amp2'].value)

	objectives_dict = dict()

	objectives_dict['f1'] = objective1
	objectives_dict['f2'] = objective2

	constraints_dict = dict()
	constraints_dict['c1'] = constraint1

        spacing = np.linspace(0, 1, 1000)[:,None]

	mean1 = objective1.predict(spacing)[0]
	mean2 = objective2.predict(spacing)[0]
	var1 = objective1.predict(spacing)[1]
	var2 = objective2.predict(spacing)[1]

	mean_c1 = constraint1.predict(spacing)[0]
	var_c1 = constraint1.predict(spacing)[1]

	total_samples = 0
	k = 0

	np.random.seed(int(iteration))
	
	while total_samples < 10:

		print 'Total Samples:%d Sample:%d' % (total_samples, k)

		gp_samples = dict()
		gp_samples['objectives'] = [ sample_gp_with_random_features(objective_gp, NUM_RANDOM_FEATURES) \
			for objective_gp in objectives_dict.values() ]
		gp_samples['constraints'] = [ sample_gp_with_random_features(constraint1, NUM_RANDOM_FEATURES) ]
		funs = gp_samples['objectives']

		grid = np.linspace(0,1,50)[:,None]
	
		val_c1 = gp_samples['constraints'][0](grid, False)
                grid = grid[np.where(val_c1>=0)] #Grid will only contain feasible points.

		val_f1 = funs[ 0 ](grid, False)
		val_f2 = funs[ 1 ](grid, False)

		values = np.vstack((val_f1, val_f2)).T
		selection = _cull_algorithm(values)
		pareto_set_locations = grid[ selection, : ]
		front = values[ selection, : ]

		print '\tPareto Set size Before Summary:%f' % (float(pareto_set_locations.shape[ 0 ]))

		result = _compute_pareto_front_and_set_summary_x_space(front, pareto_set_locations, 3)

		pareto_set_locations = result['pareto_set']
		front = result['frontier']

#		moo = MOOP_basis_functions(funs, 1)
#		moo.evolve(100, 100)
#		result = moo.compute_pareto_front_and_set_summary(3)
#		pareto_set = result['pareto_set']
#		front = result['frontier']

		# We generate samples from the posterior that are compatible with the pareto points observed

#		pareto_set_locations = np.zeros((0, 1))

#		for i in range(pareto_set.shape[ 0 ]):
#			to_include = grid[np.where(grid < pareto_set[ i, : ])[0]][-1]
#			if to_include not in pareto_set_locations:
#				pareto_set_locations = np.vstack((pareto_set_locations, to_include))

		print '\tPareto Set size:%f' % (float(pareto_set_locations.shape[ 0 ]))

		n_total = 0

		samples_f1 = np.array([])
		samples_f2 = np.array([])
		samples_c1 = np.array([])

		for i in range(100):
	
			# We sample a GP from the posterior
			grid = np.linspace(0,1,50)[:,None]		
			sample = [ sample_gp_with_random_features(objective_gp, NUM_RANDOM_FEATURES) for objective_gp in objectives_dict.values() ]
			sample_constraints = [ sample_gp_with_random_features( constraint1, NUM_RANDOM_FEATURES ) ]

			# We evaluate the GPs on the grid
	
			funs = sample
	
			val_c1 = sample_constraints[0](grid, False)
                        grid = grid[np.where(val_c1 >= 0)]
			val_f1 = funs[ 0 ](grid, False)
			val_f2 = funs[ 1 ](grid, False)
	
			values = np.vstack((val_f1, val_f2)).T
	
			selection = _cull_algorithm(values)
			optimal_locations = grid[ selection, : ]
			optimal_values = values[ selection, : ]
	
			all_included = True
			for j in range(pareto_set_locations.shape[ 0 ]):
				if not pareto_set_locations[ j, : ] in optimal_locations:
					all_included = False

			if all_included:

				if n_total == 0:
					samples_f1 = funs[ 0 ](spacing, False)
					samples_f2 = funs[ 1 ](spacing, False)
					samples_c1 = sample_constraints[0](spacing, False)
				else:
					samples_f1 = np.vstack((samples_f1, funs[ 0 ](spacing, False)))
					samples_f2 = np.vstack((samples_f2, funs[ 1 ](spacing, False)))
					samples_c1 = np.vstack((samples_c1, sample_constraints[0](spacing, False)))
	
				n_total += 1

		print(n_total)

		if n_total > 10:

			epSolution = ep(objectives_dict, pareto_set_locations, con_models = constraints_dict, minimize=True)

			# We plot the approx acquisition function and the exact (over a single sample of the pareto set)
	
			ret = predictEP_multiple_iter_optim_robust(objectives_dict, constraints_dict, epSolution, pareto_set_locations, spacing, n_iters = 1, 
				damping = .1, no_negatives = True)
			var1_post_ext = np.var(samples_f1, axis = 0)
			var2_post_ext = np.var(samples_f2, axis = 0)
			cons1_post_ext = np.var(samples_c1, axis = 0)
			var1_post_ap = ret['vf']['f1']
			var2_post_ap = ret['vf']['f2']
			cons1_post_ap = ret['vc']['c1']
			initial_entropy_1 = 0.5 * np.log(2.0 * np.pi * var1 * np.exp(1))
			initial_entropy_2 = 0.5 * np.log(2.0 * np.pi * var2 * np.exp(1))
			initial_entropy_3 = 0.5 * np.log(2.0 * np.pi * var_c1 * np.exp(1))
			posterior_entropy_ext_1 = np.zeros(spacing.shape[ 0 ])
			posterior_entropy_ext_2 = np.zeros(spacing.shape[ 0 ])
			posterior_entropy_ext_3 = np.zeros(spacing.shape[ 0 ])
			
			for u in range(spacing.shape[ 0 ]):
				s_f1 = samples_f1[ :, u ].reshape((samples_f1.shape[ 0 ], 1)).tolist()
				s_f2 = samples_f2[ :, u ].reshape((samples_f2.shape[ 0 ], 1)).tolist()
				s_c1 = samples_c1[ :, u ].reshape((samples_c1.shape[ 0 ], 1)).tolist()
				posterior_entropy_ext_1[ u ] = entropy(s_f1, k = 1, base = np.exp(1))
				posterior_entropy_ext_2[ u ] = entropy(s_f2, k = 1, base = np.exp(1))
				posterior_entropy_ext_3[ u ] = entropy(s_c1, k = 1, base = np.exp(1))

			posterior_entropy_ap_1 = 0.5 * np.log(2.0 * np.pi * var1_post_ap * np.exp(1)) 
			posterior_entropy_ap_2 = 0.5 * np.log(2.0 * np.pi * var2_post_ap * np.exp(1))
			posterior_entropy_ap_3 = 0.5 * np.log(2.0 * np.pi * cons1_post_ap * np.exp(1))
	
			if total_samples == 0:
				acq_ext_1 = np.array(initial_entropy_1 - posterior_entropy_ext_1).reshape((1, 1000))
				acq_ext_2 = np.array(initial_entropy_2 - posterior_entropy_ext_2).reshape((1, 1000))
				acq_ext_3 = np.array(initial_entropy_3 - posterior_entropy_ext_3).reshape((1, 1000))
				acq_ap_1 = np.array(initial_entropy_1 - posterior_entropy_ap_1).reshape((1, 1000))
				acq_ap_2 = np.array(initial_entropy_2 - posterior_entropy_ap_2).reshape((1, 1000))
				acq_ap_3 = np.array(initial_entropy_3 - posterior_entropy_ap_3).reshape((1, 1000))
			else:
				acq_ext_1 = np.vstack((acq_ext_1, np.array(initial_entropy_1 - posterior_entropy_ext_1).reshape((1, 1000))))
				acq_ext_2 = np.vstack((acq_ext_2, np.array(initial_entropy_2 - posterior_entropy_ext_2).reshape((1, 1000))))
				acq_ext_3 = np.vstack((acq_ext_3, np.array(initial_entropy_3 - posterior_entropy_ext_3).reshape((1, 1000))))
				acq_ap_1 = np.vstack((acq_ap_1, np.array(initial_entropy_1 - posterior_entropy_ap_1).reshape((1, 1000))))
				acq_ap_2 = np.vstack((acq_ap_2, np.array(initial_entropy_2 - posterior_entropy_ap_2).reshape((1, 1000))))
				acq_ap_3 = np.vstack((acq_ap_3, np.array(initial_entropy_3 - posterior_entropy_ap_3).reshape((1, 1000))))
			
			total_samples += 1

		k += 1

	# We save the results

	name_exact = './results/decoupled/exact_%s' % (iteration)
	name_ap = './results/decoupled/ap_%s' % (iteration)

	np.save(name_exact + '_1', acq_ext_1)
	np.save(name_exact + '_2', acq_ext_2)
	np.save(name_exact + '_3', acq_ext_3)
	np.save(name_ap + '_1', acq_ap_1)
	np.save(name_ap + '_2', acq_ap_2)
	np.save(name_ap + '_3', acq_ap_3)

	import matplotlib.pyplot as plt

#	plt.figure()
#	plt.plot(inputs, vals2 * 0, color='black', marker='x', markersize=10, linestyle='None')
#	plt.plot(spacing, np.mean(acq_ext, axis = 0), color='red', marker='.', markersize=1)
#	plt.plot(spacing, np.mean(acq_ap, axis = 0),  color='blue', marker='.', markersize=1)
#       plt.show()

#	plt.figure()
#	plt.plot(inputs, vals1, color='r', marker='o', markersize=10, linestyle='None')
#	plt.plot(inputs, vals2, color='g', marker='x', markersize=10, linestyle='None')
#	plt.plot(spacing, np.mean(samples_f1, axis = 0), 'r.')
#	plt.plot(spacing, np.mean(samples_f1, axis = 0) + np.std(samples_f1, axis = 0), color = 'r', marker = '.', markersize = 1)
#	plt.plot(spacing, np.mean(samples_f1, axis = 0) - np.std(samples_f1, axis = 0), color = 'r', marker = '.', markersize = 1)
#	plt.plot(spacing, np.mean(samples_f2, axis = 0), 'g.')
#	plt.plot(spacing, np.mean(samples_f2, axis = 0) + np.std(samples_f2, axis = 0), color = 'g', marker = '.', markersize = 1)
#	plt.plot(spacing, np.mean(samples_f2, axis = 0) - np.std(samples_f2, axis = 0), color = 'g', marker = '.', markersize = 1)
#	plt.plot(pareto_set_locations, pareto_set_locations * 0, 'b.')
#	plt.show()

#	plt.figure()
#	plt.plot(inputs, vals1, color='r', marker='o', markersize=10, linestyle='None')
#	plt.plot(inputs, vals2, color='g', marker='x', markersize=10, linestyle='None')
#	plt.plot(spacing, ret['mf']['f1'], 'r.')
#	plt.plot(spacing, ret['mf']['f1'] + np.sqrt(ret['vf']['f1']), color = 'r', marker = '.', markersize = 1)
#	plt.plot(spacing, ret['mf']['f1'] - np.sqrt(ret['vf']['f1']), color = 'r', marker = '.', markersize = 1)
#	plt.plot(spacing, ret['mf']['f2'], 'g.')
#	plt.plot(spacing, ret['mf']['f2'] + np.sqrt(ret['vf']['f2']), color = 'g', marker = '.', markersize = 1)
#	plt.plot(spacing, ret['mf']['f2'] - np.sqrt(ret['vf']['f2']), color = 'g', marker = '.', markersize = 1)
#       plt.plot(pareto_set_locations, pareto_set_locations * 0, 'b.')
#        plt.show()

#        plt.figure()
#        plt.plot(inputs, vals1, color='r', marker='o', markersize=10, linestyle='None')
#        plt.plot(inputs, vals2, color='g', marker='x', markersize=10, linestyle='None')
# 	 plt.plot(spacing, mean1, 'r.')
#	 plt.plot(spacing, mean1 + np.sqrt(var1), color = 'r', marker = '.', markersize = 1)
#	 plt.plot(spacing, mean1 - np.sqrt(var1), color = 'r', marker = '.', markersize = 1)
#        plt.plot(spacing, mean2, 'g.')
#	 plt.plot(spacing, mean2 + np.sqrt(var2), color = 'g', marker = '.', markersize = 1)
#	 plt.plot(spacing, mean2 - np.sqrt(var2), color = 'g', marker = '.', markersize = 1)
#        plt.show()



def test_plot_results_decoupled(num_results):

	np.random.seed(3)

	D = 1
	N = 7
    
	inputs1  = npr.rand(N,D)
	np.savetxt("R/plot_acq/decoupled/txts/inputs1.txt", inputs1, delimiter = "\n")
	inputs2  = npr.rand(N,D)
	np.savetxt("R/plot_acq/decoupled/txts/inputs2.txt", inputs2, delimiter = "\n")
	inputs3  = npr.rand(N,D)
	np.savetxt("R/plot_acq/decoupled/txts/inputs3.txt", inputs3, delimiter = "\n")

	vals1 = np.sin(np.sum(inputs1,axis=1)*7.0)*np.sum(inputs1,axis=1)
	np.savetxt("R/plot_acq/decoupled/txts/vals1.txt", vals1, delimiter = "\n")
	vals2 = np.cos(np.sum(inputs2,axis=1)*7.0)*np.sum(inputs2,axis=1)
	np.savetxt("R/plot_acq/decoupled/txts/vals2.txt", vals2, delimiter = "\n")
	vals3 = np.tan(np.sum(inputs3,axis=1)*7.0)*np.sum(inputs3,axis=1)
	np.savetxt("R/plot_acq/decoupled/txts/vals3.txt", vals3, delimiter = "\n")

	objective1 = GP(D)#, kernel='SquaredExp')#, likelihood='noiseless')
	objective2 = GP(D)#, kernel='SquaredExp')#, likelihood='noiseless')
	constraint1 = GP(D)#, kernel='SquaredExp')#, likelihood='noiseless')
	objective1.fit(inputs1, vals1, fit_hypers = False)
	objective2.fit(inputs2, vals2, fit_hypers = False)
	constraint1.fit(inputs3, vals3, fit_hypers = False)

	print 'ls=%s' % str(objective1.params['ls'].value)
	print 'noise=%f' % float(objective1.params['noise'].value)
	print 'amp2=%f' % float(objective1.params['amp2'].value)

	print '\n'

	print 'ls=%s' % str(objective2.params['ls'].value)
	print 'noise=%f' % float(objective2.params['noise'].value)
	print 'amp2=%f' % float(objective2.params['amp2'].value)

	print '\n'

        print 'ls=%s' % str(constraint1.params['ls'].value)
        print 'noise=%f' % float(constraint1.params['noise'].value)
        print 'amp2=%f' % float(constraint1.params['amp2'].value)
	
	objectives_dict = dict()

	objectives_dict['f1'] = objective1
	objectives_dict['f2'] = objective2

	constraints_dict = dict()

	constraints_dict['c1'] = constraint1

        spacing = np.linspace(0, 1, 1000)[:,None]
	np.savetxt("R/plot_acq/decoupled/txts/spacing.txt", spacing, delimiter = "\n")

	mean1 = objective1.predict(spacing)[0]
	np.savetxt("R/plot_acq/decoupled/txts/mean1.txt", mean1, delimiter = "\n")
	mean2 = objective2.predict(spacing)[0]
	np.savetxt("R/plot_acq/decoupled/txts/mean2.txt", mean2, delimiter = "\n")
	var1 = objective1.predict(spacing)[1]
	np.savetxt("R/plot_acq/decoupled/txts/vars1.txt", var1, delimiter = "\n")
	var2 = objective2.predict(spacing)[1]
	np.savetxt("R/plot_acq/decoupled/txts/vars2.txt", var2, delimiter = "\n")
	mean_c1 = constraint1.predict(spacing)[0]
	np.savetxt("R/plot_acq/decoupled/txts/mean_c1.txt", mean_c1, delimiter = "\n")
	var_c1 = constraint1.predict(spacing)[1]
	np.savetxt("R/plot_acq/decoupled/txts/vars_c1.txt", var_c1, delimiter = "\n")

	total_samples = 0
	k = 0

	import matplotlib.pyplot as plt

	for i in range(num_results):

		name_exact_1 = 'results/decoupled/exact_%d_1.npy' % (i + 1)
		name_exact_2 = 'results/decoupled/exact_%d_2.npy' % (i + 1)
		name_exact_3 = 'results/decoupled/exact_%d_3.npy' % (i + 1)
		name_ap_1 = 'results/decoupled/ap_%d_1.npy' % (i + 1)
		name_ap_2 = 'results/decoupled/ap_%d_2.npy' % (i + 1)
		name_ap_3 = 'results/decoupled/ap_%d_3.npy' % (i + 1)

		if i == 0:
			acq_ext_1 = np.load(name_exact_1)
			acq_ext_2 = np.load(name_exact_2)
			acq_ext_3 = np.load(name_exact_3)
			acq_ap_1 = np.load(name_ap_1)
			acq_ap_2 = np.load(name_ap_2)
			acq_ap_3 = np.load(name_ap_3)
		else:
			acq_ext_1 = np.vstack((acq_ext_1, np.load(name_exact_1)))
			acq_ext_2 = np.vstack((acq_ext_2, np.load(name_exact_2)))
			acq_ext_3 = np.vstack((acq_ext_3, np.load(name_exact_3)))
			acq_ap_1 = np.vstack((acq_ap_1, np.load(name_ap_1)))
			acq_ap_2 = np.vstack((acq_ap_2, np.load(name_ap_2)))
			acq_ap_3 = np.vstack((acq_ap_3, np.load(name_ap_3)))

	np.savetxt("R/plot_acq/decoupled/txts/acq_ext_1.txt", acq_ext_1, delimiter = "\n")
	np.savetxt("R/plot_acq/decoupled/txts/acq_ext_2.txt", acq_ext_2, delimiter = "\n")
	np.savetxt("R/plot_acq/decoupled/txts/acq_ext_3.txt", acq_ext_3, delimiter = "\n")
	np.savetxt("R/plot_acq/decoupled/txts/acq_ap_1.txt", acq_ap_1, delimiter = "\n")
	np.savetxt("R/plot_acq/decoupled/txts/acq_ap_2.txt", acq_ap_2, delimiter = "\n")
	np.savetxt("R/plot_acq/decoupled/txts/acq_ap_3.txt", acq_ap_3, delimiter = "\n")

	import pdb; pdb.set_trace();

	plt.figure()
	plt.plot(inputs1, vals1 * 0, color='black', marker='x', markersize=10, linestyle='None')
	plt.plot(spacing, np.mean(acq_ext_1, axis = 0), color='red', marker='.', markersize=1)
	plt.plot(spacing, np.mean(acq_ap_1, axis = 0),  color='blue', marker='.', markersize=1)
        plt.show()

	plt.figure()
	plt.plot(inputs2, vals2 * 0, color='black', marker='x', markersize=10, linestyle='None')
	plt.plot(spacing, np.mean(acq_ext_2, axis = 0), color='red', marker='.', markersize=1)
	plt.plot(spacing, np.mean(acq_ap_2, axis = 0),  color='blue', marker='.', markersize=1)
        plt.show()

	plt.figure()
        plt.plot(inputs3, vals3 * 0, color='black', marker='x', markersize=10, linestyle='None')
        plt.plot(spacing, np.mean(acq_ext_3, axis = 0), color='red', marker='.', markersize=1)
        plt.plot(spacing, np.mean(acq_ap_3, axis = 0),  color='blue', marker='.', markersize=1)
        plt.show()


	plt.figure()
	plt.plot(inputs1, vals1, color='b', marker='o', markersize=10, linestyle='None')
	plt.plot(inputs2, vals2, color='g', marker='s', markersize=10, linestyle='None')
	plt.plot(inputs3, vals3, color='r', marker='x', markersize=10, linestyle='None')
	plt.plot(spacing, mean1, 'b.')
	plt.plot(spacing, mean1 + np.sqrt(var1), color = 'b', marker = '.', markersize = 1)
	plt.plot(spacing, mean1 - np.sqrt(var1), color = 'b', marker = '.', markersize = 1)
	plt.plot(spacing, mean2, 'g.')
	plt.plot(spacing, mean2 + np.sqrt(var2), color = 'g', marker = '.', markersize = 1)
	plt.plot(spacing, mean2 - np.sqrt(var2), color = 'g', marker = '.', markersize = 1)
	plt.plot(spacing, mean_c1, 'r.')
        plt.plot(spacing, mean_c1 + np.sqrt(var_c1), color = 'r', marker = '.', markersize = 1)
        plt.plot(spacing, mean_c1 - np.sqrt(var_c1), color = 'r', marker = '.', markersize = 1)
	plt.show()


def test_plot_results(num_results):

	np.random.seed(2)

	D = 1
	N = 7
    
	inputs  = npr.rand(N,D)	
	np.savetxt("R/plot_acq/coupled/txts/inputs.txt", inputs, delimiter = "\n")
	vals1_no_noise = np.sin(np.sum(inputs,axis=1)*7.0)*np.sum(inputs,axis=1)
	vals1 = vals1_no_noise +npr.randn(N)*0.1
	np.savetxt("R/plot_acq/coupled/txts/vals1.txt", vals1, delimiter = "\n")
	vals2_no_noise = np.cos(np.sum(inputs,axis=1)*7.0)*np.sum(inputs,axis=1)
	vals2 = vals2_no_noise +npr.randn(N)*0.1
	np.savetxt("R/plot_acq/coupled/txts/vals2.txt", vals2, delimiter = "\n")
	vals3_no_noise = np.tan(np.sum(inputs,axis=1)*7.0)*np.sum(inputs,axis=1)
	vals3 = vals3_no_noise + npr.randn(N)*0.1
	np.savetxt("R/plot_acq/coupled/txts/vals3.txt", vals3, delimiter = "\n")
	objective1 = GP(D)#, kernel='SquaredExp')#, likelihood='noiseless')
	objective2 = GP(D)#, kernel='SquaredExp')#, likelihood='noiseless')
	constraint1 = GP(D)#, kernel='SquaredExp')#, likelihood='noiseless')
	objective1.fit(inputs, vals1, fit_hypers = False)
	objective2.fit(inputs, vals2, fit_hypers = False)
	constraint1.fit(inputs, vals3, fit_hypers = False)

	print 'ls=%s' % str(objective1.params['ls'].value)
	print 'noise=%f' % float(objective1.params['noise'].value)
	print 'amp2=%f' % float(objective1.params['amp2'].value)

	print '\n'

	print 'ls=%s' % str(objective2.params['ls'].value)
	print 'noise=%f' % float(objective2.params['noise'].value)
	print 'amp2=%f' % float(objective2.params['amp2'].value)

	print '\n'

        print 'ls=%s' % str(constraint1.params['ls'].value)
        print 'noise=%f' % float(constraint1.params['noise'].value)
        print 'amp2=%f' % float(constraint1.params['amp2'].value)
	
	objectives_dict = dict()

	objectives_dict['f1'] = objective1
	objectives_dict['f2'] = objective2

	constraints_dict = dict()

	constraints_dict['c1'] = constraint1

        spacing = np.linspace(0,1,1000)[:,None]
	np.savetxt("R/plot_acq/coupled/txts/spacing.txt", spacing, delimiter = "\n")

	mean1 = objective1.predict(spacing)[0]
	np.savetxt("R/plot_acq/coupled/txts/mean1.txt", mean1, delimiter = "\n")
	mean2 = objective2.predict(spacing)[0]
	np.savetxt("R/plot_acq/coupled/txts/mean2.txt", mean2, delimiter = "\n")
	var1 = objective1.predict(spacing)[1]
	np.savetxt("R/plot_acq/coupled/txts/var1.txt", var1, delimiter = "\n")
	var2 = objective2.predict(spacing)[1]
	np.savetxt("R/plot_acq/coupled/txts/var2.txt", var2, delimiter = "\n")
	mean_c1 = constraint1.predict(spacing)[0]
	np.savetxt("R/plot_acq/coupled/txts/mean_c1.txt", mean_c1, delimiter = "\n")
	var_c1 = constraint1.predict(spacing)[1]
	np.savetxt("R/plot_acq/coupled/txts/var_c1.txt", var_c1, delimiter = "\n")

	import matplotlib.pyplot as plt

	for i in range(num_results):

		name_exact = 'results/coupled/exact_%d.npy' % (i + 1)
		name_ap = 'results/coupled/ap_%d.npy' % (i + 1)

		if i == 0:
			acq_ext = np.load(name_exact)
			acq_ap = np.load(name_ap)
		else:
			acq_ext = np.vstack((acq_ext, np.load(name_exact)))
			acq_ap = np.vstack((acq_ap, np.load(name_ap)))

	np.savetxt("R/plot_acq/coupled/txts/acq_ext.txt", acq_ext, delimiter = "\n")
	np.savetxt("R/plot_acq/coupled/txts/acq_ap.txt", acq_ap, delimiter = "\n")
	import pdb; pdb.set_trace()

	plt.figure()
	plt.plot(inputs, vals2 * 0, color='black', marker='x', markersize=10, linestyle='None')
	plt.plot(spacing, np.mean(acq_ext, axis = 0), color='red', marker='.', markersize=1)
	plt.plot(spacing, np.mean(acq_ap, axis = 0),  color='blue', marker='.', markersize=1)
        plt.show()

	plt.figure()
	plt.plot(inputs, vals1, color='b', marker='o', markersize=10, linestyle='None')
	plt.plot(inputs, vals2, color='g', marker='s', markersize=10, linestyle='None')
	plt.plot(inputs, vals3, color='r', marker='x', markersize=10, linestyle='None')
	plt.plot(spacing, mean1, 'b.')
	plt.plot(spacing, mean1 + np.sqrt(var1), color = 'b', marker = '.', markersize = 1)
	plt.plot(spacing, mean1 - np.sqrt(var1), color = 'b', marker = '.', markersize = 1)
	plt.plot(spacing, mean2, 'g.')
	plt.plot(spacing, mean2 + np.sqrt(var2), color = 'g', marker = '.', markersize = 1)
	plt.plot(spacing, mean2 - np.sqrt(var2), color = 'g', marker = '.', markersize = 1)
	plt.plot(spacing, mean_c1, 'r.')
        plt.plot(spacing, mean_c1 + np.sqrt(var_c1), color = 'r', marker = '.', markersize = 1)
        plt.plot(spacing, mean_c1 - np.sqrt(var_c1), color = 'r', marker = '.', markersize = 1)
	plt.show()


import scipy.spatial as ss
from scipy.special import digamma,gamma
from math import log,pi
import numpy.random as nr
import numpy as np
import random

def entropy(x,k=3,base=2):
  """ The classic K-L k-nearest neighbor continuous entropy estimator
      x should be a list of vectors, e.g. x = [[1.3],[3.7],[5.1],[2.4]]
      if x is a one-dimensional scalar and we have four samples
  """
  assert k <= len(x)-1, "Set k smaller than num. samples - 1"
  d = len(x[0])
  N = len(x)
  intens = 1e-10 #small noise to break degeneracy, see doc.
  x = [list(p + intens*nr.rand(len(x[0]))) for p in x]
  tree = ss.cKDTree(x)
  nn = [tree.query(point,k+1,p=float('inf'))[0][k] for point in x]
  const = digamma(N)-digamma(k) + d*log(2)
  return (const + d*np.mean(map(log,nn)))/log(base)



if __name__ == "__main__":

#	assert len(sys.argv) > 1
	
#	for i in range(10):
#		test_acquisition_function(str(int(sys.argv[1]) + i))
#		test_acquisition_function_decoupled(str(int(sys.argv[ 1 ]) + i))

	#test_plot_results_decoupled(999)
	test_acquisition_function()
	#test_random_features_sampling()


