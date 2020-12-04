from collections import defaultdict
import autograd.numpy as np
#import numpy as np
import autograd.misc.flatten as flatten
import autograd.scipy.stats as sps
#import scipy.stats as sps
from autograd.numpy.linalg import solve
#from numpy.linalg import solve
from scipy.spatial.distance import cdist
import numpy.linalg   as npla

def my_log(x):
    # The computation of the gradient sometimes fails as a consequence of evaluating log(0.0)
    # Uncomment the code below if that is the case.
    #if np.any(x == 0.0):
    #    import pdb; pdb.set_trace()
    return np.log(x + 1e-10)

SQRT_5 = np.sqrt(5)

def two_by_two_matrix_inverse(a, b, c, d):

        det = a * d - c * b
        a_new = 1.0 / det * d
        b_new = 1.0 / det * -b
        c_new = 1.0 / det * - c
        d_new = 1.0 / det * a

        return a_new, b_new, c_new, d_new

def log_1_minus_exp_x(x):
    #if not isinstance(x, np.ndarray) or x.size==1:
    if x.shape == () or x.size==1:
        return log_1_minus_exp_x_scalar(x)

    assert np.all(x <= 0)

    case1 = x < my_log(1e-6) # -13.8
    case2 = x > -1e-6
    case3 = np.logical_and(x >= my_log(1e-6), x <= -1e-6)
    assert np.all(case1+case2+case3 == 1)

    #These three operations has to be done using two np.where.
    #Test this.
    """
    result = np.zeros(x.shape)
    result[case1] = -np.exp(x[case1])
    with np.errstate(divide='ignore'): # if x is exactly 0, give -inf without complaining
        result[case2] = np.log(-x[case2])
    result[case3] = np.log(1.0-np.exp(x[case3]))

    return result
    """
    return np.where(x < my_log(1e-6), -np.exp(x), np.where(x > -1e-6, my_log(-x), my_log(1.0-np.exp(x))))

def logcdf_robust(x):

    #if isinstance(x, np.ndarray):
    if x.shape != ():
        ret = sps.norm.logcdf(x)
        #ret[x > 5] = -sps.norm.sf(x[x > 5])
        ret = np.where(x <= 5, ret, -sps.norm.sf(x))
    elif x > 5:
        ret = -sps.norm.sf(x)
    else:
        ret = sps.norm.logcdf(x)

    return ret

def two_by_two_symmetric_matrix_product_vector(a, b, c, v_a, v_b):

        return a * v_a + c * v_b, c * v_a + b * v_b

def two_by_two_symmetric_matrix_inverse(a, b, c):

        det = a * b - c * c
        a_new = 1.0 / det * b
        b_new = 1.0 / det * a
        c_new = 1.0 / det * - c

        return a_new, b_new, c_new

def build_unconditioned_predictive_distributions(all_tasks, all_constraints, X):
    mPred         = dict()
    Vpred         = dict()
    VpredInv      = dict()

    mPred_cons         = dict()
    Vpred_cons         = dict()
    VpredInv_cons      = dict()

    for t in all_tasks:
        mPred[t], Vpred[t] = predict(all_tasks[t], X)
        VpredInv[t]        = np.linalg.inv(Vpred[t])

    for t in all_constraints:
        mPred_cons[t], Vpred_cons[t] = predict(all_constraints[t], X)
        VpredInv_cons[t]        = np.linalg.inv(Vpred_cons[t])

    return mPred, Vpred, VpredInv, mPred_cons, Vpred_cons, VpredInv_cons

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

    n_total = X.shape[ 0 ]
    n_pset = pareto_set.shape[ 0 ]

    return X, n_obs, n_pset, n_test, n_total

def original_autograd_cdist(xx1, xx2):
    txx1 = np.tile(xx1, xx2.shape[0]).reshape((xx1.shape[0], xx2.shape[0], xx1.shape[1]))
    txx2 = np.tile(flatten(xx2)[0], xx1.shape[0]).reshape((xx1.shape[0], xx2.shape[0], xx1.shape[1]))
    return np.sum(np.power(txx1-txx2, 2), axis=2)

def autograd_cdist(xx1, xx2):
    return np.outer(np.sum(xx1**2, axis=1), np.ones(xx2.shape[0])) - 2.0 * np.dot(xx1, xx2.T) + \
            np.outer(np.ones(xx1.shape[0]), np.sum(xx2**2, axis=1))

def original_dist2(ls, x1, x2=None):
    if x2 is None:
        # Find distance with self for x1.

        # Rescale.
        xx1 = x1 / ls
        xx2 = xx1

    else:
        # Rescale.
        xx1 = x1 / ls
        xx2 = x2 / ls

    return original_autograd_cdist(xx1, xx2)

def dist2(ls, x1, x2=None):
    if x2 is None:
        # Find distance with self for x1.

        # Rescale.
        xx1 = x1 / ls
        xx2 = xx1

    else:
        # Rescale.
        xx1 = x1 / ls
        xx2 = x2 / ls

    return autograd_cdist(xx1, xx2)

def original_cov(ls_values, inputs):
        return original_cross_cov(ls_values, inputs, inputs)

def cov(ls_values, inputs):
        return cross_cov(ls_values, inputs, inputs, squared=True)

def original_cross_cov(ls_values, inputs_1, inputs_2):
        r2  = np.abs(original_dist2(ls_values, inputs_1, inputs_2))
        r = np.sqrt(r2)
        cov = (1.0 + SQRT_5*r + (5.0/3.0)*r2) * np.exp(-SQRT_5*r)
        return cov

def cross_cov(ls_values, inputs_1, inputs_2, squared=False):
        r2  = np.abs(dist2(ls_values, inputs_1, inputs_2))
        r2 = np.where(r2==0.0, r2 + 1e-10, r2)
        r = np.sqrt(r2)

        cov = (1.0 + SQRT_5*r + (5.0/3.0)*r2) * np.exp(-SQRT_5*r)
        return cov

def predict(gp, xstar):
        x = gp.inputs
        y = gp.values
        mean = gp.mean.value
        scale = gp.params['amp2'].value

        #Se le anaden los noise_scale y el jitter para emular a la suma del matern+scale_kernel+noise_kernel de Spearmint.
        cov_f_f = cov(gp.params['ls'].value, xstar) * scale + np.eye(len(xstar)) * gp.stability_noise_kernel.noise.value
        #cov_f_f_o = original_cov(gp.params['ls'].value, xstar) * scale + np.eye(len(xstar)) * gp.stability_noise_kernel.noise.value
        #print(np.abs(cov_f_f-cov_f_f_o))
        #assert np.all(np.abs(cov_f_f-cov_f_f_o)) < 1e-10)
        #if np.any(np.abs(cov_f_f-cov_f_f_o) > 1e-10):
            #import pdb; pdb.set_trace();
        cov_y_f = cross_cov(gp.params['ls'].value, x, xstar) * scale
        #cov_y_f_o = original_cross_cov(gp.params['ls'].value, x, xstar) * scale
        #print(np.abs(cov_y_f-cov_y_f_o))
        #assert np.all(np.abs(cov_y_f-cov_y_f_o)) < 1e-10)
        #if np.any(np.abs(cov_y_f-cov_y_f_o) > 1e-10):
            #import pdb; pdb.set_trace();
        cov_y_y = cov(gp.params['ls'].value, x) * scale  + np.eye(len(y)) * gp.stability_noise_kernel.noise.value
        #cov_y_y_o = original_cov(gp.params['ls'].value, x) * scale  + np.eye(len(y)) * gp.stability_noise_kernel.noise.value
        #print(np.abs(cov_y_y-cov_y_y_o))
        #assert np.all(np.abs(cov_y_y-cov_y_y_o)) < 1e-10)
        #if np.any(np.abs(cov_y_y-cov_y_y_o) > 1e-10):
            #import pdb; pdb.set_trace();

        InvMat = solve(cov_y_y, cov_y_f).T
        pred_mean = mean + np.dot(InvMat, y - mean)
        #pred_mean = mean + np.matmul(cov_y_f ,np.matmul(np.linalg.inv(cov_y_y), y - mean))
        pred_cov = cov_f_f - np.dot(InvMat, cov_y_f)
        #pred_cov = cov_f_f - np.matmul(cov_y_f.T, np.matmul(np.linalg.inv(cov_y_y), cov_y_f.T))

        return pred_mean, pred_cov

def compute_unconstrained_variances_and_init_acq_fun(obj_models_dict, cand, con_models): #, log, fun_log):
            unconstrainedVariances = dict()
            constrainedVariances = dict()
            acq = dict()

            for obj in obj_models_dict:
                unconstrainedVariances[ obj ] = predict(obj_models_dict[ obj ], cand)[ 1 ]
                #fun_log('predict', log, {'obj_models_dict[ obj ]': obj_models_dict[ obj ], \
                 #                        'cand' : cand, 'unconstrainedVariances[ obj ]': unconstrainedVariances[ obj ]})
                unconstrainedVariances[ obj ] = unconstrainedVariances[ obj ] + np.eye(unconstrainedVariances[ obj ].shape[0]) * obj_models_dict[ obj ].noise_value()
            for cons in con_models:
                unconstrainedVariances[ cons ] = predict(con_models[ cons ], cand)[ 1 ]
                #fun_log('predict', log, {'con_models[ cons ]': con_models[ cons ], \
                 #                        'cand' : cand, 'unconstrainedVariances[ cons ]': unconstrainedVariances[ cons ]})
                unconstrainedVariances[ cons ] = unconstrainedVariances[ cons ] + np.eye(unconstrainedVariances[ cons ].shape[0]) * con_models[ cons ].noise_value()
            for t in unconstrainedVariances:
                acq[t] = 0

            return acq, unconstrainedVariances, constrainedVariances

def update_full_marginals(a):
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
                for cons in all_constraints:

                        # Summing all the factors into the diagonal. Reescribir.

                        vTilde_cons = np.diag(np.append(np.append(np.sum(a['a_c_hfhat'][ :, : , ntask ], axis = 1), \
                                        np.sum(a['c_c_hfhat'][ :, : , ntask ], axis = 1) + a['ehfhat'][ :, ntask ]), \
                                        np.sum(a['g_c_hfhat'][ :, : , ntask ], axis = 1)))

                        mTilde_cons = np.append(np.append(np.sum(a['b_c_hfhat'][ :, : , ntask ], axis = 1), \
                                np.sum(a['d_c_hfhat'][ :, : , ntask ], axis = 1) + a['fhfhat'][ :, ntask ]), \
                                 np.sum(a['h_c_hfhat'][ :, : , ntask ], axis = 1))

                        # Natural parameter conversion and update of the marginal variance matrices.

                        a['Vinv_cons'][cons] = a['VpredInv_cons'][cons] + vTilde_cons
                        a['V_cons'][cons] = np.linalg.inv(a['VpredInv_cons'][cons] + vTilde_cons)

                        # Natural parameter conversion and update of the marginal mean vector.

                        a['m_nat_cons'][cons] = np.dot(a['VpredInv_cons'][cons], a['mPred_cons'][cons]) + mTilde_cons
                        a['m_cons'][cons] = np.dot(a['V_cons'][cons], a['m_nat_cons'][ cons ])

                        ntask = ntask + 1
                ntask = 0
                for obj in all_tasks:

                        vTilde = np.zeros((n_total,n_total))

                        vTilde = np.zeros((n_total,n_total))
                        diagVtilde = np.identity(n_total) * np.append(np.append(np.sum(a['ahfhat'][ :, : , ntask, 0, 0 ], axis = 1), \
                                            np.sum(a['ahfhat'][ :, : , ntask, 1, 1 ], axis = 0) + \
                                            np.sum(a['chfhat'][ :, : , ntask, 0, 0 ], axis = 1) + \
                                            np.sum(a['chfhat'][ :, : , ntask, 1, 1 ], axis = 0) + \
                                            np.sum(a['ghfhat'][ :, : , ntask, 1, 1 ], axis = 0)), \
                                            np.sum(a['ghfhat'][ :, : , ntask, 0, 0 ], axis = 1))

                        #Building full matrices from blocks.

                        block_2 = a['chfhat'][ :, : , ntask, 0, 1 ] + a['chfhat'][ :, : , ntask, 1, 0 ].T
                        block_2 = np.hstack([np.zeros((n_pset, n_obs)), block_2])
                        block_2 = np.hstack([block_2, np.zeros((n_pset, n_test))])
                        block_2 = np.vstack([np.zeros((n_obs, n_total)), block_2])
                        block_2 = np.vstack([block_2, np.zeros((n_test, n_total))])

                        block_3 = a['ahfhat'][ :, :, ntask, 0, 1]
                        block_3 = np.hstack([np.zeros((n_obs, n_obs)), block_3])
                        block_3 = np.hstack([block_3, np.zeros((n_obs, n_test))])
                        block_3 = np.vstack([block_3, np.zeros((n_pset + n_test, n_total))])        

                        block_4 = a['ahfhat'][ :, :, ntask, 0, 1].transpose()
                        block_4 = np.hstack([block_4, np.zeros((n_pset, n_pset+n_test))])
                        block_4 = np.vstack([np.zeros((n_obs, n_total)), block_4])
                        block_4 = np.vstack([block_4, np.zeros((n_test, n_total))])
                        
                        block_5 = a['ghfhat'][ :, :, ntask, 0, 1]
                        block_5 = np.hstack([np.zeros((n_test, n_obs)), block_5])
                        block_5 = np.hstack([block_5, np.zeros((n_test, n_test))])
                        block_5 = np.vstack([np.zeros((n_obs+n_pset, n_total)), block_5])
    
                        block_6 = a['ghfhat'][ :, :, ntask, 0, 1].transpose()
                        block_6 = np.hstack([np.zeros((n_pset, n_obs + n_pset)), block_6])
                        block_6 = np.vstack([np.zeros((n_obs, n_total)), block_6])
                        block_6 = np.vstack([block_6, np.zeros((n_test, n_total))])
            
                        #Adding to final matrix all the blocks.

                        vTilde += diagVtilde   
                        vTilde += block_2
                        vTilde += block_3
                        vTilde += block_4
                        vTilde += block_5
                        vTilde += block_6 
                           
                        a['Vinv'][obj] = a['VpredInv'][obj] + vTilde
                        a['V'][obj] = np.linalg.inv(a['VpredInv'][obj] + vTilde)

                        mTilde = np.append(np.append(np.sum(a['bhfhat'][ :, : , ntask, 0 ], axis = 1),
                                np.sum(a['bhfhat'][ :, : , ntask, 1 ], axis = 0) + np.sum(a['hhfhat'][ :, : , ntask, 1 ], axis = 0) +\
                                np.sum(a['dhfhat'][ :, : , ntask, 0 ], axis = 1) + np.sum(a['dhfhat'][ :, : , ntask, 1 ], axis = 0)), \
                                np.sum(a['hhfhat'][ :, : , ntask, 0 ], axis = 1))

                        a['m_nat'][obj] = np.dot(a['VpredInv'][obj], a['mPred'][obj]) + mTilde
                        a['m'][obj] = np.dot(a['V'][obj], a['m_nat'][ obj ])

                        ntask = ntask + 1

                return a

def get_test_predictive_distributions(a):
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

def compute_PPESMOC_approximation(predictionEP, obj_models_dict, con_models, unconstrainedVariances, constrainedVariances, acq):

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

    m_pset = np.array([])
    #m_pset = np.zeros((q, n_pset, n_test))
    m_test = np.array([])
    #m_test = np.zeros((q, n_pset, n_test))
    v_pset = np.array([])
    #v_pset = np.zeros((q, n_pset, n_test))
    v_test = np.array([])
    #v_test = np.zeros((q, n_pset, n_test))
    v_cov = np.array([])
    #v_cov = np.zeros((q, n_pset, n_test))

    # Data structures for constraint npset nobs cavities (c_a, c_b).

    c_m = np.array([])
    #c_m = np.zeros((c, n_pset, n_test))
    c_v = np.array([])
    #c_v = np.zeros((c, n_pset, n_test))

    # Update marginals: a['m'] , a['V']
    #n_task = 0

    for obj in all_tasks: #OK
        m_test = np.append(m_test, np.tile(a['m'][ obj ][ n_obs + n_pset : n_total ], n_pset).reshape((n_pset, n_test))) #OK
        #m_test[ n_task, :, : ] = np.tile(a['m'][ obj ][ n_obs + n_pset : n_total ], n_pset).reshape((n_pset, n_test))
        m_pset = np.append(m_pset, np.tile(a['m'][ obj ][ n_obs : n_obs + n_pset ], n_test).reshape((n_test, n_pset)).T) #OK
        #m_pset[ n_task, :, : ] = np.tile(a['m'][ obj ][ n_obs : n_obs + n_pset ], n_test).reshape((n_test, n_pset)).T
        v_cov = np.append(v_cov, a['V'][ obj ][ n_obs : n_obs + n_pset, n_obs + n_pset : n_total ])
        #v_cov[ n_task, :, : ] = a['V'][ obj ][ n_obs : n_obs + n_pset, n_obs + n_pset : n_total ]
        v_test = np.append(v_test, np.tile(np.diag(a['V'][ obj ])[ n_obs + n_pset : n_total ], n_pset).reshape((n_pset, n_test))) #CASO 1: OK.
        #v_test[ n_task, :, : ] = np.tile(np.diag(a['V'][ obj ])[ n_obs + n_pset : n_total ], n_pset).reshape((n_pset, n_test))
        v_pset = np.append(v_pset, np.tile(np.diag(a['V'][ obj ])[ n_obs : n_obs + n_pset ], n_test).reshape((n_test, n_pset)).T) #CASO 2: OK.
        #v_pset[ n_task, :, : ] = np.tile(np.diag(a['V'][ obj ])[ n_obs : n_obs + n_pset ], n_test).reshape((n_test, n_pset)).T
        #n_task += 1

    m_test = m_test.reshape((q, n_pset, n_test))
    m_pset = m_pset.reshape((q, n_pset, n_test))
    v_cov = v_cov.reshape((q, n_pset, n_test))
    v_test = v_test.reshape((q, n_pset, n_test))
    v_pset = v_pset.reshape((q, n_pset, n_test))
    #n_task = 0

    for cons in all_constraints: #OK
        c_m = np.append(c_m, np.tile(a['m_cons'][ cons ][ n_obs + n_pset : n_total ], n_pset))
        #c_m[ n_task, :, : ] = a['m_cons'][ cons ][ n_obs + n_pset : n_total ]
        c_v = np.append(c_v, np.tile(np.diag(a['V_cons'][ cons ])[ n_obs + n_pset : n_total ], n_pset))
        #c_v[ n_task, :, : ] = np.diag(a['V_cons'][ cons ])[ n_obs + n_pset : n_total ]
        #n_task += 1

    c_m = c_m.reshape((c, n_pset, n_test)) #OK
    c_v = c_v.reshape((c, n_pset, n_test)) #OK

    vTilde_test = a['ghfhat'][ :, :, :, 0, 0 ].T #OK
    vTilde_pset = a['ghfhat'][ :, :, :, 1, 1 ].T #OK
    vTilde_cov = a['ghfhat'][ :, :, :, 0, 1 ].T #OK
    mTilde_test = a['hhfhat'][ :, :, :, 0 ].T #OK
    mTilde_pset = a['hhfhat'][ :, :, :, 1 ].T #OK

    vTilde_test_cons = a['g_c_hfhat'][:, :, :].T #OK
    mTilde_test_cons = a['h_c_hfhat'][:, :, :].T #OK
     # Obtaining cavities.
    inv_v_test, inv_v_pset, inv_v_cov = two_by_two_symmetric_matrix_inverse(v_test, v_pset, v_cov) #OK
    inv_c_v = 1.0 / c_v #OK

    inv_vOld_test = inv_v_test - vTilde_test #OK
    inv_vOld_pset = inv_v_pset - vTilde_pset #OK
    inv_vOld_cov =  inv_v_cov - vTilde_cov #OK
    inv_c_vOld = inv_c_v - vTilde_test_cons #OK

    vOld_test, vOld_pset, vOld_cov = two_by_two_symmetric_matrix_inverse(inv_vOld_test, inv_vOld_pset, inv_vOld_cov) #OK
    c_vOld = 1.0 / inv_c_vOld #OK

    mOld_test, mOld_pset = two_by_two_symmetric_matrix_product_vector(inv_v_test, inv_v_pset, inv_v_cov, m_test, m_pset) #OK
    mOld_test = mOld_test - mTilde_test #OK
    mOld_pset = mOld_pset - mTilde_pset #OK
    mOld_test, mOld_pset  = two_by_two_symmetric_matrix_product_vector(vOld_test, vOld_pset, vOld_cov, mOld_test, mOld_pset) #OK

    c_mOld = c_vOld * (c_m / c_v - mTilde_test_cons)

    # Computing factors.

    s = vOld_pset + vOld_test - 2 * vOld_cov
    s_cons = c_vOld

    if np.any(vOld_pset < 0):
        #raise npla.linalg.LinAlgError("Negative variance in the sqrt!")
        vOld_pset = np.where(vOld_pset < 0, -vOld_pset, vOld_pset)
        print('Careful!!! Negative variances have appear before sqrt!')

    if np.any(vOld_test < 0):
        vOld_test = np.where(vOld_test < 0, -vOld_test, vOld_test)
        #raise npla.linalg.LinAlgError("Negative variance in the sqrt!")
        print('Careful!!! Negative variances have appear before sqrt!')

    if np.any(c_vOld < 0):
        c_vOld = np.where(c_vOld < 0, -c_vOld, c_vOld)
        #raise npla.linalg.LinAlgError("Negative value in the sqrt!")
        print('Careful!!! Negative variances have appear before sqrt!')

    alpha_cons = c_mOld / np.sqrt(c_vOld) #OK

    scale = 1.0 - 1e-4
    while np.any(s / (vOld_pset + vOld_test) < 1e-6): #OK
        scale = scale**2
        s = vOld_pset + vOld_test - 2 * vOld_cov * scale

    s = np.where(s==0.0, 1.0, s)
    ss = np.sqrt(s)
    ss = np.where(ss==1.0, 1e-15, ss)
    alpha = (mOld_test - mOld_pset) / ss * sgn #OK

    log_phi = logcdf_robust(alpha)
    log_phi_cons = logcdf_robust(alpha_cons) #OK

    logZ_orig = log_1_minus_exp_x(np.sum(log_phi, axis = 0)) #OK

    #Hay que sustituir este bloque de codigo por un np.where.
    logZ_orig = np.where(logZ_orig == -np.inf, logcdf_robust(-np.min(alpha, axis = 0)), logZ_orig) #OK

    logZ_term1 = np.sum(log_phi_cons, axis = 0) + logZ_orig #OK
    logZ_term2 = log_1_minus_exp_x(np.sum(log_phi_cons, axis = 0)) #OK

    logZ_term2 = np.where(logZ_term2 == -np.inf, logcdf_robust(-np.min(alpha_cons, axis = 0)), logZ_term2) #OK

    max_value = np.maximum(logZ_term1, logZ_term2) #OK

    logZ = my_log(np.exp(logZ_term1 - max_value) + np.exp(logZ_term2 - max_value)) + max_value
    for i in range(q-1):
        logZ = np.hstack((logZ, my_log(np.exp(logZ_term1 - max_value) + np.exp(logZ_term2 - max_value)) + max_value))
    logZ = logZ.reshape((n_pset, q, n_test)).swapaxes(0, 1)
    #logZ = np.tile(np.log(np.exp(logZ_term1 - max_value) + np.exp(logZ_term2 - max_value)) + \
        #max_value, q).reshape((n_pset, q, n_test)).swapaxes(0, 1) #SOSPECHOSO, SE PUEDE SIMULAR.
    
    logZ_cons = my_log(np.exp(logZ_term1 - max_value) + np.exp(logZ_term2 - max_value)) + max_value
    for i in range(c-1):
        logZ_cons = np.hstack((logZ_cons, my_log(np.exp(logZ_term1 - max_value) + np.exp(logZ_term2 - max_value)) + max_value))
    logZ_cons = logZ_cons.reshape((n_pset, c, n_test)).swapaxes(0, 1)
    #logZ_cons = np.tile(np.log(np.exp(logZ_term1 - max_value) + np.exp(logZ_term2 - max_value)) + \
        #max_value, c).reshape((n_pset, c, n_test)).swapaxes(0, 1) #OK

    log_phi_sum = np.sum(log_phi, axis = 0)
    for i in range(q-1):
        log_phi_sum = np.hstack((log_phi_sum, np.sum(log_phi, axis = 0)))
    log_phi_sum = log_phi_sum.reshape((n_pset, q, n_test)).swapaxes(0, 1)
    #log_phi_sum = np.tile(np.sum(log_phi, axis = 0), q).reshape((n_pset, q, n_test)).swapaxes(0, 1) #SOSPECHOSO, SE PUEDE SIMULAR.

    log_phi_sum_cons = np.sum(log_phi_cons, axis = 0)
    for i in range(q-1):
        log_phi_sum_cons = np.hstack((log_phi_sum_cons, np.sum(log_phi_cons, axis = 0)))
    log_phi_sum_cons = log_phi_sum_cons.reshape((n_pset, q, n_test)).swapaxes(0, 1)
    #log_phi_sum_cons = np.tile(np.sum(log_phi_cons, axis = 0), q).reshape((n_pset, q, n_test)).swapaxes(0, 1) #OK

    ratio = - np.exp(sps.norm.logpdf(alpha) - logZ + log_phi_sum - logcdf_robust(alpha) + log_phi_sum_cons)

    logZ_orig_cons = logZ_orig
    for i in range(c-1):
        logZ_orig_cons = np.hstack((logZ_orig_cons, logZ_orig))
    logZ_orig_cons = logZ_orig_cons.reshape((n_pset, c, n_test)).swapaxes(0, 1)
    #logZ_orig_cons = np.tile(logZ_orig, c).reshape((n_pset, c, n_test)).swapaxes(0, 1) #OK?
    log_phi_sum_cons = np.sum(log_phi_cons, axis = 0)
    for i in range(c-1):
        log_phi_sum_cons = np.hstack((log_phi_sum_cons, np.sum(log_phi_cons, axis = 0)))
    log_phi_sum_cons = log_phi_sum_cons.reshape((n_pset, c, n_test)).swapaxes(0, 1)
    #log_phi_sum_cons = np.tile(np.sum(log_phi_cons, axis = 0), c).reshape((n_pset, c, n_test)).swapaxes(0, 1) #OK

    ratio_cons = np.exp(sps.norm.logpdf(alpha_cons) - logZ_cons + logZ_orig_cons + log_phi_sum_cons - logcdf_robust(alpha_cons)) - \
        np.exp(sps.norm.logpdf(alpha_cons) - logZ_cons + log_phi_sum_cons - logcdf_robust(alpha_cons)) #OK
    
    s = np.where(s==0.0, 1.0, s)
    ss = np.sqrt(s)  
    ss = np.where(ss==1.0, 1e-15, ss)
    dlogZdmfOld_test = ratio / ss * sgn
    dlogZdmfOld_pset = ratio / ss * -1.0 * sgn

    dlogZdmfOld_test2 = - ratio / s * (alpha + ratio)
    dlogZdmfOld_pset2 = - ratio / s * (alpha + ratio)
    dlogZdmfOld_cov2 = - ratio / s * (alpha + ratio) * -1.0

    s_cons = np.where(s_cons==0.0, 1.0, s_cons)
    sc = np.sqrt(s_cons)  
    sc = np.where(sc==1.0, 1e-15, sc)
    dlogZdmcOld = ratio_cons / sc #OK
    dlogZdmcOld2 = - ratio_cons / s_cons * (alpha_cons + ratio_cons) #OK

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
        dlogZdmfOld_pset2, dlogZdmfOld_cov2, mOld_test, mOld_pset) #OK

    v_1 = dlogZdmfOld_test - v_1 #OK
    v_2 = dlogZdmfOld_pset - v_2 #OK
    mTilde_test_new = v_1 * a_inv + v_2 * c_inv #OK
    mTilde_pset_new = v_1 * b_inv + v_2 * d_inv #OK

    vTilde_cons =  - dlogZdmcOld2 / (1.0 + dlogZdmcOld2 * c_vOld) #OK
    mTilde_cons = (dlogZdmcOld - c_mOld * dlogZdmcOld2) / (1.0 + dlogZdmcOld2 * c_vOld)  #OK
   
    if no_negative_variances_nor_nands == True:  #OK full.

        #finite = np.logical_and(np.logical_and(np.logical_and(np.logical_and(np.isfinite(vTilde_test_new), np.isfinite(vTilde_pset_new)), \
            #np.isfinite(vTilde_cov_new)), np.isfinite(mTilde_test_new)), np.isfinite(mTilde_pset_new))

        #c_finite = np.logical_and(np.isfinite(vTilde_cons), np.isfinite(mTilde_cons))

        #neg1 = np.where(np.logical_or(np.logical_not(finite), vTilde_test_new < 0))
        #neg2 = np.where(np.logical_or(np.logical_not(finite), vTilde_pset_new < 0))
        #c_neg = np.where(np.logical_or(np.logical_not(c_finite), vTilde_cons < 0))

        o_cond = np.logical_or(vTilde_test_new < 0, vTilde_pset_new < 0, \
            np.logical_not(np.logical_and(np.logical_and(np.logical_and(np.logical_and( \
            np.isfinite(vTilde_test_new), np.isfinite(vTilde_pset_new)), np.isfinite(vTilde_cov_new)), \
            np.isfinite(mTilde_test_new)), np.isfinite(mTilde_pset_new))))

        c_cond = np.logical_or(np.logical_not(np.logical_and(np.isfinite(vTilde_cons), np.isfinite(mTilde_cons))), vTilde_cons < 0)

        vTilde_test_new = np.where(o_cond, 0.0, vTilde_test_new)
        vTilde_pset_new = np.where(o_cond, 0.0, vTilde_pset_new)
        vTilde_cov_new = np.where(o_cond, 0.0, vTilde_cov_new)
        mTilde_test_new = np.where(o_cond, 0.0, mTilde_test_new)
        mTilde_pset_new = np.where(o_cond, 0.0, mTilde_pset_new)
        vTilde_cons = np.where(c_cond, 0.0, vTilde_cons)
        mTilde_cons = np.where(c_cond, 0.0, mTilde_cons)

        #vTilde_test_new[ neg1 ] = 0.0
        #vTilde_test_new[ neg2 ] = 0.0
        #vTilde_pset_new[ neg1 ] = 0.0
        #vTilde_pset_new[ neg2 ] = 0.0
        #vTilde_cov_new[ neg1 ] = 0.0
        #vTilde_cov_new[ neg2 ] = 0.0
        #mTilde_test_new[ neg1 ] = 0.0
        #mTilde_test_new[ neg2 ] = 0.0
        #mTilde_pset_new[ neg1 ] = 0.0
        #mTilde_pset_new[ neg2 ] = 0.0
        #vTilde_cons[ c_neg ] = 0.0
        #mTilde_cons[ c_neg ] = 0.0

    # We do the actual update
    g_c_hfHatNew = vTilde_cons #OK
    h_c_hfHatNew = mTilde_cons #OK

    ghfhat = np.array([])
    ghfhata = np.array([])
    hhfhat = np.array([])
    g00 =  vTilde_test_new.T * damping + (1 - damping) * a['ghfhat'][ :, :, :, 0, 0 ]
    g01 = vTilde_cov_new.T * damping + (1 - damping) * a['ghfhat'][ :, :, :, 0, 1 ]
    g10 = vTilde_cov_new.T * damping + (1 - damping) * a['ghfhat'][ :, :, :, 1, 0 ]
    g11 = vTilde_pset_new.T * damping + (1 - damping) * a['ghfhat'][ :, :, :, 1, 1 ]
    h0 = mTilde_test_new.T * damping + (1 - damping) * a['hhfhat'][ :, :, :, 0 ] #OK
    h1 = mTilde_pset_new.T * damping + (1 - damping) * a['hhfhat'][ :, :, :, 1 ] #OK

    #for tp in range(n_test):
    #    for pp in range(n_pset):
    #        for qp in range(q):
    #            #ES UN TILE! Segun la traza de error de autograd parece que un TILE la esta liando seriamente, y esta relacionado con ghfhat.
    #            ghfhat = np.append(ghfhat, g00[tp, pp, qp]) 
    #            ghfhat = np.append(ghfhat, g01[tp, pp, qp]) 
    #            ghfhat = np.append(ghfhat, g10[tp, pp, qp]) 
    #            ghfhat = np.append(ghfhat, g11[tp, pp, qp]) 
    #            hhfhat = np.append(hhfhat, np.array([h0[tp, pp, qp], h1[tp, pp, qp]])) #OK

    ghfhat_new = np.stack((np.stack((g00, g01), axis = 3), np.stack((g01, g11), axis = 3)), axis = 3)
    hhfhat_new = np.stack((h0, h1), axis = 3)

    #ghfhat = ghfhat.reshape((n_test, n_pset, q, 2, 2))
    #ghfhat = ghfhat.reshape((n_test, n_pset, q, 2, 2))
    #hhfhat = hhfhat.reshape((n_test, n_pset, q, 2)) #OK
    #a['ghfhat'] = ghfhat
    #a['hhfhat'] = hhfhat
    a['ghfhat'] = ghfhat_new
    a['hhfhat'] = hhfhat_new
    #a['ghfhat'][ :, :, :, 0, 0 ] = vTilde_test_new.T * damping + (1 - damping) * a['ghfhat'][ :, :, :, 0, 0 ]
    #a['ghfhat'][ :, :, :, 1, 1 ] = vTilde_pset_new.T * damping + (1 - damping) * a['ghfhat'][ :, :, :, 1, 1 ]
    #a['ghfhat'][ :, :, :, 0, 1 ] = vTilde_cov_new.T * damping + (1 - damping) * a['ghfhat'][ :, :, :, 0, 1 ]
    #a['ghfhat'][ :, :, :, 1, 0 ] = vTilde_cov_new.T * damping + (1 - damping) * a['ghfhat'][ :, :, :, 1, 0 ]
    #a['hhfhat'][ :, :, :, 0 ] = mTilde_test_new.T * damping + (1 - damping) * a['hhfhat'][ :, :, :, 0 ]
    #a['hhfhat'][ :, :, :, 1 ] = mTilde_pset_new.T * damping + (1 - damping) * a['hhfhat'][ :, :, :, 1 ]
    a['g_c_hfhat'] = g_c_hfHatNew.T * damping + (1 - damping) * a['g_c_hfhat'][ :, :, : ]
    #a['g_c_hfhat'][ :, :, : ] = g_c_hfHatNew.T * damping + (1 - damping) * a['g_c_hfhat'][ :, :, : ]
    a['h_c_hfhat'] = h_c_hfHatNew.T * damping + (1 - damping) * a['h_c_hfhat'][ :, :, : ]
    #a['h_c_hfhat'][ :, :, : ] = h_c_hfHatNew.T * damping + (1 - damping) * a['h_c_hfhat'][ :, :, : ]

    return a

def compute_acq_fun_wrt_test_points(X_test, obj_model_dict, con_models_dict, pareto_set, info_gps, tasks):#, log, fun_log):

            #Execute necessary functionality for the test factors update and the recomputation of the marginals and PPESMOC.
            acq, unconstrainedVariances, constrainedVariances = compute_unconstrained_variances_and_init_acq_fun \
                                                                                (obj_model_dict, X_test, con_models_dict)#, log, fun_log)
            #Pareto set samples loop.
            acqs = {}
            for ps in pareto_set.keys():
                pareto_set_sample = pareto_set[ps]
                info_gps_ps = info_gps[ps] #Verificar que los factores con los que empezamos son los que termina la ejecucion normal.
                X, n_obs, n_pset, n_test, n_total = build_set_of_points_that_conditions_GPs(obj_model_dict, con_models_dict, \
                                pareto_set_sample, X_test)
                mPred, Vpred, VpredInv, mPred_cons, Vpred_cons, VpredInv_cons = \
                                build_unconditioned_predictive_distributions(obj_model_dict, con_models_dict, X)
                #Modify information of a according to previous computations.
                q = len(obj_model_dict)
                c = len(con_models_dict)
                info_gps_ps['ghfhat'] =  np.zeros((n_test, n_pset, q, 2, 2))
                info_gps_ps['hhfhat'] =  np.zeros((n_test, n_pset, q, 2))
                info_gps_ps['g_c_hfhat'] = np.zeros((n_test, n_pset, c))
                info_gps_ps['h_c_hfhat'] = np.zeros((n_test, n_pset, c))
                info_gps_ps['m'] = defaultdict(lambda: np.zeros(n_total))
                info_gps_ps['m_nat'] = defaultdict(lambda: np.zeros(n_total))
                info_gps_ps['V'] = defaultdict(lambda: np.zeros((n_total, n_total)))
                info_gps_ps['Vinv'] = defaultdict(lambda: np.zeros((n_total, n_total)))
                info_gps_ps['m_cons'] = defaultdict(lambda: np.zeros(n_total))
                info_gps_ps['m_nat_cons'] = defaultdict(lambda: np.zeros(n_total))
                info_gps_ps['V_cons'] = defaultdict(lambda: np.zeros((n_total, n_total)))
                info_gps_ps['Vinv_cons'] = defaultdict(lambda: np.zeros((n_total, n_total)))
                info_gps_ps['n_obs'] = n_obs
                info_gps_ps['n_pset'] = n_pset
                info_gps_ps['n_test'] = n_test
                info_gps_ps['n_total'] = n_total
                info_gps_ps['mPred'] = mPred
                info_gps_ps['VPred'] = Vpred
                info_gps_ps['VpredInv'] = VpredInv
                info_gps_ps['mPred_cons'] = mPred_cons
                info_gps_ps['Vpred_cons'] = Vpred_cons
                info_gps_ps['VpredInv_cons'] = VpredInv_cons
                info_gps_ps['X'] = X

                #Creo que hay que cambiar mas cosas en infogps, quiza como los factores de test? Pero no estoy seguro.
                #Por precaucion, dejar asi.

                #Execute EP modification of the test factors and the PPESMOC approximation.
                #First we have to do an EP update marginals.
                info_gps_ps = update_full_marginals(info_gps_ps)
                info_gps_ps = update_full_Factors_only_test_factors(info_gps_ps, 0.1, \
                                minimize = True, no_negative_variances_nor_nands = True)
                
                info_gps_ps = update_full_marginals(info_gps_ps)
                predictionEP = get_test_predictive_distributions(info_gps_ps)[0]
                #Hay que dar la suma de las adquisiciones tambien.
                acqs[ps] = compute_PPESMOC_approximation(predictionEP, obj_model_dict, con_models_dict, \
                                                unconstrainedVariances, constrainedVariances, acq)

            #Sumar las adquisiciones de los puntos de pareto y dividirlas entre el numero de puntos de pareto.
            final_acqs_dict = dict.fromkeys(acqs[acqs.keys()[0]].keys())
            num_samples = len(acqs)
            for ps_sample in acqs:
                ps_acq = acqs[ps_sample]
                for bb_acq in ps_acq:
                    if final_acqs_dict[bb_acq] is None:
                        final_acqs_dict[bb_acq] = ps_acq[bb_acq]
                    else:
                        final_acqs_dict[bb_acq] += ps_acq[bb_acq]
            final_acqs_dict.update((x, y / num_samples) for x, y in final_acqs_dict.items())
            #fun_log('BB acqs Autograd', log, {'acq' : final_acqs_dict})
            total_acq = 0.0
            for task in tasks:
                total_acq += final_acqs_dict[ task ]
            #Sumar las adquisiciones de cada BB.
            #fun_log('total_acq Autograd', log, {'acq' : total_acq})
            return total_acq
