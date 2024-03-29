from collections import defaultdict
import autograd.numpy as np
import autograd.misc.flatten as flatten
import autograd.scipy.stats as sps
from autograd.numpy.linalg import solve
from scipy.spatial.distance import cdist
import numpy.linalg   as npla

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

    case1 = x < np.log(1e-6) # -13.8
    case2 = x > -1e-6
    case3 = np.logical_and(x >= np.log(1e-6), x <= -1e-6)
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
    return np.where(x < np.log(1e-6), -np.exp(x), np.where(x > -1e-6, np.log(-x), np.log(1.0-np.exp(x))))

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

def autograd_cdist(xx1, xx2):
    txx1 = np.tile(xx1, xx2.shape[0]).reshape((xx1.shape[0], xx2.shape[0], xx1.shape[1]))
    txx2 = np.tile(flatten(xx2)[0], xx1.shape[0]).reshape((xx1.shape[0], xx2.shape[0], xx1.shape[1]))
    return np.sum(np.power(txx1-txx2, 2), axis=2)

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

def cov(ls_values, inputs):
        return cross_cov(ls_values, inputs, inputs)

def cross_cov(ls_values, inputs_1, inputs_2):
        r2  = np.abs(dist2(ls_values, inputs_1, inputs_2))
        #r = r2
        r2 = np.where(r2==0, 1e-15, r2)
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
        cov_y_f = cross_cov(gp.params['ls'].value, x, xstar) * scale
        cov_y_y = cov(gp.params['ls'].value, x) * scale  + np.eye(len(y)) * gp.stability_noise_kernel.noise.value

        pred_mean = mean + np.dot(solve(cov_y_y, cov_y_f).T, y - mean)
        #pred_mean = mean + np.matmul(cov_y_f ,np.matmul(np.linalg.inv(cov_y_y), y - mean))
        pred_cov = cov_f_f - np.dot(solve(cov_y_y, cov_y_f).T, cov_y_f)
        #pred_cov = cov_f_f - np.matmul(cov_y_f.T, np.matmul(np.linalg.inv(cov_y_y), cov_y_f.T))

        return pred_mean, pred_cov

def compute_unconstrained_variances_and_init_acq_fun(obj_models_dict, cand, con_models, log, fun_log):
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

                        #vTilde_cons = np.zeros((n_total,n_total))
                        vTilde_cons = np.diag(np.append(np.append(np.sum(a['a_c_hfhat'][ :, : , ntask ], axis = 1), \
                                        np.sum(a['c_c_hfhat'][ :, : , ntask ], axis = 1) + a['ehfhat'][ :, ntask ]), \
                                        np.sum(a['g_c_hfhat'][ :, : , ntask ], axis = 1)))

                        #vTilde_cons[ np.eye(n_total).astype('bool') ] = np.append(np.append(np.sum(a['a_c_hfhat'][ :, : , ntask ], axis = 1), \
                                #np.sum(a['c_c_hfhat'][ :, : , ntask ], axis = 1) + a['ehfhat'][ :, ntask ]), \
                                #np.sum(a['g_c_hfhat'][ :, : , ntask ], axis = 1))

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

                        #vTilde = np.zeros((n_total,n_total))

                        block_1 = np.diag(np.append(np.append(np.sum(a['ahfhat'][ :, : , ntask, 0, 0 ], axis = 1), \
                                    np.sum(a['ahfhat'][ :, : , ntask, 1, 1 ], axis = 0) + \
                                    np.sum(a['chfhat'][ :, : , ntask, 0, 0 ], axis = 1) + \
                                    np.sum(a['chfhat'][ :, : , ntask, 1, 1 ], axis = 0) + \
                                    np.sum(a['ghfhat'][ :, : , ntask, 1, 1 ], axis = 0)), \
                                    np.sum(a['ghfhat'][ :, : , ntask, 0, 0 ], axis = 1)))

                        
                        """
                        #DEBUG. #############################################
                        import pdb; pdb.set_trace();
                        vd = np.zeros((n_total,n_total))
                        vd[ np.eye(n_total).astype('bool') ] = np.append(np.append(np.sum(a['ahfhat'][ :, : , ntask, 0, 0 ], axis = 1), \
                                np.sum(a['ahfhat'][ :, : , ntask, 1, 1 ], axis = 0) + np.sum(a['chfhat'][ :, : , ntask, 0, 0 ], axis = 1) + \
                                np.sum(a['chfhat'][ :, : , ntask, 1, 1 ], axis = 0) + np.sum(a['ghfhat'][ :, : , ntask, 1, 1 ], axis = 0)._value), \
                                np.sum(a['ghfhat'][ :, : , ntask, 0, 0 ], axis = 1)._value)
                        vd[ n_obs : n_obs + n_pset, n_obs : n_obs + n_pset ] = vd[ n_obs : n_obs + n_pset, n_obs : n_obs + n_pset ] + \
                                a['chfhat'][ :, : , ntask, 0, 1 ] + a['chfhat'][ :, : , ntask, 1, 0 ].T
                        vd[ 0 : n_obs, n_obs : n_obs + n_pset ] = a['ahfhat'][ :, :, ntask, 0, 1]
                        vd[ n_obs : n_obs + n_pset, 0 : n_obs ] =  a['ahfhat'][ :, :, ntask, 0, 1].transpose()

                        vd[ n_obs + n_pset : n_total, n_obs : n_obs + n_pset ] = a['ghfhat'][ :, :, ntask, 0, 1]._value
                        vd[ n_obs : n_obs + n_pset, n_obs + n_pset : n_total ] =  a['ghfhat'][ :, :, ntask, 0, 1].transpose()._value
                        ######################################################
                        """
                        """
                        vTilde[ np.eye(n_total).astype('bool') ] = np.append(np.append(np.sum(a['ahfhat'][ :, : , ntask, 0, 0 ], axis = 1), \
                                np.sum(a['ahfhat'][ :, : , ntask, 1, 1 ], axis = 0) + np.sum(a['chfhat'][ :, : , ntask, 0, 0 ], axis = 1) + \
                                np.sum(a['chfhat'][ :, : , ntask, 1, 1 ], axis = 0) + np.sum(a['ghfhat'][ :, : , ntask, 1, 1 ], axis = 0)), \
                                np.sum(a['ghfhat'][ :, : , ntask, 0, 0 ], axis = 1))
                        """
                        block_2 = block_1[ n_obs : n_obs + n_pset, n_obs : n_obs + n_pset ] + a['chfhat'][ :, : , ntask, 0, 1 ] + a['chfhat'][ :, : , ntask, 1, 0 ].T

                        #vTilde[ n_obs : n_obs + n_pset, n_obs : n_obs + n_pset ] = vTilde[ n_obs : n_obs + n_pset, n_obs : n_obs + n_pset ] + \
                                #a['chfhat'][ :, : , ntask, 0, 1 ] + a['chfhat'][ :, : , ntask, 1, 0 ].T
                        block_3 = a['ahfhat'][ :, :, ntask, 0, 1]
                        block_4 = a['ahfhat'][ :, :, ntask, 0, 1].transpose()
                        block_5 = a['ghfhat'][ :, :, ntask, 0, 1]
                        block_6 = a['ghfhat'][ :, :, ntask, 0, 1].transpose()

                        #Building the matrix.
                        vTilde = np.array([])
                        for x_index in range(n_total):
                            for y_index in range(n_total):
                                #Block_2
                                if (x_index >= n_obs and x_index < n_obs + n_pset and y_index >= n_obs and y_index < n_obs + n_pset) \
                                or (x_index==y_index and x_index >= n_obs and x_index < n_obs + n_pset and y_index >= n_obs and y_index < n_obs + n_pset):
                                        vTilde = np.append(vTilde, block_2[x_index - n_obs, y_index - n_obs])
                                #Block_1
                                elif x_index == y_index:
                                        vTilde = np.append(vTilde, block_1[x_index, y_index])
                                #Block_3
                                elif x_index < n_obs and y_index >= n_obs and y_index < n_obs + n_pset:
                                        vTilde = np.append(vTilde, block_3[x_index, y_index - n_obs])
                                #Block_4
                                elif x_index >= n_obs and x_index < n_obs + n_pset and y_index < n_obs:
                                        vTilde = np.append(vTilde, block_4[x_index - n_obs, y_index])
                                #Block_5
                                elif x_index >=  n_obs + n_pset and y_index >= n_obs and y_index < n_obs + n_pset:
                                        vTilde = np.append(vTilde, block_5[x_index - n_obs - n_pset, y_index - n_obs])
                                #Block_6
                                elif x_index >= n_obs and x_index < n_obs + n_pset and y_index >= n_obs + n_pset:
                                        vTilde = np.append(vTilde, block_6[x_index - n_obs, y_index - n_obs - n_pset])
                                #Default 0
                                else:   
                                        vTilde = np.append(vTilde, 1e-15)

                        #TODO: Test that acq==autograd_acq. No da lo mismo. Hacer una version con cambios y otra sin, ir incorporandolos.
                        #TODO: Test that grad_acq==autograd_grad_acq
                        vTilde = vTilde.reshape((n_total, n_total))
                        """
                        vTilde[ 0 : n_obs, n_obs : n_obs + n_pset ] = a['ahfhat'][ :, :, ntask, 0, 1]
                        vTilde[ n_obs : n_obs + n_pset, 0 : n_obs ] =  a['ahfhat'][ :, :, ntask, 0, 1].transpose()

                        vTilde[ n_obs + n_pset : n_total, n_obs : n_obs + n_pset ] = a['ghfhat'][ :, :, ntask, 0, 1]
                        vTilde[ n_obs : n_obs + n_pset, n_obs + n_pset : n_total ] =  a['ghfhat'][ :, :, ntask, 0, 1].transpose()   
                        """

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

                """
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

                """
                for t in unconstrainedVariances:

                    # DHL replaced np.log(np.linalg.det()) to avoid precision errors

                    value = 0.5 * np.linalg.slogdet(unconstrainedVariances[t])[ 1 ] #- 0.5 * np.linalg.slogdet(constrainedVariances[t])[ 1 ]

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
    for obj in all_tasks:   
        m_test = np.append(m_test, np.tile(a['m'][ obj ][ n_obs + n_pset : n_total ], n_pset).reshape((n_pset, n_test)))
        #m_test[ n_task, :, : ] = np.tile(a['m'][ obj ][ n_obs + n_pset : n_total ], n_pset).reshape((n_pset, n_test))
        m_pset = np.append(m_pset, np.tile(a['m'][ obj ][ n_obs : n_obs + n_pset ], n_test).reshape((n_test, n_pset)).T)
        #m_pset[ n_task, :, : ] = np.tile(a['m'][ obj ][ n_obs : n_obs + n_pset ], n_test).reshape((n_test, n_pset)).T
        v_cov = np.append(v_cov, a['V'][ obj ][ n_obs : n_obs + n_pset, n_obs + n_pset : n_total ])
        #v_cov[ n_task, :, : ] = a['V'][ obj ][ n_obs : n_obs + n_pset, n_obs + n_pset : n_total ]
        v_test = np.append(v_test, np.tile(np.diag(a['V'][ obj ])[ n_obs + n_pset : n_total ], n_pset).reshape((n_pset, n_test)))
        #v_test[ n_task, :, : ] = np.tile(np.diag(a['V'][ obj ])[ n_obs + n_pset : n_total ], n_pset).reshape((n_pset, n_test))
        v_pset = np.append(v_pset, np.tile(np.diag(a['V'][ obj ])[ n_obs : n_obs + n_pset ], n_test).reshape((n_test, n_pset)).T)
        #v_pset[ n_task, :, : ] = np.tile(np.diag(a['V'][ obj ])[ n_obs : n_obs + n_pset ], n_test).reshape((n_test, n_pset)).T
        #n_task += 1
    
    m_test = m_test.reshape((q, n_pset, n_test))
    m_pset = m_pset.reshape((q, n_pset, n_test))
    v_cov = v_cov.reshape((q, n_pset, n_test))
    v_test = v_test.reshape((q, n_pset, n_test))
    v_pset = v_pset.reshape((q, n_pset, n_test))
    #n_task = 0

    for cons in all_constraints:
        c_m = np.append(c_m, np.tile(a['m_cons'][ cons ][ n_obs + n_pset : n_total ], n_pset))
        #c_m[ n_task, :, : ] = a['m_cons'][ cons ][ n_obs + n_pset : n_total ]
        c_v = np.append(c_v, np.tile(np.diag(a['V_cons'][ cons ])[ n_obs + n_pset : n_total ], n_pset))
        #c_v[ n_task, :, : ] = np.diag(a['V_cons'][ cons ])[ n_obs + n_pset : n_total ]
        #n_task += 1

    c_m = c_m.reshape((c, n_pset, n_test))
    c_v = c_v.reshape((c, n_pset, n_test))

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
        import pdb; pdb.set_trace();
        raise npla.linalg.LinAlgError("Negative variance in the sqrt!")

    if np.any(vOld_test < 0):
        import pdb; pdb.set_trace();
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

    #Hay que sustituir este bloque de codigo por un np.where.
    """
    if np.any(np.logical_not(logZ_orig == -np.inf)):
        sel = (logZ_orig == -np.inf)
        import pdb; pdb.set_trace();
        logZ_orig[ sel ] = logcdf_robust(-np.min(alpha[ :, sel ], axis = 0))
    """
    logZ_orig = np.where(logZ_orig == -np.inf, logcdf_robust(-np.min(alpha, axis = 0)), logZ_orig)

    logZ_term1 = np.sum(log_phi_cons, axis = 0) + logZ_orig
    logZ_term2 = log_1_minus_exp_x(np.sum(log_phi_cons, axis = 0))

    """
    if np.any(np.logical_not(logZ_term2 == -np.inf)):
        sel = (logZ_term2 == -np.inf)
        logZ_term2[ sel ] = logcdf_robust(-np.min(alpha_cons[ :, sel ], axis = 0))
    """

    logZ_term2 = np.where(logZ_term2 == -np.inf, logcdf_robust(-np.min(alpha_cons, axis = 0)), logZ_term2)

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

        vTilde_test_new = np.where(o_cond, 1e-15, vTilde_test_new)
        vTilde_pset_new = np.where(o_cond, 1e-15, vTilde_pset_new)
        vTilde_cov_new = np.where(o_cond, 1e-15, vTilde_cov_new)
        mTilde_test_new = np.where(o_cond, 1e-15, mTilde_test_new)
        mTilde_pset_new = np.where(o_cond, 1e-15, mTilde_pset_new)
        vTilde_cons = np.where(c_cond, 1e-15, vTilde_cons)
        mTilde_cons = np.where(c_cond, 1e-15, mTilde_cons)

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

    g_c_hfHatNew = vTilde_cons
    h_c_hfHatNew = mTilde_cons
   
    ghfhat = np.array([])
    hhfhat = np.array([])
    g00 =  vTilde_test_new.T * damping + (1 - damping) * a['ghfhat'][ :, :, :, 0, 0 ]
    g01 = vTilde_cov_new.T * damping + (1 - damping) * a['ghfhat'][ :, :, :, 0, 1 ]
    g10 = vTilde_cov_new.T * damping + (1 - damping) * a['ghfhat'][ :, :, :, 1, 0 ]
    g11 = vTilde_pset_new.T * damping + (1 - damping) * a['ghfhat'][ :, :, :, 1, 1 ]
    h0 = mTilde_test_new.T * damping + (1 - damping) * a['hhfhat'][ :, :, :, 0 ]
    h1 = mTilde_pset_new.T * damping + (1 - damping) * a['hhfhat'][ :, :, :, 1 ]
    for tp in range(n_test):
        for pp in range(n_pset):
            for qp in range(q):
                ghfhat = np.append(ghfhat, np.array([[g00[tp, pp, qp], g01[tp, pp, qp]], [g10[tp, pp, qp], g11[tp, pp, qp]]]))
                hhfhat = np.append(hhfhat, np.array([h0[tp, pp, qp], h1[tp, pp, qp]]))
    ghfhat = ghfhat.reshape((n_test, n_pset, q, 2, 2))
    hhfhat = hhfhat.reshape((n_test, n_pset, q, 2))
    a['ghfhat'] = ghfhat
    a['hhfhat'] = hhfhat
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

def compute_acq_fun_wrt_test_points(X_test, obj_model_dict, con_models_dict, pareto_set, info_gps, tasks, log, fun_log):

            #Execute necessary functionality for the test factors update and the recomputation of the marginals and PPESMOC.
            acq, unconstrainedVariances, constrainedVariances = compute_unconstrained_variances_and_init_acq_fun \
                                                                                (obj_model_dict, X_test, con_models_dict, log, fun_log)
            #fun_log('compute_unconstrained_variances_and_init_acq_fun', log, \
                    #{'obj_model_dict': obj_model_dict, 'cand' : X_test, 'con_models_dict' : con_models_dict, \
                    #'acq': acq, 'unconstrainedVariances': unconstrainedVariances, 'constrainedVariances': constrainedVariances})
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
                """
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
                """
                """
                fun_log('update_full_marginals', log, \
                        {'Vinv_cons' : info_gps_ps['Vinv_cons'], \
                         'V_cons' : info_gps_ps['V_cons'], \
                         'm_nat_cons' : info_gps_ps['m_nat_cons'], \
                         'm_cons' : info_gps_ps['m_cons'], \
                         'Vinv' : info_gps_ps['Vinv'], \
                         'V' : info_gps_ps['V'], \
                         'm_nat' : info_gps_ps['m_nat'], \
                         'm' : info_gps_ps['m']
                        })
                """
                """

                info_gps_ps = update_full_Factors_only_test_factors(info_gps_ps, 0.1, \
                                minimize = True, no_negative_variances_nor_nands = True)
                """
                """
                fun_log('update_full_Factors_only_test_factors', log, \
                    {'a[ghfhat][ :, :, :, 0, 0 ]': info_gps_ps['ghfhat'][ :, :, :, 0, 0 ], \
                     'a[ghfhat][ :, :, :, 1, 1 ]': info_gps_ps['ghfhat'][ :, :, :, 1, 1 ], \
                     'a[ghfhat][ :, :, :, 0, 1 ]': info_gps_ps['ghfhat'][ :, :, :, 0, 1 ], \
                     'a[ghfhat][ :, :, :, 1, 0 ]': info_gps_ps['ghfhat'][ :, :, :, 1, 0 ], \
                     'a[hhfhat][ :, :, :, 0 ]': info_gps_ps['hhfhat'][ :, :, :, 0 ], \
                     'a[hhfhat][ :, :, :, 1 ]': info_gps_ps['hhfhat'][ :, :, :, 1 ], \
                     'a[g_c_hfhat][ :, :, : ]': info_gps_ps['g_c_hfhat'][ :, :, : ], \
                     'a[h_c_hfhat][ :, :, : ]': info_gps_ps['h_c_hfhat'][ :, :, : ], \
                     'a[m]': info_gps_ps['m'], 'a[V]': info_gps_ps['V'], 'a[m_cons]': info_gps_ps['m_cons'], 'a[V_cons]': info_gps_ps['V_cons']})
                """
                """
                #import pdb; pdb.set_trace();
                info_gps_ps = update_full_marginals(info_gps_ps)
                """
                """
                fun_log('update_full_marginals', log, \
                        {'Vinv_cons' : info_gps_ps['Vinv_cons'], \
                         'V_cons' : info_gps_ps['V_cons'], \
                         'm_nat_cons' : info_gps_ps['m_nat_cons'], \
                         'm_cons' : info_gps_ps['m_cons'], \
                         'Vinv' : info_gps_ps['Vinv'], \
                         'V' : info_gps_ps['V'], \
                         'm_nat' : info_gps_ps['m_nat'], \
                         'm' : info_gps_ps['m']
                        })
                """
                """
                predictionEP = get_test_predictive_distributions(info_gps_ps)[0]
                """
                """
                fun_log('get_test_predictive_distributions', log, \
                        {'mfs' : predictionEP['mf'], \
                         'vfs' : predictionEP['vf'], \
                         'mcs' : predictionEP['mc'], \
                         'vcs' : predictionEP['vc'], \
                         'unconstrainedVariances' : unconstrainedVariances})
                """
                """
                """
                #Hay que dar la suma de las adquisiciones tambien.
                acqs[ps] = compute_PPESMOC_approximation(None, obj_model_dict, con_models_dict, \
                                                unconstrainedVariances, constrainedVariances, acq)
                fun_log('compute_PPESMOC_approximation Autograd', log, {'acq' : acqs[ps]})
                
    
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
            fun_log('BB acqs', log, {'acq' : final_acqs_dict})
            total_acq = 0.0
            for task in tasks:
                total_acq += final_acqs_dict[ task ]
            #Sumar las adquisiciones de cada BB.
            fun_log('total_acq Autograd', log, {'acq' : total_acq})
            return total_acq
