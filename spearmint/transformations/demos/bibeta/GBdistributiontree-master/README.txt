This library holds most of the distributions found in the GB family (refer to Gbdistributiontree). The most flexible distribution included, the GB, takes 5 parameters
a, b, c, p, q. All the rest of these distributions are special cases of the GB, and take on 4 or less parameters. 

Included distributions with their required parameters (These can be found in the gb2library.py):

*Gb(a, b, c, p, q)      Generalized Beta
Gb1(a, b, p, q)         Generalized Beta of the 1st kind
Gb2(a, b, p, q)         Generalized Beta of the 2nd kind
B1(b, p, q)             Beta of the 1st kind
GG(a, b, p)             Generalized Gamma
B2(b, p, q)             Beta of the 2nd kind
Br12(a, b, q)           Burr 12 (Singh-Maddala)
Br3(a, b, p)            Burr 3  (Dagum)
**Pareto(b, p)          Pareto
Ln(u, sig)              Lognormal
GA(b, p)                Gamma
F(b, q)                 F-distribution
W(a,b)                  Weibull distribution
**L(b, q)               Lomax
InvL(b, q)              Inverse Lomax
Fisk(a, b)              Fisk    
Chi2(p***)              Chi square
Exp(b)                  Exponential distribution
Loglog(b)               Loglogistic distribution

***p = degrees of freedomm
*Gb distribution only includes pdf method.
** Pareto and Lomax distribution's loglikelihood methods do not take the natural log of the parameters, but the parameters themselves (see loglike method below)

Methods included:
pdf(y): the distribution's pdf evaluated at the point, or array of points, given
cdf(y): the distribution's cdf evaluated at the point, or array of points, given
mom(h): The hth moment of the distribution. 
mean(): 
std():
skew():
kurt():
loglike(data, logparavec, sign=1): The loglikelikelihood function evaluated for a set of data.
                                    Use this function for MLE of the distribution parameters.
                                    Except for the Pareto and the Lomax, all distributions take the natural log of the parameters so optimization 
                                    methods are constrained to estimate postive a, b, p, q parameters. Since Python's optimization methods are usually minimization methods, sign=-1 switches the maximization problem to the minimization problem. 



