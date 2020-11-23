from __future__ import division
import scipy as sp
import numpy as np
from scipy.special import beta, gamma, betaln, gammaln, betainc, gammainc, erf
import matplotlib.pyplot as plt
from scipy.special import betaln, gammaln, beta, gamma, betainc, gammainc 
from scipy import optimize as opt 

class Gb(object):
    def __init__(self, a, b, c, p, q):
        self.a = a
        self.b = b
        self.c = c
        self.p = p
        self.q = q

    def pdf(self, y):
        a = self.a 
        b = self.b 
        c = self.c
        p = self.p 
        q = self.q

        bta = np.exp(gammaln(p)+gammaln(q) - gammaln(p+q))
        pdf = abs(a)*y**(a*p-1)*(1-(1-c)*(y/b)**a)**(q-1) / (b**(a*p)*bta*(1+c*(y/b)**a)**(p+q))
        return pdf  

    def mom(self, h):
        a = self.a 
        b = self.b 
        c = self.c
        p = self.p 
        q = self.q

        bta = np.exp(gammaln(p) + gammaln(q) - gammaln(p+q))
        pass
    
class Gb1(Gb):
    def __init__(self, a, b, p, q):
        self.a = a
        self.b = b
        self.c = 0
        self.p = p
        self.q = q

    def cdf(self, y):
        a = self.a
        b = self.b
        c = self.c
        p = self.p
        q = self.q

        z = (y/b)**a
        cdf = betainc(p, q, z)
        return cdf

    def mom(self, h):
        a = self.a
        b = self.b
        c = self.c
        p = self.p
        q = self.q

        mom = np.exp(h*np.log(b) + gammaln(p+q) + gammaln(p + h/a) - gammaln(p + q + h/a) - gammaln(p))
        return mom

    def mean(self):
        mean = self.mom(1)
        return mean

    def std(self):
        var = self.mom(2) - (self.mom(1))**2
        std = var**(1/2)
        return std   
        
    def skew(self):
        mom1 = self.mom(1)
        mom2 = self.mom(2)
        mom3 = self.mom(3)
        var = (self.std())**2

        skew = (mom3 - 3*mom1*mom2 + 2*mom1**3)/(var)**(3/2)        
        return skew

    def kurt(self):
        mom1 = self.mom(1)
        mom2 = self.mom(2)
        mom3 = self.mom(3)
        mom4 = self.mom(4)
        var = (self.std())**2

        kurtosis = (mom4 - 4*mom1*mom3 + 6*mom1**2*mom2 - 3*mom1**4) / var**2         
        return kurtosis

    def loglike(self, data, paravec_log, sign = 1):

        a, b, p, q = np.exp(paravec_log)
        n = len(data)

        lnb = gammaln(p) + gammaln(q) - gammaln(p+q)

        loglike = n*np.log(abs(a)) + (a*p-1)*np.sum(np.log(data)) + \
        (q-1)*np.sum(np.log(1-(data/b)**a)) - n*a*p*np.log(b) - n*lnb

        loglike = sign*loglike
        return loglike

class B1(Gb1):
    def __init__(self, b, p, q):
        self.a = 1
        self.c = 0
        self.b = b
        self.p = p
        self.q = q

    def loglike(self, data, paravec_log, sign = 1):

        a = 1
        b, p, q = np.exp(paravec_log)
        n = len(data)
        
        lnb = gammaln(p) + gammaln(q) - gammaln(p+q)

        loglike = n*np.log(abs(a)) + (a*p-1)*np.sum(np.log(data)) + \
        (q-1)*np.sum(np.log(1-(data/b)**a)) - n*a*p*np.log(b) - n*lnb

        loglike = sign*loglike
        return loglike

class Pareto(Gb1):
    def __init__(self, b, p):
        self.a = -1
        self.q = 1
        self.c = 0
        self.b = b
        self.p = p

    def cdf(self, y):
        b = self.b
        p = self.p
        if len(y) > 1:
            cdf = np.zeros(len(y))
            ind1 = y >= b 
            cdf[ind1] = 1 - (b/y[ind1])** p
        elif len(y) == 1:
            if y >= b:
                cdf = 1 - (b/y)**p
            elif y<b:
                cdf = 0
        return cdf

    def loglike(self, data, paravec, sign = 1):
        #we are not taking the log of the paramters, because in this case, a = -1, so 
        #it doesn't have to be constrained to be positive. 
        n = len(data)
        a = -1
        q = 1
        b, p = paravec
        loglike = n*sp.log(np.exp(a)) + (np.exp(a)*np.exp(p)-1)*np.sum(data) + \
        (np.exp(q)-1)*np.sum(1-(data/np.exp(b))**np.exp(a)) - n*np.exp(a)*np.exp(p)*np.log(b)\
        - n*betaln(np.exp(p), np.exp(q))
        loglike = sign*loglike
        return loglike




class B(Gb):
    def __init__(self, b, c, p, q):
        self.a = 1
        self.b = b
        self.c = c
        self.p = p
        self.q = q
    def mom(self, h):
        a = self.a
        b = self.b
        c = self.c
        p = self.p
        q = self.q

        pass
        #Does the beta function have moments? 
    def cdf(self, h):
        pass
        #does the beta have a cdf? 

class Gb2(object):
    def __init__(self, a, b, p, q):
        self.a = a
        self.b = b
        self.p = p
        self.q = q

    def pdf(self, y):
        a = self.a
        b = self.b
        p = self.p
        q = self.q
        bta = np.exp(gammaln(p) + gammaln(q) - gammaln(p+q))
        pdf = np.exp(np.log(abs(a)) + (a*p - 1)*np.log(y)- (a*p)*np.log(b)\
             - np.log(bta) - (p+q)*np.log((1 + (y/b)**a)))
        return pdf

    def cdf(self, y):
        a = self.a
        b = self.b
        p = self.p
        q = self.q

        z = np.exp(a * np.log((y/b)) - a*np.log(1 + (y/b)))
        cdf = betainc(p, q, z)
        return cdf

    def mom(self, h):
        """
        Gives us the moments about zero. We will used these in our calculations of the moments 
        around the mean

        h is the moment we would like
        """
        a = self.a
        b = self.b 
        p = self.p 
        q = self.q
        mom = np.exp(h*np.log(b) + gammaln(p + h/a) + gammaln(q - h/a) - gammaln(p) - gammaln(q))
        return mom

    def mean(self):
        a = self.a
        b = self.b 
        p = self.p 
        q = self.q
        mean = self.mom(1)
        return mean

    def std(self):
        a = self.a
        b = self.b 
        p = self.p 
        q = self.q 
        mom2 = self.mom(2)
        ex = self.mom(1)
        var = mom2 - ex**2
        std = var**(1/2)
        return std

    def skew(self):
        mom1 = self.mom(1)
        mom2 = self.mom(2)
        mom3 = self.mom(3)
        var = (self.std())**2

        skew = (mom3 - 3*mom1*mom2 + 2*mom1**3)/(var)**(3/2)        
        return skew
        
    def kurt(self):   
        mom1 = self.mom(1)
        mom2 = self.mom(2)
        mom3 = self.mom(3)
        mom4 = self.mom(4)
        var = (self.std())**2

        kurtosis = (mom4 - 4*mom1*mom3 + 6*mom1**2*mom2 - 3*mom1**4) / var**2         
        return kurtosis

    def loglike(self, data, paravec, sign = 1):
        """
        The sign option allows for the function to return the negative of the log-likelihood
        so that it can be estimated using minization libraries of python. 

        The parameters must come in as logs. 
        """
        la, lb, lp, lq = paravec        
        loglike = len(data) * sp.log(np.exp(la)) + (np.exp(la)*np.exp(lp)-1) * sum(sp.log(data)) \
                 - len(data)*np.exp(la)*np.exp(lp)*sp.log(np.exp(lb)) - len(data) * betaln(np.exp(lp), np.exp(lq)) -\
                (np.exp(lp)+np.exp(lq)) * sum(sp.log(1+(data/np.exp(lb))**np.exp(la)))
        loglike = sign*loglike
        return loglike

    
class B2(Gb2):
    def __init__(self, b, p, q):
        self.a = 1          
        self.b = b
        self.p = p
        self.q = q

    def loglike(self, data, paravec, sign = 1):
        la = 0
        lb, lp, lq = paravec        
        loglike = len(data) * sp.log(np.exp(la)) + (np.exp(la)*np.exp(lp)-1) * sum(sp.log(data)) \
                 - len(data)*np.exp(la)*np.exp(lp)*sp.log(np.exp(lb)) - len(data) * betaln(np.exp(lp), np.exp(lq)) -\
                (np.exp(lp)+np.exp(lq)) * sum(sp.log(1+(data/np.exp(lb))**np.exp(la)))
        loglike = sign*loglike
        return loglike

 
class Br12(Gb2):
    def __init__(self, a, b, q):
        self.p = 1
        self.a = a
        self.b = b
        self.q = q
        
    def loglike(self, data, paravec, sign = 1):
        lp = 0
        la, lb, lq = paravec        
        loglike = len(data) * sp.log(np.exp(la)) + (np.exp(la)*np.exp(lp)-1) * sum(sp.log(data)) \
                 - len(data)*np.exp(la)*np.exp(lp)*sp.log(np.exp(lb)) - len(data) * betaln(np.exp(lp), np.exp(lq)) -\
                (np.exp(lp)+np.exp(lq)) * sum(sp.log(1+(data/np.exp(lb))**np.exp(la)))
        loglike = sign*loglike
        return loglike

class Br3(Gb2):
    def __init__(self, a, b, p):
        self.q = 1
        self.a = a
        self.b = b
        self.p = p

    def loglike(self, data, paravec, sign = 1):
        lq = 0
        la, lb, lp = paravec
        loglike = len(data) * sp.log(np.exp(la)) + (np.exp(la)*np.exp(lp)-1) * sum(sp.log(data)) \
                 - len(data)*np.exp(la)*np.exp(lp)*sp.log(np.exp(lb)) - len(data) * betaln(np.exp(lp), np.exp(lq)) -\
                (np.exp(lp)+np.exp(lq)) * sum(sp.log(1+(data/np.exp(lb))**np.exp(la)))
        loglike = sign*loglike   
        return loglike

class L(Gb2):
    def __init__(self, b, q):
        self.a = 1
        self.p = 1
        self.b = b
        self.q = q

    def loglike(self, data, paravec, sign = 1):
        la, lp = 0, 0
        lb, lq = paravec
        loglike = len(data) * sp.log(np.exp(la)) + (np.exp(la)*np.exp(lp)-1) * sum(sp.log(data)) \
                 - len(data)*np.exp(la)*np.exp(lp)*sp.log(np.exp(lb)) - len(data) * betaln(np.exp(lp), np.exp(lq)) -\
                (np.exp(lp)+np.exp(lq)) * sum(sp.log(1+(data/np.exp(lb))**np.exp(la)))
        loglike = sign*loglike
        return loglike

class InvL(Gb2):
    def __init__(self, b, q):
        self.a = -1
        self.p = 1
        self.b = b
        self.q = q
    def loglike(self, data, paravec, sign = 1):
        a, p = - 1, 1
        b, q = paravec
        loglike = len(data) * sp.log(np.exp(a)) + (np.exp(a)*np.exp(p)-1) * sum(sp.log(data)) \
                 - len(data)*np.exp(a)*np.exp(p)*sp.log(np.exp(b)) - len(data) * betaln(np.exp(p), np.exp(q)) -\
                (np.exp(p)+np.exp(q)) * sum(sp.log(1+(data/np.exp(b))**np.exp(a)))
        loglike = sign*loglike
        return loglike

class Fisk(Gb2):
    def __init__(self, a, b):
        self.p = 1
        self.q = 1
        self.a = a
        self.b = b

    def loglike(self, data, paravec, sign = 1):
        lp, lq = 0, 0
        la, lb = paravec
        loglike = len(data) * sp.log(np.exp(la)) + (np.exp(la)*np.exp(lp)-1) * sum(sp.log(data)) \
                 - len(data)*np.exp(la)*np.exp(lp)*sp.log(np.exp(lb)) - len(data) * betaln(np.exp(lp), np.exp(lq)) -\
                (np.exp(lp)+np.exp(lq)) * sum(sp.log(1+(data/np.exp(lb))**np.exp(la)))
        loglike = sign*loglike
        return loglike

class LogLog(Gb2):
    def __init__(self, b):
        self.a = 1
        self.p = 1
        self.q = 1
        self.b = b

    def loglike(self, data, paravec, sign = 1):
        la, lp, lq = 0, 0, 0
        lb = paravec
        loglike = len(data) * sp.log(np.exp(la)) + (np.exp(la)*np.exp(lp)-1) * sum(sp.log(data)) \
                 - len(data)*np.exp(la)*np.exp(lp)*sp.log(np.exp(lb)) - len(data) * betaln(np.exp(lp), np.exp(lq)) -\
                (np.exp(lp)+np.exp(lq)) * sum(sp.log(1+(data/np.exp(lb))**np.exp(la)))
        loglike = sign*loglike
        return loglike

class Gg(object):
    def __init__(self, a, b, p):
        self.a = a
        self.b = b
        self.p = p
    def pdf(self, y):
        a = self.a
        b = self.b 
        p = self.p
        pdf = abs(a)*y**(a*p-1)*np.exp(-(y/b)*a)/ (b**(a*p)*gamma(p))
        return pdf
    def cdf(self, y):
        a = self.a
        b = self.b
        p = self.p
        z = (y/b)**a
        cdf = gammainc(p, z)
        return cdf 
    def mean(self):
        a = self.a
        b = self.b
        p = self.p
        mean = b* gamma(p + 1/a)/gamma(p)
        return mean
    def std(self):
        a = self.a
        b = self.b
        p = self.p
        mom2 = b**2*gamma(p + 2/a)/gamma(p)
        var = mom2 - (self.mean())**2
        std = var**(1/2)
        return std
    def skew(self):
        a = self.a
        b = self.b
        p = self.p
        g0 = gamma(p)
        g1 = gamma(p + 1/a)
        g2 = gamma(p + 2/a)
        g3 = gamma(p + 3/a)

        skew = (g0**2*g3 - 3*g0*g1*g2 + 2*g1**3)/(g0*g2 - g1**2)**(3/2)
        return skew

    def kurt(self):
        a = self.a
        b = self.b
        p = self.p
        g0 = gamma(p)
        g1 = gamma(p + 1/a)
        g2 = gamma(p + 2/a)
        g3 = gamma(p + 3/a)
        g4 = gamma(p + 4/a)

        kurt = (g0**3*g4 - 4*g0**2*g1*g3 + 6*g0*g1**2*g2 - 3*g1**4)/(g0*g2 - g1**2)**2
        return kurt

    def loglike(self, data, paravec, sign = 1):

        la, lb, lp = paravec
        loglike = len(data) * sp.log(np.exp(la)) + (np.exp(la)*np.exp(lp)-1) * sum(sp.log(data)) - (1/np.exp(lb)**np.exp(la)) * sum(data**np.exp(la))\
        -len(data)*np.exp(la)*np.exp(lp)*sp.log(np.exp(lb)) - len(data) * gammaln(np.exp(lp))
        loglike = sign*loglike
        return loglike

    #I'm writing the below function so we don't have to copy and paste the loglike so much    
    def loglike_param(self, data, la, lb, lp, sign = 1):        
        ll =len(data) * sp.log(np.exp(la)) + (np.exp(la)*np.exp(lp)-1) * sum(sp.log(data)) - (1/np.exp(lb)**np.exp(la)) * sum(data**np.exp(la))\
        -len(data)*np.exp(la)*np.exp(lp)*sp.log(np.exp(lb)) - len(data) * gammaln(np.exp(lp))
        ll = sign*ll       
        return ll

class Ga(Gg):
    def __init__(self, b, p):
        self.a = 1
        self.b = b
        self.p = p

    def loglike(self, data, paravec, sign = 1):
        la = 0        
        lb, lp = paravec 
        ll = self.loglike_param(data, la, lb, lp, sign)
        return ll

class W(Gg):
    def __init__(self, a, b):
        self.p = 1
        self.a = a
        self.b = b
        # super(W, self).__init__(a, b, self.p)

    def loglike(self, data, paravec, sign = 1):
        lp = 0
        la, lb = paravec
        ll = self.loglike_param(data, la, lb, lp, sign)
        return ll

class Chi2(Gg):
    def __init__(self, p):
        self.a = 1
        self.b = 2
        self.p = p
        # super(Chi2, self).__init__(self.a, self.b, p)
    def loglike(self, data, paravec, sign = 1):
        la, lb = 0, np.log(2)
        lp = paravec
        ll = self.loglike_param(data, la, lb, lp, sign)
        return ll

class Exp(Gg):
    def __init__(self, b):
        self.a = 1
        self.p = 1
        # super(Exp, self).__init__(self.a, b, self.p)
        self.b = b
    def loglike(self, data, paravec, sign = 1):
        la, lp = 0, 0
        lb = paravec
        ll = self.loglike_param(data, la, lb, lp, sign)
        return ll 



class Ln(object):
    def __init__(self, u, sig):
        self.u = u
        self.sig = sig

    def pdf(self, y):
        u = self.u
        sig = self.sig
        pdf = 1/(y*sig*(2*np.pi)**(1/2)) * np.exp(-(np.log(y) - u)**2 / (2*sig**2))
        return pdf
    def cdf(self, y):
        u = self.u
        sig = self.sig
        z = (np.log(y) - u)/(sig*2**(1/2))
        cdf = (1/2)*(1 + erf(z))
        return cdf

    def mom(self, h):
        u = self.u
        sig = self.sig
        mom = np.exp(h*u + h**2*sig**2/2)
        return mom

    def mean(self):
        mean = self.mom(1)
        return mean

    def std(self):
        var = self.mom(2) - self.mom(1)**2
        std = var**(1/2)
        return std

    def skew(self):

        mom1 = self.mom(1)
        mom2 = self.mom(2)
        mom3 = self.mom(3)
        var = (self.std())**2

        skew = (mom3 - 3*mom1*mom2 + 2*mom1**3)/(var)**(3/2)
        return skew

    def kurt(self):          

        mom1 = self.mom(1)
        mom2 = self.mom(2)
        mom3 = self.mom(3)
        mom4 = self.mom(4)
        var = (self.std())**2

        kurtosis = (mom4 - 4*mom1*mom3 + 6*mom1**2*mom2 - 3*mom1**4) / var**2         
        return kurtosis 


    def loglike(self, data, paravec, sign = 1): 

        lu, lsig = paravec
        loglike = (-1/(2*np.exp(lsig)**2)) * sum((sp.log(data)-np.exp(lu))**2) - (len(data)/2) * sp.log(2*sp.pi) \
        - len(data) * sp.log(np.exp(lsig)) - sum(sp.log(data))
        loglike = sign*loglike
        return loglike






