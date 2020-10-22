import autograd.numpy as np
import pylab as pl
import os,sys
import celerite
from celerite import terms
import scipy.optimize as op
import emcee
import corner

from information_criteria import *

class CustomTerm(terms.Term):
    parameter_names = ("log_a", "log_c", "log_P")

    def get_real_coefficients(self, params):
        log_a, log_c, log_P = params

        return (
            np.exp(log_a) * 0.5, np.exp(log_c),
        )

    def get_complex_coefficients(self, params):
        log_a, log_c, log_P = params

        return (
            np.exp(log_a) / (2.0), 0.0,
            np.exp(log_c), 2*np.pi*np.exp(-log_P),
        )

def read_data(input_dir, filename1):

    infile = input_dir + "/" + filename1
    nu,lam_squared,stokesQ,stokesU = np.loadtxt(infile,delimiter=' ',usecols=(0,1,2,3),unpack=True)

    return nu, lam_squared, stokesQ, stokesU

def shiftU(l2,Q,U,shift):
    
    dt = l2[1]-l2[0]
    nstep = int(shift/dt)
    
    if nstep>0:
        newU = U[nstep:]
        newQ = Q[:-nstep]
        newl2 = l2[:-nstep]
    elif nstep<0:
        nstep = np.abs(nstep)
        newU = U[:-nstep]
        newQ = Q[nstep:]
        newl2 = l2[nstep:]
    else:
        newU = U
        newQ = Q
        newl2 = l2
    
    return newl2,newQ,newU

def get_sign(l2,Q,U,shift):
    
    newl2,newQ,newU = shiftU(l2,Q,U,-shift)
    cc1 = np.dot(newQ,newU)
    newl2,newQ,newU = shiftU(l2,Q,U,shift)
    cc2 = np.dot(newQ,newU)
    
    if cc1>=cc2:
        sgn = -1.
    else:
        sgn = 1.
    
    return sgn

# set the logprior
def lnprior(p):

    lna = p[0]
    lnc = p[1]
    lnP = p[2]
    lns = p[3]
    #lns = 0.

    if (-25.<lna<5.) and (-25.<lnc<5.) and (-5.<lnP<10.) and (-25.<lns<5.):
        return 0.0

    return -np.inf


class MyCelerite():
    def __init__(self, noise):
    
        log_a = 0.1; log_c = 0.1; log_P = 1.0

        k1 = CustomTerm(log_a, log_c, log_P)
        k2 = terms.JitterTerm(log_sigma = np.log(noise))
        kernel = k1+k2
        self.gp = celerite.GP(kernel, mean=0.0, fit_mean=False)
        
        # set noise prior:
        
        
    # set the logposterior:
    def lnprob(self, p, x, y1, y2):

        lp = lnprior(p)

        return lp + self.lnlike(p, x, y1, y2) if np.isfinite(lp) else -np.inf


    # set the loglikelihood:
    def lnlike(self, p, x, y1, y2):

        ln_a = p[0]
        ln_c = p[1]
        ln_p = p[2]
        ln_s = p[3]

        p0 = np.array([ln_a,ln_c,ln_p,ln_s])
        #p0 = np.array([ln_a,ln_c,ln_p])


        # update kernel parameters:
        self.gp.set_parameter_vector(p0)

        # calculate the likelihood:
        ll1 = self.gp.log_likelihood(y1)
        ll2 = self.gp.log_likelihood(y2)
        ll = ll1 + ll2

        return ll if np.isfinite(ll) else 1e25
        
    
    def celerite_optimize(self, l2, stokesQ, stokesU):
        
        self.gp.compute(l2[::-1])
        
        data = (l2[::-1],stokesQ[::-1],stokesU[::-1])
        
        p = self.gp.get_parameter_vector()
        
        initial = np.array(np.zeros(len(p)))
        
        # initial log(likelihood):
        init_logL = self.gp.log_likelihood(stokesQ[::-1]) + self.gp.log_likelihood(stokesU[::-1])
        
        # set the dimension of the prior volume
        self.ndim = len(initial)
        
        nwalkers = 32
        
        p0 = [np.array(initial) + 1e-5 * np.random.randn(self.ndim) for i in range(nwalkers)]
        
        # initalise the sampler:
        self.sampler = emcee.EnsembleSampler(nwalkers, self.ndim, self.lnprob, args=data)
        
        # run a few samples as a burn-in:
        p0, lnp, _ = self.sampler.run_mcmc(p0, 500)
        self.sampler.reset()
        
        # take the highest likelihood point from the burn-in as a
        # starting point and now begin the production run:
        p = p0[np.argmax(lnp)]
        p0 = [p + 1e-5 * np.random.randn(self.ndim) for i in range(nwalkers)]
        p0, lnp, _ = self.sampler.run_mcmc(p0, 5000)
        
        # Find the maximum likelihood values:
        self.ml = p0[np.argmax(lnp)]
        
        return


    def celerite_predict(self, stokesQ, stokesU, t1):

        self.t1 = t1
        
        p = np.array(self.ml)
        self.gp.set_parameter_vector(p) 

        mu_q, cov_q = self.gp.predict(stokesQ[::-1], self.t1) 
        if np.all(np.diag(cov_q)>0.):
            std_q = np.sqrt(np.diag(cov_q))
        else:
            std_q = None

        mu_u, cov_u = self.gp.predict(stokesU[::-1], self.t1) 
        if np.all(np.diag(cov_u)>0.):
            std_u = np.sqrt(np.diag(cov_u))
        else:
            std_u = None
            
        self.mu_q = mu_q
        self.mu_u = mu_u
        self.std_q = std_q
        self.std_u = std_u
        
        return self.mu_q, self.std_q, self.mu_u, self.std_u


    def celerite_rmpred(self):
    
        samples = self.sampler.chain[:, 100:, :].reshape((-1, self.ndim))
        p1 = np.percentile(samples[:,2], 16)  # one sigma
        p2 = np.percentile(samples[:,2], 50)  # expectation
        p3 = np.percentile(samples[:,2], 84)  # one sigma

        upper = p3-p2
        lower = p2-p1
        
        abs_rm = np.pi/np.exp(p2)
        hi = np.abs(np.pi/np.exp(p3) - abs_rm)
        lo = np.abs(np.pi/np.exp(p1) - abs_rm)
        
        T_by_4 = np.pi/(4*abs_rm)
        rm_sgn = get_sign(self.t1,self.mu_q,self.mu_u,T_by_4)
        
        return rm_sgn*np.pi/np.exp(self.ml[2]), rm_sgn*np.pi/np.exp(p2), lo, hi
