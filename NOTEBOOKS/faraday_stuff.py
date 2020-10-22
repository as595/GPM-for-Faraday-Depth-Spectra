import numpy as np

def calc_k(w):
    
    k = np.sum(w)
    
    return k

def calc_l0(w,l2):
    
    k = calc_k(w)
    l0 = (1./k)*np.sum(w*l2)
    
    return l0

def calc_f(phi,l2,q,u,w):
    
    p = q+1j*u
    k = calc_k(w)
    l0 = calc_l0(w,l2)
    f = (1./k)*np.sum(w*p*np.exp(-2*1j*(l2-l0)*phi))
    
    return f

def calc_r(phi,l2,w):
    
    k = calc_k(w)
    l0 = calc_l0(w,l2)
    r = (1./k)*np.sum(w*np.exp(-2*1j*(l2-l0)*phi))
    
    return r


