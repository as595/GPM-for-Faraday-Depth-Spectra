import numpy as np

def calc_BIC(L,K,N):
    
    bic = -2.*L + K*np.log(N)
    
    return bic

def calc_AIC(L,K):
    
    aic = 2.*K - 2.*L
    
    return aic

def calc_AICc(L,K,N):
    
    aic = calc_AIC(L,K)
    aicc= aic - 2.*K*(K+1)/(N - K - 1)
    
    return aicc
