'''
    Probability distributions. 
    Convention: suffix all probability distributions with "pd"
'''
import numpy as np
from scipy.special import gammaln

def schultzpd(R,pars):
    '''The Schultz probability distribution.'''
    #check which kind of pars it is
    if(hasattr(pars['z'],"value")):
        z = pars['z'].value
        rbar = pars['rbar'].value
    else:
        z = pars['z']
        rbar = pars['rbar']
    res = (z+1)*np.log((z+1)/rbar)
    res += np.log(R)*z
    res -= (z+1)/rbar*R
    res -= gammaln(z+1)

    return np.exp(res)

def negbinomialdist(k,pars):
    '''The probability of intensity of cts k
        with parameters:
        pars['kbar'] - the avg intensity 
        pars['M'] - the coherence factor (beta = 1/M)
    '''
    if(hasattr(pars['M'],"value")):
        M = pars['M'].value
        kbar = pars['kbar'].value
    else:
        M = pars['M']
        kbar = pars['kbar']

    res = gammaln(k + M) - gammaln(M) - gammaln(k+1)
    res -= k*np.log(1 + M/kbar)
    res -= M*np.log(1 + kbar/M)
    return np.exp(res)

def poissondist(k,pars):
    '''' Prob of Poisson.
        pars['kbar'] - avg intensity
    '''
    kbar = pars['kbar']
    if(hasattr(kbar,"value")):
        kbar = kbar.value
    return np.exp(k*np.log(kbar) - kbar - gammaln(k+1))
