import numpy as np
import h5py
from datetime import datetime

def getPosteriorSamples(filename):
    # get the data
    # the posterior distribution is in samples
    filename = 'data/edelweiss_corr_C_systematicErrors_sampler_nll_allpars_gausPrior.h5'
    f = h5py.File(filename,'r')

    # need to store data in an array:
    # The sampler will now have a chains attribute 
    # which is an array with shape (Nwalker,N,Ndim) 
    # where N is the number of interations (500 in our inital run)
    # and Ndim is the number of fit parameters
    path='{}/{}/'.format('mcmc','sampler')

    aH = np.asarray(f[path+'aH'])
    C = np.asarray(f[path+'C'])
    m = np.asarray(f[path+'m'])
    scale = np.asarray(f[path+'scale'])
    A = np.asarray(f[path+'A'])
    B = np.asarray(f[path+'B'])
    samples = np.asarray(f[path+'samples'])
    f.close()

    #print ("sampler dimensions are: ", np.shape(samples))
    nwalkers, N, ndim = np.shape(samples)

    return ndim, nwalkers, samples

def checkDifference_yieldVariance():
    samples = getPosteriorSamples('data/edelweiss_corr_C_systematicErrors_sampler_nll_allpars_gausPrior.h5')
