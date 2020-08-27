import numpy as np
import pandas as pd
import h5py
import emcee
import os

import sys
sys.path.append('../python/')
sys.path.append('../python/log_likelihood_funcs/')
from EdwRes import *
from prob_dist import *
from edw_data_util import *

def runMCMC(ll_file, data_file, nwalkers, nburn, nsteps, pos0, sampler_args):
    import importlib
    ll = importlib.import_module(ll_file, package=None)

    ndim = np.size(pos0)
    [aH_fit, C_fit, m_fit, scale_fit, A_fit, B_fit] = pos0

    # sample the posterior distribution
    pos0 = [pos0 + 1e-8*np.random.randn(ndim) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, ll.log_likelihood, args=sampler_args)
    pos, prob, state = sampler.run_mcmc(pos0, nburn, storechain=False)
    sampler.reset()
    sampler.run_mcmc(pos, nsteps)
    samples = sampler.chain

    # save the sampler data
    path='{}/{}/'.format('mcmc','sampler')

    #remove vars
    f = h5py.File(data_file,'a')
    exaH = path+'aH' in f
    exC = path+'C' in f
    exm = path+'m' in f
    exscale = path+'scale' in f
    exA = path+'A' in f
    exB = path+'B' in f
    exsamp = path+'samples' in f

    if exaH:
      del f[path+'aH']
    if exC:
      del f[path+'C']
    if exm:
      del f[path+'m']
    if exscale:
      del f[path+'scale']
    if exA:
      del f[path+'A']
    if exB:
      del f[path+'B']
    if exsamp:
      del f[path+'samples']

    dset = f.create_dataset(path+'aH',np.shape(aH_fit),dtype=np.dtype('float64').type)
    dset[...] = aH_fit
    dset = f.create_dataset(path+'C',np.shape(C_fit),dtype=np.dtype('float64').type)
    dset[...] = C_fit
    dset = f.create_dataset(path+'m',np.shape(m_fit),dtype=np.dtype('float64').type)
    dset[...] = m_fit
    dset = f.create_dataset(path+'scale',np.shape(scale_fit),dtype=np.dtype('float64').type)
    dset[...] = scale_fit
    dset = f.create_dataset(path+'A',np.shape(A_fit),dtype=np.dtype('float64').type)
    dset[...] = A_fit
    dset = f.create_dataset(path+'B',np.shape(B_fit),dtype=np.dtype('float64').type)
    dset[...] = B_fit
    dset = f.create_dataset(path+'samples',np.shape(samples),dtype=np.dtype('float64').type, \
    compression="gzip",compression_opts=9)
    dset[...] = samples

    f.close()