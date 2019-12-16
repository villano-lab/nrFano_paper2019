#!/usr/bin/env python

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import pandas as pd
#warnings.resetwarnings()
import re
import os
import time
import argparse
import prob_dist as prob
import fano_calc as fc
import resfuncRead as rfr
import time
from argparse import ArgumentParser, ArgumentTypeError
import h5py
import EdwRes as er
from sklearn.neighbors import KernelDensity
import scipy.interpolate as inter
# Author: A. Villano
#

#the stuff below is so this functionality can be used as a script
########################################################################
if __name__ == "__main__":

        #make a parser for the input
        parser = argparse.ArgumentParser(description='options')
        parser.add_argument('-n','--nsamp', type=int, dest='Nsamp', default=4, help='number of samples to draw')
        parser.add_argument('-i','--Eidx', type=int, dest='Eidx', default=-1, help='NR point index for energy')
        parser.add_argument('-E','--energy', type=float, dest='Etest', default=25.0, help='NR test energy')
        #parser.add_argument('-x','--regex', type=str, dest='regstr', default=r'(.*?)', help='regex for picking up files')
        #parser.add_argument('-o','--outfile', type=str, dest='outfile', default='data.h5', help='output file for data')
        #parser.add_argument('-c','--cuts', type=str, dest='cuts', default='NR', help='kind of cuts to apply')
        #parser.set_defaults(filedir='./');

        args = parser.parse_args()

        try:

         #find what energies relevant for NR/ER data
         # import data from Edelweiss
         resNR_data = pd.read_csv("data/edelweiss_NRwidth_GGA3_data.txt", skiprows=1, \
                       names=['E_recoil', 'sig_NR', 'E_recoil_err', 'sig_NR_err'], \
                       delim_whitespace=True)


         resER_data = pd.read_csv("data/edelweiss_ERwidth_GGA3_data.txt", skiprows=1, \
                         names=['E_recoil', 'sig_ER', 'sig_ER_err'], \
                         delim_whitespace=True)

         # set the data up for the fits
         # Edelweiss discards ER points near peaks
         # and first two NR points since they're affected by the threshold
         mask = [True, True, False, False, True, True, True, True, True]
         ER_data = {'Erecoil': resER_data["E_recoil"][mask], 'sigma': resER_data["sig_ER"][mask], 'sigma_err': resER_data["sig_ER_err"][mask]}
         NR_data = {'Erecoil': resNR_data["E_recoil"][2::], 'sigma': resNR_data["sig_NR"][2::], 'sigma_err': resNR_data["sig_NR_err"][2::]}

         E = np.sort(NR_data['Erecoil'])
         print(E)

         if args.Eidx==-1:
           Etest = args.Etest
         else:
           Etest = E[args.Eidx]

         print('Energy is: {:03.1f}'.format(Etest))

         filename = 'data/edelweiss_C_systematicErrors_sampler_nll_allpars_gausPrior.h5'
         #remove vars
         f = h5py.File(filename,'r')

         #get path within file
         path='{}/{}/'.format('mcmc','sampler')

         samples = np.asarray(f[path+'samples'])
         A_bf = np.asarray(f[path+'A'])
         B_bf = np.asarray(f[path+'B'])
         aH_bf = np.asarray(f[path+'aH'])
         mu_bf = np.asarray(f[path+'scale'])

         ndim = 6
         print(np.shape(samples))
         samples_corner = samples[:, 300:, :].reshape((-1, ndim))
         print(np.shape(samples_corner))
         f.close()

         #print(samples_corner[0:5,:])

         Nsamp = args.Nsamp 
         permute = np.zeros((Nsamp+2,4))
         #start with nominal params, then the best-fit params
         permute[0,0] = 0.16 
         permute[0,1] = 0.18 
         permute[0,2] = 0.0381 
         permute[0,3] = 1.0 

         permute[1,0] = A_bf
         permute[1,1] = B_bf
         permute[1,2] = aH_bf/er.FWHM_to_SIG 
         permute[1,3] = mu_bf 

         #for i, aH, C, m, scale, A, B in enumerate(samples_corner[np.random.randint(len(samples_corner), size=Nsamp)]):
         for i, parvec in enumerate(samples_corner[np.random.randint(len(samples_corner), size=Nsamp)]):
           #print(i)
           #print(parvec[0]/er.FWHM_to_SIG)
           aH = parvec[0]/er.FWHM_to_SIG
           A = parvec[4]
           B = parvec[5]
           mu = parvec[3]
           permute[i+2,0] = A
           permute[i+2,1] = B
           permute[i+2,2] = aH 
           permute[i+2,3] = mu 

         print(permute)

         #getting some sample parameter permutations
         #Nsamp = 3 
         #A0 = 0.16 
         #dA = 0.01 
         #B0 = 0.18 
         #dB = 0.01 
         #aH0 = 0.0381 
         #daH = 0.001 
         #mu0 = 1.0 
         #dmu = 0.05                                                                                                                                                  

         #Avec = np.concatenate(([A0],np.random.uniform(A0-dA,A0+dA,Nsamp))) 
         #Bvec = np.concatenate(([B0],np.random.uniform(B0-dB,B0+dB,Nsamp))) 
         #aHvec = np.concatenate(([aH0],np.random.uniform(aH0-daH,aH0+daH,Nsamp))) 
         #muvec = np.concatenate(([mu0],np.random.uniform(mu0-dmu,mu0+dmu,Nsamp))) 
         #print(Avec) 
         #print(Bvec) 
         #print(aHvec) 
         #print(muvec)                                                                                                                                                

         #permute=np.array(np.meshgrid(Avec, Bvec, aHvec,muvec)).T.reshape(-1,4) 
         #print(np.shape(permute))                                                                                                                                   

         output,frac = prob.diffmap(permute,Etest=Etest,outfile='data/sigdiff_test1.h5')  
         print(output)
         print(frac)

         start = time.time()
         corr_func = inter.NearestNDInterpolator(permute,output)
         print(corr_func([0.13,0.17,0.0381,1.0]))
         end = time.time()
         print('Interpolate Time: {:1.5f} sec.'.format(end-start))
        except KeyboardInterrupt:
          print('Shutdown requested .... exiting')
        except Exception:
          traceback.print_exc(file=sys.stderr)


