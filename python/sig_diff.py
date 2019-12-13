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
import prob_dist as pd
import fano_calc as fc
import resfuncRead as rfr
import time
from argparse import ArgumentParser, ArgumentTypeError
# Author: A. Villano
#

#the stuff below is so this functionality can be used as a script
########################################################################
if __name__ == "__main__":

        #make a parser for the input
        parser = argparse.ArgumentParser(description='options')
        #parser.add_argument('-d','--filedir', type=str, dest='filedir', default='./', help='directory to look for files')
        #parser.add_argument('-x','--regex', type=str, dest='regstr', default=r'(.*?)', help='regex for picking up files')
        #parser.add_argument('-o','--outfile', type=str, dest='outfile', default='data.h5', help='output file for data')
        #parser.add_argument('-c','--cuts', type=str, dest='cuts', default='NR', help='kind of cuts to apply')
        #parser.set_defaults(filedir='./');

        args = parser.parse_args()

        try:
         Nsamp = 3 
         A0 = 0.16 
         dA = 0.01 
         B0 = 0.18 
         dB = 0.01 
         aH0 = 0.0381 
         daH = 0.001 
         mu0 = 1.0 
         dmu = 0.05                                                                                                                                                  

         Avec = np.concatenate(([A0],np.random.uniform(A0-dA,A0+dA,Nsamp))) 
         Bvec = np.concatenate(([B0],np.random.uniform(B0-dB,B0+dB,Nsamp))) 
         aHvec = np.concatenate(([aH0],np.random.uniform(aH0-daH,aH0+daH,Nsamp))) 
         muvec = np.concatenate(([mu0],np.random.uniform(mu0-dmu,mu0+dmu,Nsamp))) 
         print(Avec) 
         print(Bvec) 
         print(aHvec) 
         print(muvec)                                                                                                                                                

         permute=np.array(np.meshgrid(Avec, Bvec, aHvec,muvec)).T.reshape(-1,4) 
         print(np.shape(permute))                                                                                                                                   

         #pd.diffmap(permute,Etest=25.0,outfile='data/sigdiff_test.h5')   
        except KeyboardInterrupt:
          print('Shutdown requested .... exiting')
        except Exception:
          traceback.print_exc(file=sys.stderr)


