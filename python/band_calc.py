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
          V=4.0
          FWHM_to_SIG = 1 / (2*np.sqrt(2*np.log(2)))
          eps=3/1000.0
          alpha=(1/100)
          alphap=(1/100000)
          F=15
          Fp = 0.0001
          #yield models
          a=0.16
          b=0.18
          Qbar = lambda Er: a*Er**b
          Qer = lambda Er: 1
          aH=0.035
          (nrsigma,nrE) = fc.calcQWidth(100,F,V,eps,alpha,Qbar,aH,'../paper_notebooks/')
          (ersigma,erE) = fc.calcQWidth(100,Fp,V,eps,alphap,Qer,aH,'../paper_notebooks/')
        except KeyboardInterrupt:
          print('Shutdown requested .... exiting')
        except Exception:
          traceback.print_exc(file=sys.stderr)


