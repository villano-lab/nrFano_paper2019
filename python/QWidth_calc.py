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
          fc.storeQWidth(200,filename='../paper_notebooks/data/res_calc.h5',maxEr=200)
          fc.storeQWidth(200,filename='../paper_notebooks/data/res_calc.h5',band='NR',alpha=(1/18.0),maxEr=200)
          fc.storeQWidth(200,filename='../paper_notebooks/data/res_calc.h5',band='NR',maxEr=200)
          fc.storeQWidthVaryF(200,filename='../paper_notebooks/data/res_calc.h5',MSfile='../paper_notebooks/data/mcmc_fits.h5',Ffile='../paper_notebooks/data/mcmc_fano.h5',band='NR',alpha=(1/18.0),maxEr=200,erase=False)
          fc.storeFMCMC(50,infile='../paper_notebooks/data/mcmc_fits.h5',filename='../paper_notebooks/data/mcmc_fano.h5',erase=True)
        except KeyboardInterrupt:
          print('Shutdown requested .... exiting')
        except Exception:
          traceback.print_exc(file=sys.stderr)


