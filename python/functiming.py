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
          #v = getPosition(args.iteration[0],args.n,pit) 
          ptres = rfr.getRFunc('../data/jardin_ptres.txt')

          qres = rfr.getRFunc('../data/jardin_qsummaxres.txt')
          
          fp = rfr.makeRFunc(ptres[1]['sqrt'])
          
          fq = rfr.makeRFunc(qres[1]['lin'],True)
          
          sigp = lambda x: fp(x) #convert from eV
          
          sigq = lambda x: fq(x) #convert from eV
          
          f = pd.YEr_v2_2D(sigp,sigq,4,(3.3/1000),1)
          ff = pd.YEr_v2_2D_fast(sigp,sigq,4,(3.3/1000),1)
          
          
          #print(f(0.25,10,10))
          
          
          g = pd.YErSpec_v2_2D(f)
          gg = pd.YErSpec_v2_2D(ff)
          
          #print(g(0.06,10))

          print('simple sin function: ')
          start = time.time()
          np.sin(30)
          end = time.time()
          print('{} s'.format(end - start))

          print('pre-integrated P(Y,Etr,Er)')
          start = time.time()
          f(0.25,40,40)
          end = time.time()
          print('{} s'.format(end - start))
          
          print('pre-integrated P(Y,Etr,Er) - fast')
          start = time.time()
          ff(0.25,40,40)
          end = time.time()
          print('{} s'.format(end - start))

          print('integrated P(Y,Etr)')
          start = time.time()
          g(0.07,10)
          end = time.time()
          print('{} s'.format(end - start))

          print('integrated P(Y,Etr) - fast')
          start = time.time()
          gg(0.07,10)
          end = time.time()
          print('{} s'.format(end - start))

          print('root search for sigY(Er,F)')
          start = time.time()
          pd.sigroot(10,10)
          end = time.time()
          print('{} s'.format(end - start))

        except KeyboardInterrupt:
          print('Shutdown requested .... exiting')
        except Exception:
          traceback.print_exc(file=sys.stderr)


