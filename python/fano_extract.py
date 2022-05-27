import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import h5py
#warnings.resetwarnings()
from scipy.integrate import quad
import resfuncRead as rfr
import scipy.optimize as so
import prob_dist as pd
import os
import scipy.interpolate as inter
from scipy import optimize

def get_sigvec(Evec,csample,cpsample):

    #check Evec to be an numpy vector of appropriate shape?

    #check csample and cpsample for correct size

    #construct rhs
    #print(csample[3])
    var = lambda E: pd.series_NRQ_var_corr1(E,0.0,csample[3]*4.0,csample[0],(1/18.0),csample[4],csample[5])
    varv = np.vectorize(var)
    #print(varv(Evec))

    #below is diagnostic for ms fit
    #print(np.sqrt(varv(Evec)+(cpsample[0]+Evec*cpsample[1])**2))
    #below is diagnostic for main fit
    #print(np.sqrt(varv(Evec)+(csample[1]+Evec*csample[2])**2))

    retv = np.sqrt(varv(Evec)-(cpsample[0]+Evec*cpsample[1])**2+(csample[1]+Evec*csample[2])**2)

    return retv

def get_root_from_sigvec(Evec,csample,cpsample):

    #get the vector of sigmas
    sigvec = get_sigvec(Evec,csample,cpsample)

    #loop through Eveca
    Fvec = np.zeros(np.shape(Evec))
    for Er,i in enumerate(Evec):

      #set up the fano function
      fano = lambda F: np.sqrt(pd.series_NRQ_var_corr1(Er,F,csample[3]*4.0,csample[0],(1/18.0),csample[4],csample[5])) - sigvec[i]

      print(pd.series_NRQ_var_corr1(Er,10,csample[3]*4.0,csample[0],(1/18.0),csample[4],csample[5]))
      #get the root
      #Fvec[i] = optimize.root_scalar(fano, bracket=[0.1, 200], method='brentq').root 

    return Fvec
