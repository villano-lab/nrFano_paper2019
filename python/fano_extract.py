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

def get_sigvec(Evec,csample,cpsample):

    #check Evec to be an numpy vector of appropriate shape?

    #check csample and cpsample for correct size

    #construct rhs
    #print(csample[3])
    var = lambda E: pd.series_NRQ_var_corr1(E,0.0,csample[3]*4.0,csample[0],(1/18.0),csample[4],csample[5])
    varv = np.vectorize(var)
    #print(varv(Evec))

    print(np.sqrt(varv(Evec)+(cpsample[0]+Evec*cpsample[1])**2))

    return
