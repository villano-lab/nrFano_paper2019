# import "standard" python modules
import numpy as np
import h5py
from datetime import datetime
from astropy.table import Table, Column
import os
import argparse

# import our custom NR Fano analysis code
import sys
sys.path.append('../python/')
from EdwRes import *
from prob_dist import *

def getPosteriorSamples(filename):
    # get the data
    # the posterior distribution is in samples
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

    return ndim, nwalkers, N, samples

def checkDifference_yieldVariance(Erecoil, numSamples, posteriorFile, datadir='./data'):
    # get the samples
    # for the most accurate fit, 'data/edelweiss_corr_C_systematicErrors_sampler_nll_allpars_gausPrior.h5'
    ndim, nwalkers, nsteps, samples = getPosteriorSamples(posteriorFile)
    
    # reshape the samples
    samples = samples[:, 300:, :].reshape((-1, ndim))
    #print(len(samples))

    # not wise to ask for more samples than there were steps in the origianl
    # sample chain
    if numSamples > nsteps:
        numSamples = nsteps
        print ("You are requesting more samples than were in the original sample chain. \n Reducing the number of samples to ", numSamples)

    aH_col, C_col, m_col, scale_col, A_col, B_col = [], [], [], [], [], []
    sig_yield_col, sig_yield_estimate_col = [], []
    energy_col = np.repeat(Erecoil, numSamples) #np.full((numSamples, 1), Erecoil)

    for aH, C, m, scale, A, B in samples[np.random.randint(len(samples), size=numSamples)]:
        V = scale*4.0 #,'eps_eV' : 3.0, 'a': A, 'b': B

        # calculate the yield standard deviation used for fitting in edelweiss_fit_allParameters_
        ## get the NR prediction for the input parameters
        # series_NRQ_var_corr1(Er=10.0,F=0.0,V=4.0,aH=0.0381,alpha=(1/18.0),A=0.16,B=0.18,label='GGA3',corr1file='data/sigdiff_test.h5')
        # series_NRQ_var_corr1 returns the *variance*
        model_NR_0 = np.sqrt(series_NRQ_var_corr1(Erecoil, 0, V, aH, 1/18.0, A, B, 'GGA3')) 
        #model_NR = np.sqrt(np.power(C + m*Erecoil, 2) + model_NR_0)


        # calculate the exact yield standard deviation
        """
        sig_real = []
        for Er_val in Er:
            sig_real.append(sigmomEdw(Er_val, band='NR', F=0.000001, V=scale*4.0, aH=aH, alpha=(1/100), A=A, B=B))
        """
        true_NR_sig = sigmomEdw(Erecoil,band='NR',label='GGA3',F=0.000001,V=V,aH=aH,alpha=(1/18.0), A=A, B=B)

        # store the parameter data
        #print (aH, C, m, scale, A, B)
        aH_col.append(aH)
        C_col.append(C)
        m_col.append(m)
        scale_col.append(scale)
        A_col.append(A)
        B_col.append(B)
    
        # and store the yield information
        sig_yield_col.append(true_NR_sig)
        sig_yield_estimate_col.append(model_NR_0)

    #############################
    # Store the information
    #############################
    now = datetime.now()
    time = now.strftime('%Y%h%d_%H%M')
    #print (time)
    filename = os.path.join(datadir, 'yield_accuracy_Erecoil_%.2f_keV_%s.h5' % (Erecoil, time))
    #print(filename)

    # make an astropy table
    # thank you astropy!!
    # energy would probably be handled better with metadata but OH WELL
    # having it in a column is easier for retrieval from the hdf5 file
    data_tab = Table()
    data_tab['energy_recoil_keV'] = energy_col
    data_tab['aH'] = aH_col
    data_tab['C'] = C_col
    data_tab['m'] = m_col
    data_tab['scale'] = scale_col
    data_tab['A'] = A_col
    data_tab['B'] = B_col
    
    data_tab['true_yield_sig'] = sig_yield_col
    data_tab['cor1_yield_sig'] = sig_yield_estimate_col
    
    #print(data_tab)
    data_tab.write(filename, format='hdf5', path='table')
    return filename

def main(args):
    # We'll look at the Er values of the data points
    # import data from Edelweiss
    resNR_filename = os.path.join(args.repoPath, 'analysis_notebooks/data/edelweiss_NRwidth_GGA3_data.txt')
    resNR_data = pd.read_csv(resNR_filename, skiprows=1, \
                           names=['E_recoil', 'sig_NR', 'E_recoil_err', 'sig_NR_err'], \
                           delim_whitespace=True)

    # the sorting is necessary!
    # otherwise the mask defined below will select the wrong data
    resNR_data = resNR_data.sort_values(by='E_recoil')
    NR_data = {'Erecoil': resNR_data["E_recoil"][2::], 'sigma': resNR_data["sig_NR"][2::], 'sigma_err': resNR_data["sig_NR_err"][2::]}
    Er = np.sort(NR_data['Erecoil'])
    Erecoil = Er[args.energyIndex]

    # generate and store the data
    MCMC_data_filename = os.path.join(args.repoPath, 'analysis_notebooks/data/edelweiss_corr_C_systematicErrors_sampler_nll_allpars_gausPrior.h5')
    checkDifference_yieldVariance(Erecoil, args.numSamples, MCMC_data_filename, args.dataPath)

"""
Example use:
(nr_fano) aroberts@DESKTOP-F1SLP9K python$ python checkDifference_yieldVariance.py --energyIndex 0 --numSamples 5 --repoPath "/mnt/c/Users/canto/Repositories/nrFano_paper2019" --dataPath "../
analysis_notebooks/data"
"""
if __name__ == "__main__":
    # execute only if run as a script

    # add several arguments: which energy and how many samples
    parser = argparse.ArgumentParser(description='Sample the posterior distribution')
    parser.add_argument('--energyIndex', type=int, 
                       help='an integer between 0 and 4 to specify the energy index')
    parser.add_argument('--numSamples', type=int,
                       help='number of samples to draw from the posterior distribution')
    parser.add_argument('--repoPath', 
                       help='path to the repository')
    parser.add_argument('--dataPath', 
                       help='path to the repository')

    args = parser.parse_args()

    main(args)