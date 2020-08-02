import numpy as np
from scipy.stats import multivariate_normal

import sys
sys.path.append('..')
from EdwRes import *
from prob_dist import *

def getCov_fromCoeff(a, b, c):
    Var_x = 4*np.pi/(np.sqrt(c)*np.power(4*a - b**2/c ,3/2))
    Var_y = 4*np.pi*a/np.power(4*a*c - b**2 ,3/2)
    Var_xy = 2*np.pi*b/np.power(4*a*c - b**2 ,3/2)
    
    return [[Var_x, Var_xy], [Var_xy, Var_y]]

def getCoeff(sig_x, sig_y, theta):
    a = np.power(np.cos(theta),2)/(2*sig_x*sig_x) + np.power(np.sin(theta),2)/(2*sig_y*sig_y)
    b = 2*np.cos(theta)*np.sin(theta)/(2*sig_y*sig_y) - 2*np.cos(theta)*np.sin(theta)/(2*sig_x*sig_x)
    c = np.power(np.cos(theta),2)/(2*sig_y*sig_y) + np.power(np.sin(theta),2)/(2*sig_x*sig_x)
    
    return a, b, c
    
def getCov(sig_x, sig_y, theta):
    a, b, c = getCoeff(sig_x, sig_y, theta)
    return getCov_fromCoeff(a, b, c)

def evalBivariatePDF(energies_keV, mu_x, mu_y, sig_x, sig_y, A, B):
    pdf_arr = []
    
    for Erecoil_keV in energies_keV:
        slope = -1/(0.16*np.log(Erecoil_keV))
        theta = np.arctan(slope) + np.pi/2
        cov_matrix = getCov(sig_x, sig_y, theta)
        rv = multivariate_normal([mu_x, mu_y], cov_matrix)
        pdf_arr.append(rv.pdf([[A, B]]))
        
    return pdf_arr

def log_likelihood(theta, ER_data, NR_data):
    aH, C, m, scale, A, B = theta
    V = np.abs(scale)*4.0
    
    # extract the data
    x_ER, y_ER, yerr_ER = ER_data['Erecoil'], ER_data['sigma'], ER_data['sigma_err']
    x_NR, y_NR, yerr_NR = NR_data['Erecoil'], NR_data['sigma'], NR_data['sigma_err']
    
    # expected parameter values and widths
    # uncertainty on aH is the uncertainty on the parameter aH when fitting only the ER band
    # scale width estimated by assuming a 10 mV error on V and 0.5 eV error on epsilon
    # information for A and B from Astroparticle Physics 14 (2001) 329±337
    # Anthony's function use the FWHM version of aH
    exp_aH = 0.016*2*np.sqrt(2*np.log(2))
    exp_aH_sig = exp_aH*0.046
    exp_scale = 1
    exp_scale_sig = 0.17
    exp_A = 0.16
    exp_A_sig = 0.07
    exp_B = 0.18
    exp_B_sig = 0.1
    exp_Y = 0.3
    exp_Y_sig = 0.1
    
    ## get the ER prediction for the input parameters
    model_ER = [np.sqrt(series_NRQ_var(x,V=V,aH=aH,A=1.0,B=0.0,alpha=0.0000001)) for x in x_ER]
    sigma2_ER = yerr_ER**2
    
    ## get the NR prediction for the input parameters
    # series_NRQ_var_corr1(Er=10.0,F=0.0,V=4.0,aH=0.0381,alpha=(1/18.0),A=0.16,B=0.18,label='GGA3',corr1file='data/sigdiff_test.h5')
    # series_NRQ_var_corr1 returns the *variance*
    model_NR_0 = [series_NRQ_var_corr1(x, 0, V, aH, 1/18.0, A, B, 'GGA3') for x in x_NR] 
    model_NR = np.sqrt(np.power(C + m*x_NR, 2) + model_NR_0)
    sigma2_NR = yerr_NR**2
    
    return -0.5*(np.sum((y_NR-model_NR)**2/sigma2_NR + np.log(2*np.pi*sigma2_NR)) \
                 + np.sum((y_ER-model_ER)**2/sigma2_ER + np.log(2*np.pi*sigma2_ER)) \
                 + (aH - exp_aH)**2/exp_aH_sig**2 \
                 + (scale - exp_scale)**2/exp_scale_sig**2) \
                 + np.sum(np.log(evalBivariatePDF([np.max(x_NR)], exp_A, exp_B, exp_A_sig, exp_B_sig, A, B)))