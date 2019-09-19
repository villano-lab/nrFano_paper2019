import numpy as np
import re
from functools import partial

# note that np.log() is the natural log (base e)
FWHM_to_SIG = 1 / (2*np.sqrt(2*np.log(2)))

# EPS_eV, the average energy required to create an e/h pair
EPS_eV = 3.0

def get_heatRes(sig0, a, E_keV):
    """return the heat resolution (1 sigma) at energy E_keV.  sig0, E_keV assumed to be in units of keV."""
    
    # see eqn (5) in 2004 NIMA Edelweiss paper
    sigH = np.sqrt(sig0**2 + (a*E_keV)**2)
    
    # multiply by 2.355 to get FWHM
    return sigH

def get_heatRes_func(FWHM0, FWHM122, aH=None):
    """returns a resolution function given the FWHM values at 0 keV and 122 keV"""
    
    # convert from FWHM to sigma
    # note that aH is calculated from the FWHM values
    # in the Edelweiss paper, so the aH values they report
    # are 2.355 times larger than values of a calculated using eq 5
    sig0 = FWHM0 * FWHM_to_SIG
    sig122 = FWHM122 * FWHM_to_SIG
    
    # calculate aH, which is unitless
    if aH is None:
        calib_energy_keV = 122
        aH = np.sqrt((sig122**2 - sig0**2)/calib_energy_keV**2)

    #print ("aH is: ", aH)
    
    # create function
    return partial(get_heatRes, sig0, aH)

def get_ionRes_func(FWHM_center, FWHM_guard, FWHM122):
    FWHM0 = np.sqrt(FWHM_center**2 + FWHM_guard**2)
    
    return get_heatRes_func(FWHM0, FWHM122)

def Q_avg(E_keV, a=None, b=None):
    if a is None:
      a = 0.16
    if b is None:
      b = 0.18

    #print ("in Q_avg", a,b)
    return a*np.power(E_keV,b)

def get_sig_gamma(sigI, sigH, pars, E_keV):
    V = pars['V']
    eps_eV = pars['eps_eV']

    return ((1+V/eps_eV)/E_keV)*np.sqrt((sigI(E_keV))**2 + (sigH(E_keV))**2)

def get_sig_neutron(sigI, sigH, pars, C, Er_keV):
    V = pars['V']
    eps_eV = pars['eps_eV']
    A = pars.get('a')
    B = pars.get('b')
    #print ("in get_sig_neutron", A, B)

    E_keVee_I = np.multiply(Q_avg(Er_keV, A, B), Er_keV)
    E_keVee_H = np.multiply((1+(V/eps_eV)*Q_avg(Er_keV, A, B))/(1+(V/eps_eV)), Er_keV)
    # we're pretty sure Edelweiss uses the correct (above) conversion
    # and not the incorrect (below) conversion
    #E_keVee_H = np.multiply(Q_avg(Er_keV, a, b), Er_keV)

    a = np.multiply(1+(V/eps_eV)*Q_avg(Er_keV, A, B), sigI(E_keVee_I))
    b = np.multiply((1+V/eps_eV)*Q_avg(Er_keV, A, B), sigH(E_keVee_H))

    sig_0 = (1/Er_keV)*np.sqrt(a**2 + b**2)

    if C is not None:
        return np.sqrt(np.power(sig_0,2) + np.power(C,2))
    else:
        return sig_0

def get_sig_neutron_alt(sigI, sigH, pars, C, m, Er_keV):
    V = pars['V']
    eps_eV = pars['eps_eV']
    A = pars.get('a')
    B = pars.get('b')

    E_keVee_I = np.multiply(Q_avg(Er_keV, A, B), Er_keV)
    E_keVee_H = np.multiply((1+(V/eps_eV)*Q_avg(Er_keV, A, B))/(1+(V/eps_eV)), Er_keV)
    # we're pretty sure Edelweiss uses the correct (above) conversion
    # and not the incorrect (below) conversion
    #E_keVee_H = np.multiply(Q_avg(Er_keV, a, b), Er_keV)

    a = np.multiply(1+(V/eps_eV)*Q_avg(Er_keV, A, B), sigI(E_keVee_I))
    b = np.multiply((1+V/eps_eV)*Q_avg(Er_keV, A, B), sigH(E_keVee_H))

    sig_0 = (1/Er_keV)*np.sqrt(a**2 + b**2)

    if C is not None:
        return np.sqrt(np.power(sig_0,2) + np.power(C+m*Er_keV,2))
    else:
        return sig_0

def get_sig_gamma_func(FWHM_center, FWHM_guard, FWHM122_ion, FWHM0_heat, FWHM122_heat, pars, aH=None, C=None):    
    # get the ionization resolution function
    sigI = get_ionRes_func(FWHM_center, FWHM_guard, FWHM122_ion)
    
    # get the heat resolution function
    sigH = get_heatRes_func(FWHM0_heat, FWHM122_heat, aH)
    
    return partial(get_sig_gamma, sigI, sigH, pars)


def get_sig_nuc_func(FWHM_center, FWHM_guard, FWHM122_ion, FWHM0_heat, FWHM122_heat, pars, aH=None, C=None):
    # get the ionization resolution function
    sigI = get_ionRes_func(FWHM_center, FWHM_guard, FWHM122_ion)
    
    # get the heat resolution function
    sigH = get_heatRes_func(FWHM0_heat, FWHM122_heat, aH)
    
    return partial(get_sig_neutron, sigI, sigH, pars, C)

def get_sig_nuc_func_alt(FWHM_center, FWHM_guard, FWHM122_ion, FWHM0_heat, FWHM122_heat, pars, aH=None, C=None, m=None):
    # get the ionization resolution function
    sigI = get_ionRes_func(FWHM_center, FWHM_guard, FWHM122_ion)
    
    # get the heat resolution function
    sigH = get_heatRes_func(FWHM0_heat, FWHM122_heat, aH)

    return partial(get_sig_neutron_alt, sigI, sigH, pars, C, m)

def getEdw_res_pars(infile='data/edw_res_data.txt'):


    #open the file return a dictionary with label,FWHM 0keV ion, FWHM 0 keV guard,
    #FWHM 0 keV heat, FWHM 122 keV ion, FWHM 122 keV heat as elements
    f = open(infile)

    #make a list for vector identifier 
    #first two are x-y of histogram-type step function
    #second two are a sort-of smooth curve to represent the function
    vecs = ['label','FWHM0_ion','FWHM0_guard','FWHM0_heat','FWHM122_ion','FWHM122_heat']

    #make a dictionary to store the pulses
    funcs = {}

    #read file N times, is this efficient?
    regex=re.compile(r'^\s*#.+')
    #[print(regex.search(x)) for x in f.readlines()]
    funcs[vecs[0]] = [x.split()[0] for x in f.readlines() if regex.search(x) is None]
    f.seek(0)
    funcs[vecs[1]] = [x.split()[1] for x in f.readlines() if regex.search(x) is None]
    f.seek(0)
    funcs[vecs[2]] = [x.split()[2] for x in f.readlines() if regex.search(x) is None]
    f.seek(0)
    funcs[vecs[3]] = [x.split()[3] for x in f.readlines() if regex.search(x) is None]
    f.seek(0)
    funcs[vecs[4]] = [x.split()[4] for x in f.readlines() if regex.search(x) is None]
    f.seek(0)
    funcs[vecs[5]] = [x.split()[5] for x in f.readlines() if regex.search(x) is None]

    f.close()

    #convert to floats
    funcs[vecs[1]] = [float(i) for i in funcs[vecs[1]]]
    funcs[vecs[2]] = [float(i) for i in funcs[vecs[2]]]
    funcs[vecs[3]] = [float(i) for i in funcs[vecs[3]]]
    funcs[vecs[4]] = [float(i) for i in funcs[vecs[4]]]
    funcs[vecs[5]] = [float(i) for i in funcs[vecs[5]]]

    outdic = {}
    for i,val in enumerate(funcs['label']):
      vec = [funcs['FWHM0_ion'][i],funcs['FWHM0_guard'][i],funcs['FWHM0_heat'][i],funcs['FWHM122_ion'][i],funcs['FWHM122_heat'][i]]
      outdic[funcs['label'][i]] = vec

    return outdic

def getEdw_det_res(label='GGA3',V=4.0,infile='data/edw_res_data.txt',aH=None,C=None):

    eps=3.0

    #yield models
    a=0.16
    b=0.18
    Qbar = lambda Er: a*Er**b
    Qer = lambda Er: 1

    pars = getEdw_res_pars(infile)[label]
    #print(pars)

    if aH is None:
      sigH = get_heatRes_func(pars[2], pars[4])
    else:
      sigH = get_heatRes_func(pars[2], pars[4],aH*FWHM_to_SIG)

    sigI = get_ionRes_func(pars[0], pars[1], pars[3])


    #new resolution functions 
    Ehee = lambda Er: ((1+(V/(eps))*Qbar(Er))*Er)/(1+(V/(eps)))
    EIee = lambda Er: Qbar(Er)*Er


    sigH_NR = lambda Er: sigH(Ehee(Er))

    sigI_NR = lambda Er: sigI(EIee(Er))

    sigQer = lambda Etr: (1/Etr)*np.sqrt((1+(V/(eps))*Qer(Etr))**2*sigI(Etr)**2 + (1+(V/(eps)))**2*Qer(Etr)**2 \
                                             *sigH(Etr)**2)
    
    sigQnr_base = lambda Etr: (1/Etr)*np.sqrt((1+(V/(eps))*Qbar(Etr))**2*sigI_NR(Etr)**2 + (1+(V/(eps)))**2 \
                                             *Qbar(Etr)**2*sigH_NR(Etr)**2)
    #add C if specified
    if C is not None:
      sigQnr = lambda Etr: np.sqrt(sigQnr_base(Etr)**2 + C**2)
    else:
      sigQnr = lambda Etr: np.sqrt(sigQnr_base(Etr)**2)

    sigHv = np.vectorize(sigH)
    sigIv = np.vectorize(sigI)
    sigH_NRv = np.vectorize(sigH_NR)
    sigI_NRv = np.vectorize(sigI_NR)
    sigQerv = np.vectorize(sigQer)
    sigQnrv = np.vectorize(sigQnr)
    return (sigHv,sigIv,sigQerv,sigH_NRv,sigI_NRv,sigQnrv)
