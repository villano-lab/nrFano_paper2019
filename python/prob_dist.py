import numpy as np
from scipy.special import erf
import math 
from scipy.integrate import quad
import resfuncRead as rfr
import scipy.optimize as so
import EdwRes as er
import edw_data_util as edu 
import pandas as pd
import scipy.interpolate as inter

#all needed for diff mapping and nothing else
import fano_calc as fc
import h5py
import time



# returns the probability of z, where z is the yield
# z = Eq/(Ep - k*Eq)
# this distribution assumes that Eq and Ep are independent, 
# Eq has mean mu_q and width sig_q and is gaussian
# Ep has mean mu_p and width sig_p and is gaussian
# res_p = mu_p/sig_p
# res_q = mu_q/sig_q
# r = sig_p/sig_q
# k is defined as e*voltage/(energy needed to create one e/h pair)
# so for a Si detector at e.g. 4V, k ~ 4/3 
def ratio_dist_v1(z, res_p, res_q, r, k):
    F1 = np.exp(-0.5*(res_q**2 + res_p**2)) / (np.pi*(r*z**2 + (1/r)*(1+k*z)*2))
    
    G11 = (r*(z*res_q*r + (1+k*z)*res_p)) / (np.sqrt(2*np.pi)*np.power(z**2 * r**2 + (1+k*z)**2, 3/2))
    G12 = np.exp(-(z*res_p*r - (1+k*z)*res_q)**2 / (2*(z**2 * r**2 + (1+k*z)**2)))
    G13 = erf((z*res_q + (1+k*z)*res_p/r) / np.sqrt(2*(z**2 + (1+k*z)**2 / r**2)))

    return F1 + G11*G12*G13

# x is the yield, Eq/Er
# Er is the recoil energy, in units of keV
# meanN is the mean number of e/h pairs created given Er
# sdP is the standard deviation of the phonon signal, units of ??
# sdQ is the standard deviation of the charge signal, units of ??
# sdN is the standard deviation of the number of electron-hole pairs, unitless
# V is the voltage across the detector, in units of kV??

def ratio_dist_v2(x, Er, meanN, sdP, sdQ, sdN, V,e):


    
    
    k = (sdP**2)*(sdQ**2)+(V**2)*(sdQ**2)*(sdN**2)+(e**2)*(sdN**2)*(sdP**2)

    A = ((((x*(V/e)+1)*sdQ)**2)+((x*sdP)**2)+((e*sdN)**2))/(2*k)

    B = ((V/e)*(sdQ**2)*(Er*x+e*meanN)+x*e*meanN*(((V*sdQ/e)**2)+(sdP**2))+Er*((sdQ**2)+((e*sdN)**2)))/(k)

    C = ((((meanN*V+Er)*sdQ)**2)+(((meanN*sdP)**2)+((Er*sdN)**2))*(e**2))/(2*k)
    
    D = (B**2/(4*A)) - C

    #ans = (1/(2*np.sqrt(np.pi*k)))*(1/A(x))*g((B(x))/(2*np.sqrt(A(x))))*np.exp(-C)

    ans = (1/(2*np.sqrt(np.pi*k)))*(1/A)*((np.exp(-C)/(np.sqrt(np.pi))) + B/(2*np.sqrt(A))*np.exp(D)*erf(B/(2*np.sqrt(A))))

    return ans

def expband_2D(f,alpha=(1/100),widthfac=1):

    pnr = lambda Er: (1/alpha)*np.exp(-alpha*Er)


    #only integrate over important part of distribution around Etr
    #I empirically found in analysis/misc/nrFano_Constraint/extract_Edw_Fano_v2.ipynb
    #that at Etr=10keV the width should be 3 keV and at 40 keV it should be 10 keV
    m = (10-3.0)/(40-10)
    b = 3-m*10
    width = lambda Etr: m*Etr + b

    new_width = lambda r: np.piecewise(np.float(r), [r<=0, r > 0], [lambda r: 0.0, lambda r: r*m + b])

    Y_Erdist = lambda Er,Y,Etr: f(Y,Etr,Er)*pnr(Er)
    #Y_Er = lambda Y,Etr: quad(Y_Erdist, 0.1, np.inf,limit=100,args=(Y,Etr,))[0]
    Y_Er = lambda Y,Etr: quad(Y_Erdist, Etr-widthfac*new_width(Etr), Etr+widthfac*new_width(Etr),limit=100,args=(Y,Etr,))[0]

    return Y_Er

def expband_EpEq_2D_slow(Ep0,Eq0,f,alpha=(1/100),widthfac=5,V=4.0,eps=3.3/1000.0,sigp=lambda Er: 0.1):


    #it doesn't matter which one of the below we use because we're only integrating from zero to inf
    #I tried the piecewise function to test that. 
    #pnr = lambda Er: (1/alpha)*np.exp(-alpha*Er)
    pnr = lambda Er: np.piecewise(Er,[Er<0,Er>=0],[lambda Er: 0, lambda Er: (1/alpha)*np.exp(-alpha*Er)])


    #get the central value for Er
    Erec = lambda Ep,Eq: np.amax([Ep-(V/(1000*eps))*Eq,0])
    width = lambda Ep,Eq: widthfac*sigp(Erec(Ep,Eq))

    #get the full distribution
    Ep_Eqdist = lambda Er,Ep,Eq: f(Ep,Eq,Er)*pnr(Er)

    #Ep_Eq = lambda Ep,Eq: quad(Ep_Eqdist, np.amax([Erec(Ep,Eq)-width(Ep,Eq),0]), Erec(Ep,Eq)+width(Ep,Eq),limit=100,args=(Ep,Eq,))[0]

    intlow = np.amax([Erec(Ep0,Eq0)-width(Ep0,Eq0),0])
    inthigh = Erec(Ep0,Eq0)+width(Ep0,Eq0)

    print([intlow,inthigh])
    funcmin = lambda x,y: -quad(Ep_Eqdist,x,y,limit=100,args=(Ep0,Eq0,))[0]
    #f = lambda x,y: -quadrature(Ep_Eqdist,x,y,maxiter=100,args=(Ep0,Eq0,))[0]

    def fvec(x,f=funcmin):
      if(x[0]>=0):
        return f(x[0],x[1])
      elif(x[0]<0):
        return f(0,x[1])
 
    #print(funcmin(intlow,inthigh))
    
    (mini,fmini,a,b,c) = so.fmin(fvec,[intlow,inthigh],disp=False,full_output=True)
    print(mini)
    #print(fmini)
    return -fmini 


def expband_EpEq_2D(f,alpha=(1/100),widthfac=5,V=4.0,eps=3.3/1000.0,sigp=lambda Er: 0.1):


    #it doesn't matter which one of the below we use because we're only integrating from zero to inf
    #I tried the piecewise function to test that. 
    #pnr = lambda Er: (1/alpha)*np.exp(-alpha*Er)
    pnr = lambda Er: np.piecewise(Er,[Er<0,Er>=0],[lambda Er: 0, lambda Er: (1/alpha)*np.exp(-alpha*Er)])


    #get the central value for Er
    Erec = lambda Ep,Eq: np.amax([Ep-(V/(1000*eps))*Eq,0])
    width = lambda Ep,Eq: widthfac*sigp(Erec(Ep,Eq))

    #get the full distribution
    Ep_Eqdist = lambda Er,Ep,Eq: f(Ep,Eq,Er)*pnr(Er)

    Ep_Eq = lambda Ep,Eq: quad(Ep_Eqdist, np.amax([Erec(Ep,Eq)-width(Ep,Eq),0]), Erec(Ep,Eq)+width(Ep,Eq),limit=100,args=(Ep,Eq,))[0]

    return Ep_Eq

def YEr_v2_2D(sigp,sigq,V,eps,F=0.0001,ynr=lambda x: 0.16*x**0.18):
    #F=5.0
    Eqbar = lambda Er: ynr(Er)*Er
    Et = lambda Er: (1+(V/(eps*1000))*ynr(Er))*Er
    Ensig = lambda Er: np.sqrt(F*Eqbar(Er)/eps)
    
    Npqn = lambda Er: (1/np.sqrt(2*np.pi*Ensig(Er)**2))*(1/np.sqrt(2*np.pi*sigq(Eqbar(Er))**2)) \
    *(1/np.sqrt(2*np.pi*sigp(Et(Er))**2))
    
   
    #print(Npqn(10))
    #print(eps*Ensig(10))
    #print(sigp(Et(10)))
    #print(sigq(Eqbar(10)))
    Y_ErMeas_4D = lambda dQ,Y,Etr,Er: Npqn(Er)*(np.abs(Etr)/eps) \
    *np.exp(-(Etr-Er+(V/(1000*eps))*dQ)**2/(2*sigp(Et(Er))**2)) \
    *np.exp(-(dQ)**2/(2*sigq(Eqbar(Er))**2)) \
    *np.exp(-((ynr(Er)*Er/eps)-(Y*Etr/eps)+(dQ/eps))**2/(2*Ensig(Er)**2))
    
    #print(Y_ErMeas_4D(0,0.3,40,40))
   
    #@Memoize 
    Y_ErMeas = lambda Y,Etr,Er: quad(Y_ErMeas_4D,-np.inf,np.inf,args=(Y,Etr,Er,))[0]

    return Y_ErMeas

def YEr_v2_2D_fast(sigp,sigq,V,eps,F=0.0001,ynr=lambda x: 0.16*x**0.18):
    #F=5.0
    Eqbar = lambda Er: ynr(Er)*Er
    Et = lambda Er: (1+(V/(eps*1000))*ynr(Er))*Er
    Ensig = lambda Er: np.sqrt(F*(Eqbar(Er)/eps+1)) #add one pair to avoid divide-by-zero errors
    
    Npqn = lambda Er: (1/np.sqrt(2*np.pi*Ensig(Er)**2))*(1/np.sqrt(2*np.pi*sigq(Eqbar(Er))**2)) \
    *(1/np.sqrt(2*np.pi*sigp(Et(Er))**2))
   
    C = lambda Y,Etr,Er: Npqn(Er)*(np.abs(Etr)/eps) \
    *np.exp(-(Etr-Er)**2/(2*sigp(Et(Er))**2)) \
    *np.exp(-((ynr(Er)*Er/eps)-(Y*Etr/eps))**2/(2*Ensig(Er)**2))

    C0 = lambda Y,Etr,Er: Npqn(Er)*(np.abs(Etr)/eps) 

    Cexp = lambda Y,Etr,Er: -(Etr-Er)**2/(2*sigp(Et(Er))**2) -((ynr(Er)*Er/eps)-(Y*Etr/eps))**2/(2*Ensig(Er)**2)
    Cexp2 = lambda Y,Etr,Er: -((ynr(Er)*Er/eps)-(Y*Etr/eps))**2

    a = lambda Y,Etr,Er: (2*(V/(1000*eps))*(Etr-Er))/(2*sigp(Et(Er))**2)+(2*(ynr(Er)*Er-Y*Etr))/(2*eps**2*Ensig(Er)**2)

    b = lambda Y,Etr,Er: ((V/(1000*eps))**2/(2*sigp(Et(Er))**2) + 1/(2*sigq(Eqbar(Er))**2) + 1/(2*eps**2*Ensig(Er)**2))

    ABexp = lambda Y,Etr,Er: a(Y,Etr,Er)**2/(4*b(Y,Etr,Er))
  


    #return lambda Y,Etr,Er: C(Y,Etr,Er)*np.exp(a(Y,Etr,Er)**2/(4*b(Y,Etr,Er)))*np.sqrt(np.pi)*(1/np.sqrt(b(Y,Etr,Er))) 
    return lambda Y,Etr,Er: C0(Y,Etr,Er)*np.exp(Cexp(Y,Etr,Er)+ABexp(Y,Etr,Er))*np.sqrt(np.pi)*(1/np.sqrt(b(Y,Etr,Er))) 

def EpEq_v2_2D_fast(sigp,sigq,V,eps,F=0.0001,ynr=lambda x: 0.16*x**0.18):
    #F=5.0
    Eqbar = lambda Er: ynr(Er)*Er
    #Et = lambda Er: (1+(V/(eps*1000))*ynr(Er))*Er 
    #note that in this function all resolutions are assumed to be in Eee
    Et = Eqbar 
    Ensig = lambda Er: np.sqrt(F*(Eqbar(Er)/eps+1)) #add one pair to avoid divide-by-zero errors
    
    Npqn = lambda Er: (1/np.sqrt(2*np.pi*Ensig(Er)**2))*(1/np.sqrt(2*np.pi*sigq(Eqbar(Er))**2)) \
    *(1/np.sqrt(2*np.pi*sigp(Et(Er))**2))
   
    C = lambda Ep,Eq,Er: Npqn(Er) \
    *np.exp(-(Eq)**2/(2*sigq(Eqbar(Er))**2)) \
    *np.exp(-(Ep-Er)**2/(2*sigp(Et(Er))**2)) \
    *np.exp(-((ynr(Er)*Er/eps))**2/(2*Ensig(Er)**2))

    C0 = lambda Ep,Eq,Er: Npqn(Er) 

    Cexp = lambda Ep,Eq,Er: -Eq**2/(2*sigq(Eqbar(Er))**2) -(Ep-Er)**2/(2*sigp(Et(Er))**2) -((ynr(Er)*Er/eps))**2/(2*Ensig(Er)**2)

    a = lambda Ep,Eq,Er: (2*(V/1000)*(Ep-Er))/(2*sigp(Et(Er))**2)+(2*(ynr(Er)*Er)/eps)/(2*Ensig(Er)**2)+2*eps*Eq/(2*sigq(Eqbar(Er))**2)

    b = lambda Ep,Eq,Er: (V/1000)**2/(2*sigp(Et(Er))**2) + eps**2/(2*sigq(Eqbar(Er))**2) + 1/(2*Ensig(Er)**2)

    ABexp = lambda Ep,Eq,Er: a(Ep,Eq,Er)**2/(4*b(Ep,Eq,Er))
    #print(ABexp(6.3,1.1,5))
    #print(Cexp(6.3,1.1,5))
  

    return lambda Ep,Eq,Er: C0(Ep,Eq,Er)*np.exp(Cexp(Ep,Eq,Er)+ABexp(Ep,Eq,Er))*np.sqrt(np.pi)*(1/np.sqrt(b(Ep,Eq,Er)))*(1/2)*(erf(a(Ep,Eq,Er)/(2*np.sqrt(b(Ep,Eq,Er))))+1) 

def QEr_v2_2D_fast(sigh,sigi,V,eps,F=0.0001,Qbar=lambda x: 0.16*x**0.18):
   
    #new resolution functions 
    Ehee = lambda Er: ((1+(V/(1000*eps))*Qbar(Er))*Er)/(1+(V/(1000*eps)))
    EIee = lambda Er: Qbar(Er)*Er
    EIbar = lambda Er: Qbar(Er)*Er
    Ensig = lambda Er: np.sqrt(F*(EIbar(Er)/eps+1)) #add one pair to avoid divide-by-zero errors
    

    sigh_Er = lambda Er: sigh(Ehee(Er))
    sigi_Er = lambda Er: sigi(EIee(Er))
    sigp_Er = lambda Er: (1+(V/(1000*eps)))*sigh_Er(Er)

    Nihn = lambda Er: (1/np.sqrt(2*np.pi*Ensig(Er)**2))*(1/np.sqrt(2*np.pi*sigi_Er(Er)**2)) \
    *(1/np.sqrt(2*np.pi*sigh_Er(Er)**2))
   
    C = lambda Q,Etr,Er: Nihn(Er)*(np.abs(Etr)/eps)*(1/(1+(V/(1000*eps)))) \
    *np.exp(-(Etr-Er)**2/(2*sigp_Er(Er)**2)) \
    *np.exp(-((Qbar(Er)*Er/eps)-(Q*Etr/eps))**2/(2*Ensig(Er)**2))

    C0 = lambda Q,Etr,Er: Nihn(Er)*(np.abs(Etr)/eps)*(1/(1+(V/(1000*eps))))

    Cexp = lambda Q,Etr,Er: -(Etr-Er)**2/(2*sigp_Er(Er)**2) -((Qbar(Er)*Er/eps)-(Q*Etr/eps))**2/(2*Ensig(Er)**2)
    Cexp2 = lambda Q,Etr,Er:  -((Qbar(Er)*Er/eps)-(Q*Etr/eps))**2

    a = lambda Q,Etr,Er: (2*(V/(1000*eps))*(Etr-Er))/(2*sigp_Er(Er)**2)+(2*(Qbar(Er)*Er-Q*Etr))/(2*eps**2*Ensig(Er)**2)

    b = lambda Q,Etr,Er: ((V/(1000*eps))**2/(2*sigp_Er(Er)**2) + 1/(2*sigi_Er(Er)**2) + 1/(2*eps**2*Ensig(Er)**2))

    ABexp = lambda Q,Etr,Er: a(Q,Etr,Er)**2/(4*b(Q,Etr,Er))
  

    #return lambda Y,Etr,Er: C(Y,Etr,Er)*np.exp(a(Y,Etr,Er)**2/(4*b(Y,Etr,Er)))*np.sqrt(np.pi)*(1/np.sqrt(b(Y,Etr,Er))) 
    return lambda Q,Etr,Er: C0(Q,Etr,Er)*np.exp(Cexp(Q,Etr,Er)+ABexp(Q,Etr,Er))*np.sqrt(np.pi)*(1/np.sqrt(b(Q,Etr,Er))) 

def sigroot(F,Er):

    ptres = rfr.getRFunc('/home/phys/villaa/analysis/misc/nrFano_Constraint/data/jardin_ptres.txt')
    qres = rfr.getRFunc('/home/phys/villaa/analysis/misc/nrFano_Constraint/data/jardin_qsummaxres.txt')
    sigp = rfr.makeRFunc(ptres[1]['sqrt'])
    sigq = rfr.makeRFunc(qres[1]['lin'],True)

    #f0 = YEr_v2_2D_fast(sigp,sigq,4,(3.3/1000),0.0001)
    fF = YEr_v2_2D_fast(sigp,sigq,4,(3.3/1000),F)

    #g0 = YErSpec_v2_2D(f0)
    gF = YErSpec_v2_2D(fF)

    ynr = lambda x: 0.16*x**0.18

    #norm0 = lambda Er: quad(g0,-0.1,1,limit=100,args=(Er,))[0]
    #inty0 = lambda a,Er: quad(g0,ynr(Er)-a,ynr(Er)+a,limit=100,args=(Er,))[0]/norm0(Er)

    normF = lambda Er: quad(gF,-0.1,1,limit=100,args=(Er,))[0]
    intyF = lambda a,Er: quad(gF,ynr(Er)-a,ynr(Er)+a,limit=100,args=(Er,))[0]/normF(Er)


    #minsig0 = lambda a,Er: inty0(a,Er) - 0.6827 #one sigma
    #root0 = so.brentq(minsig0,0,1,rtol=0.001,maxiter=100,args=(Er,))
    minsigF = lambda a,Er: intyF(a,Er) - 0.6827 #one sigma
    rootF = so.brentq(minsigF,0,1,rtol=0.001,maxiter=100,args=(Er,))

    return rootF 

#set the Edelweiss sigma definition to default to NR band
def sigrootEdw(F,Er,V,eps,alpha=(1/100),Qbar=lambda x: 0.16*x**0.18,aH=0.0381):

    #fh2 = er.get_heatRes_func(0.4, 2.7,0.035)
    FWHM_to_SIG = 1 / (2*np.sqrt(2*np.log(2)))
    fh2 = er.get_heatRes_func(1.3, 3.5,aH*FWHM_to_SIG)
    heatRes_GGA3 = lambda x:fh2(x)

    fi2 = er.get_ionRes_func(1.3, 1.3, 2.8)
    sigI_GGA3 = lambda x:fi2(x)

    #new resolution functions 
    Ehee = lambda Er: ((1+(V/(1000*eps))*Qbar(Er))*Er)/(1+(V/(1000*eps)))
    EIee = lambda Er: Qbar(Er)*Er

    heatRes_GGA3_Er = lambda Er: heatRes_GGA3(Ehee(Er))

    sigI_GGA3_Er = lambda Er: sigI_GGA3(EIee(Er))

    #f0 = YEr_v2_2D_fast(sigp,sigq,4,(3.3/1000),0.0001)
    fF = QEr_v2_2D_fast(heatRes_GGA3,sigI_GGA3,V,eps,F,Qbar)

    #g0 = YErSpec_v2_2D(f0)
    #crude check for ER band
    if Qbar(10)>0.8:
      gF = expband_2D(fF,alpha,3)
    else:
      gF = expband_2D(fF,alpha,1.5)

    norm = quad(gF,-1,4,args=(Er,))[0]
    #print(norm)

    Qdist = lambda Q: (1/norm)*gF(Q,Er)
    #print(Qdist(1))

    intyF = lambda a: quad(Qdist,Qbar(Er)-a,Qbar(Er)+a,limit=100)[0]


    #minsig0 = lambda a,Er: inty0(a,Er) - 0.6827 #one sigma
    #root0 = so.brentq(minsig0,0,1,rtol=0.001,maxiter=100,args=(Er,))
    minsigF = lambda a: intyF(a) - 0.6827 #one sigma
    rootF = so.brentq(minsigF,0,1,rtol=0.001,maxiter=100)

    return rootF 

#set the Edelweiss sigma (second central moment) definition to default to NR band
def sigmomEdw(Er,band='ER',label='GGA3',F=0.000001,V=4.0,aH=0.0381,alpha=(1/100), A=0.16, B=0.18, lowlim=-1, verbose=False):

    #get the resolutions
    sigHv,sigIv,sigQerv,sigH_NRv,sigI_NRv,sigQnrv = \
    er.getEdw_det_res(label,V,'data/edw_res_data.txt',aH,C=None, A=A, B=B)


    #energy constant
    eps=3.0/1000.0 #Edelweiss used 3 in the early publication


    #g0 = YErSpec_v2_2D(f0)
    #crude check for ER band
    if band is 'ER':
      fF = QEr_v2_2D_fast(sigHv,sigIv,V,eps,F,Qbar=lambda x: 1)
      gF = expband_2D(fF,alpha,3)
      mean = 1
    else:
      fF = QEr_v2_2D_fast(sigHv,sigIv,V,eps,F,Qbar=lambda x: A*x**B)
      gF = expband_2D(fF,alpha,1.5)
      mean = A*Er**B

    norm = quad(gF,lowlim,4,args=(Er,))[0]
    #norm10 = 10.32813952

    #norm = norm10*(np.exp(-alpha*Er)/np.exp(-alpha*10))
    if(verbose):
      print('Normalization constant: {}'.format(norm))


    Qdist = lambda Q: (1/norm)*gF(Q,Er)
    if(verbose):
      print('f(Q=1): {}'.format(Qdist(1)))

    #get the mean 
    #NOTE: actually I found the calculation of variance from the definition (NOT 68% containment)
    #to be VERY sensitive to the mean, for ER band if I use a mean of 1 it was off by a factor
    #of two compared to the 68% region. But if I used the true mean as calculated above (which was
    #generally less than 1% off from 1--it came back to being in line
    #I guess this is because <x^2> and <x>^2 are both "large" and you have to subtract them to get
    #the answer--this amplifies numerical uncertainties.
    meanfun = lambda Q: Q*Qdist(Q)
    mean = quad(meanfun,lowlim,4)[0]
    if(verbose):
      print('Mean: {}'.format(mean))

    #intyF = lambda a: quad(Qdist,mean-a,mean+a,limit=100)[0]


    #minsigF = lambda a: intyF(a) - 0.6827 #one sigma
    #rootF = so.brentq(minsigF,0,1,rtol=0.001,maxiter=100)

    #by integration
    sigfun = lambda Q: Q**2*Qdist(Q)
    q2 = quad(sigfun,lowlim,4)[0]
    if(verbose):
      print('<q^2>: {}'.format(q2))

    #print(sigQerv(Er))
    #return norm,rootF,(np.sqrt(q2-mean**2)) 
    return (np.sqrt(q2-mean**2)) 

#analytical distributions for QEr
def analytical_NRQ_dist(Q,Er=10.0,F=0.0,V=4.0,aH=0.0381,alpha=(1/18.0),A=0.16,B=0.18,label='GGA3'):
  """
  This function is based on an approximation that is no better than the Edelweiss approximation.
  Do not use.
  """
  
  eps = 3.0
  #get the resolutions
  sigHv,sigIv,sigQerv,sigH_NRv,sigI_NRv,sigQnrv = \
  er.getEdw_det_res(label,V,'data/edw_res_data.txt',aH,C=None, A=A, B=B)

  #calculate basic variables
  scale = (V/eps)
  sa = 2*((1+scale)*sigH_NRv(Er))**2
  sb = 2*sigI_NRv(Er)**2
  sc = 2*(eps/1000.0)*F*Er*(A*Er**B) #careful with units, eps units need to be converted to pair/keV here

  print('sa: {}'.format(sa))
  print('sb: {}'.format(sb))
  print('sc: {}'.format(sc))

  #denomonators from N-MISC-19-001 pg 59,60
  denom_abc = 4*(sb + sc + 2*A*Er**B*scale*sb + A**2*Er**(2*B)*(sa+scale**2*sb))
  denom_de = 2*(sa*(sb+sc) + scale**2*sb*sc)*(np.sqrt((sb + sc + 2*A*Er**B*scale*sb + A**2*Er**(2*B)*(sa+scale**2*sb))/ \
       (sa*(sb+sc) + scale**2*sb*sc)))

  #now get the distribution constants
  inv_aq = 2*np.sqrt(np.pi)*np.sqrt(sa*sb)*(1+scale)*np.sqrt((sc/sb) + 1 + scale**2*(sc/sa)) \
      *np.sqrt((sb + sc + 2*A*Er**B*scale*sb + A**2*Er**(2*B)*(sa+scale**2*sb))/(sa*(sb+sc) + scale**2*sb*sc))*alpha*(eps/1000.0)
  aq = np.abs(Er)*(1/inv_aq)*np.exp(((sa*(sb+sc)+scale**2*sb*sc)*alpha**2 - 4*A**2*Er**(2+2*B) - 4*Er*(sb+sc)*alpha -4*A*Er**(1+B)*scale*sb*alpha)/denom_abc)
  bq = (8*Er*A*Er**(1+B) - 4*Er*scale*sb*alpha - 4*A*Er**(1+B)*sa*alpha - 4*A*Er**(1+B)*scale**2*sb*alpha)/denom_abc
  cq = 4*Er**2/denom_abc
  dq = (2*Er*(sb+sc) + 2*A*Er**(1+B)*scale*sb - sa*(sb+sc)*alpha - scale**2*sb*sc*alpha)/denom_de
  eq = (2*Er*scale*sb + 2*A*Er**(1+B)*(sa + scale**2*sb))/denom_de

  #transformation variables
  atq = cq/eq**2
  btq = ((bq/eq) + (2*cq*dq/eq**2))
  ctq = ((bq*dq/eq) + (cq*dq**2/eq**2))
  alpha_q = np.sqrt(atq)
  beta_q = - (btq/(2*np.sqrt(atq)))
  inv_norm = np.exp((btq**2/(4*atq)) - ctq)*np.sqrt(np.pi)*(1/alpha_q)*(1-erf(beta_q/(np.sqrt(alpha_q**2+1))))
  norm = eq/inv_norm/aq

  print('aq: {}'.format(aq))
  print('bq: {}'.format(bq))
  print('cq: {}'.format(cq))
  print('dq: {}'.format(dq))
  print('eq: {}'.format(eq))
  print('norm: {}'.format(norm))

  return aq*np.exp((bq-cq*Q)*Q)*(1+erf(dq+eq*Q))

def analytical_NRQ_mean(Er=10.0,F=0.0,V=4.0,aH=0.0381,alpha=(1/18.0),A=0.16,B=0.18,label='GGA3'):
  """
  This function is based on an approximation that is no better than the Edelweiss approximation.
  Do not use.
  """  
  eps = 3.0
  #get the resolutions
  sigHv,sigIv,sigQerv,sigH_NRv,sigI_NRv,sigQnrv = \
  er.getEdw_det_res(label,V,'data/edw_res_data.txt',aH,C=None, A=A, B=B)

  #calculate basic variables
  scale = (V/eps)
  sa = 2*((1+scale)*sigH_NRv(Er))**2
  sb = 2*sigI_NRv(Er)**2
  sc = 2*(eps/1000.0)*F*Er*(A*Er**B) #careful with units, eps units need to be converted to pair/keV here

  print('sa: {}'.format(sa))
  print('sb: {}'.format(sb))
  print('sc: {}'.format(sc))

  #denomonators from N-MISC-19-001 pg 59,60
  denom_abc = 4*(sb + sc + 2*A*Er**B*scale*sb + A**2*Er**(2*B)*(sa+scale**2*sb))
  denom_de = 2*(sa*(sb+sc) + scale**2*sb*sc)*(np.sqrt((sb + sc + 2*A*Er**B*scale*sb + A**2*Er**(2*B)*(sa+scale**2*sb))/ \
       (sa*(sb+sc) + scale**2*sb*sc)))

  #now get the distribution constants
  inv_aq = 2*np.sqrt(np.pi)*np.sqrt(sa*sb)*(1+scale)*np.sqrt((sc/sb) + 1 + scale**2*(sc/sa)) \
      *np.sqrt((sb + sc + 2*A*Er**B*scale*sb + A**2*Er**(2*B)*(sa+scale**2*sb))/(sa*(sb+sc) + scale**2*sb*sc))*alpha*(eps/1000.0)
  aq = np.abs(Er)*(1/inv_aq)*np.exp(((sa*(sb+sc)+scale**2*sb*sc)*alpha**2 - 4*A**2*Er**(2+2*B) - 4*Er*(sb+sc)*alpha -4*A*Er**(1+B)*scale*sb*alpha)/denom_abc)
  bq = (8*Er*A*Er**(1+B) - 4*Er*scale*sb*alpha - 4*A*Er**(1+B)*sa*alpha - 4*A*Er**(1+B)*scale**2*sb*alpha)/denom_abc
  cq = 4*Er**2/denom_abc
  dq = (2*Er*(sb+sc) + 2*A*Er**(1+B)*scale*sb - sa*(sb+sc)*alpha - scale**2*sb*sc*alpha)/denom_de
  eq = (2*Er*scale*sb + 2*A*Er**(1+B)*(sa + scale**2*sb))/denom_de

  #transformation variables
  atq = cq/eq**2
  btq = ((bq/eq) + (2*cq*dq/eq**2))
  ctq = ((bq*dq/eq) + (cq*dq**2/eq**2))
  alpha_q = np.sqrt(atq)
  beta_q = - (btq/(2*np.sqrt(atq)))
  inv_norm = np.exp((btq**2/(4*atq)) - ctq)*np.sqrt(np.pi)*(1/alpha_q)*(1-erf(beta_q/(np.sqrt(alpha_q**2+1))))
  norm = eq/inv_norm/aq

  print('aq: {}'.format(aq))
  print('bq: {}'.format(bq))
  print('cq: {}'.format(cq))
  print('dq: {}'.format(dq))
  print('eq: {}'.format(eq))
  print('norm: {}'.format(norm))
  print('beta_q: {}'.format(beta_q))
  print('sqrt(alpha_q^2+1): {}'.format(np.sqrt(alpha_q**2+1)))
  print('exp: {}'.format(np.exp(-(beta_q/np.sqrt(alpha_q**2+1))**2)))

  #calculate the mean see pg. 66 of N-MISC-19-001
  #term 1 of dK/dt evaluated at t=0
  T1 = ((btq/(2*atq*eq))-(dq/eq))
  #term 2 evaluated at t=0
  T2 = ((1/(1-erf(beta_q/(np.sqrt(alpha_q**2+1)))))*(2/np.sqrt(np.pi))*(1/np.sqrt(alpha_q**2+1))*(1/(2*np.sqrt(atq)*eq))*np.exp(-(beta_q/np.sqrt(alpha_q**2+1))**2))

  print('T1: {}'.format(T1))
  print('T2: {}'.format(T2))
  print('Qbar: {}'.format(A*Er**B))

  return T1+T2 

def analytical_NRQ_var(Er=10.0,F=0.0,V=4.0,aH=0.0381,alpha=(1/18.0),A=0.16,B=0.18,label='GGA3'):
  """
  This function is based on an approximation that is no better than the Edelweiss approximation.
  Do not use.
  """
  
  eps = 3.0
  #get the resolutions
  sigHv,sigIv,sigQerv,sigH_NRv,sigI_NRv,sigQnrv = \
  er.getEdw_det_res(label,V,'data/edw_res_data.txt',aH,C=None, A=A, B=B)

  #calculate basic variables
  scale = (V/eps)
  sa = 2*((1+scale)*sigH_NRv(Er))**2
  sb = 2*sigI_NRv(Er)**2
  sc = 2*(eps/1000.0)*F*Er*(A*Er**B) #careful with units, eps units need to be converted to pair/keV here

  #print('sa: {}'.format(sa))
  #print('sb: {}'.format(sb))
  #print('sc: {}'.format(sc))

  #denomonators from N-MISC-19-001 pg 59,60
  denom_abc = 4*(sb + sc + 2*A*Er**B*scale*sb + A**2*Er**(2*B)*(sa+scale**2*sb))
  denom_de = 2*(sa*(sb+sc) + scale**2*sb*sc)*(np.sqrt((sb + sc + 2*A*Er**B*scale*sb + A**2*Er**(2*B)*(sa+scale**2*sb))/ \
       (sa*(sb+sc) + scale**2*sb*sc)))

  #now get the distribution constants
  inv_aq = 2*np.sqrt(np.pi)*np.sqrt(sa*sb)*(1+scale)*np.sqrt((sc/sb) + 1 + scale**2*(sc/sa)) \
      *np.sqrt((sb + sc + 2*A*Er**B*scale*sb + A**2*Er**(2*B)*(sa+scale**2*sb))/(sa*(sb+sc) + scale**2*sb*sc))*alpha*(eps/1000.0)
  aq = np.abs(Er)*(1/inv_aq)*np.exp(((sa*(sb+sc)+scale**2*sb*sc)*alpha**2 - 4*A**2*Er**(2+2*B) - 4*Er*(sb+sc)*alpha -4*A*Er**(1+B)*scale*sb*alpha)/denom_abc)
  bq = (8*Er*A*Er**(1+B) - 4*Er*scale*sb*alpha - 4*A*Er**(1+B)*sa*alpha - 4*A*Er**(1+B)*scale**2*sb*alpha)/denom_abc
  cq = 4*Er**2/denom_abc
  dq = (2*Er*(sb+sc) + 2*A*Er**(1+B)*scale*sb - sa*(sb+sc)*alpha - scale**2*sb*sc*alpha)/denom_de
  eq = (2*Er*scale*sb + 2*A*Er**(1+B)*(sa + scale**2*sb))/denom_de

  #transformation variables
  atq = cq/eq**2
  btq = ((bq/eq) + (2*cq*dq/eq**2))
  ctq = ((bq*dq/eq) + (cq*dq**2/eq**2))
  alpha_q = np.sqrt(atq)
  beta_q = - (btq/(2*np.sqrt(atq)))
  inv_norm = np.exp((btq**2/(4*atq)) - ctq)*np.sqrt(np.pi)*(1/alpha_q)*(1-erf(beta_q/(np.sqrt(alpha_q**2+1))))
  norm = eq/inv_norm/aq

  #print('aq: {}'.format(aq))
  #print('bq: {}'.format(bq))
  #print('cq: {}'.format(cq))
  #print('dq: {}'.format(dq))
  #print('eq: {}'.format(eq))
  #print('norm: {}'.format(norm))
  #print('beta_q: {}'.format(beta_q))
  #print('sqrt(alpha_q^2+1): {}'.format(np.sqrt(alpha_q**2+1)))
  #print('exp: {}'.format(np.exp(-(beta_q/np.sqrt(alpha_q**2+1))**2)))

  #calculate the mean see pg. 67 of N-MISC-19-001
  #term 1 of d^2K/dt^2 evaluated at t=0
  T1 = (1/(2*atq*eq**2)) 
  #term 2 evaluated at t=0
  T2 = -((1/(1-erf(beta_q/(np.sqrt(alpha_q**2+1)))))*(2/np.sqrt(np.pi))*(1/np.sqrt(alpha_q**2+1))*(1/(2*np.sqrt(atq)*eq))**2 \
      *((2*beta_q)/np.sqrt(alpha_q**2+1))*np.exp(-(beta_q/np.sqrt(alpha_q**2+1))**2))
  #term 3 evaluated at t=0
  T3 = -((1/(1-erf(beta_q/(np.sqrt(alpha_q**2+1)))))*(2/np.sqrt(np.pi))*(1/np.sqrt(alpha_q**2+1))*(1/(2*np.sqrt(atq)*eq))*np.exp(-(beta_q/np.sqrt(alpha_q**2+1))**2))**2

  #print('T1: {}'.format(T1))
  #print('T2: {}'.format(T2))
  #print('T3: {}'.format(T3))
  #print('sigNR: {}'.format(sigQnrv(Er)))
  #print('sigER: {}'.format(sigQerv(Er)))

  return T1+T2+T3

def series_NRQ_var(Er=10.0,F=0.0,V=4.0,aH=0.0381,alpha=(1/18.0),A=0.16,B=0.18,label='GGA3'):
  
  eps = 3.0
  #get the resolutions
  sigHv,sigIv,sigQerv,sigH_NRv,sigI_NRv,sigQnrv = \
  er.getEdw_det_res(label,V,'data/edw_res_data.txt',aH,C=None, A=A, B=B)

  #calculate basic variables
  scale = (V/eps)
  qbar = A*Er**B
  chi = (1+scale)
  omega = scale

  #Get Edw terms
  TEdw = (1/Er**2)*(eps*1e-3*qbar*Er*F + chi**2*qbar**2*sigH_NRv(Er)**2 + (1+omega*qbar)**2*sigI_NRv(Er)**2)

  #Get qbar cross-terms
  T1 = (1/Er**2)*(qbar**2*chi**2*sigH_NRv(Er)**2 + (qbar**2*omega**2+2*qbar*omega)*sigI_NRv(Er)**2)

  #Get mixed second derivatives
  T2 = (1/Er**4)*(4*qbar**2*chi**2*omega**2+chi**2+2*qbar*chi**2*omega)*sigH_NRv(Er)**2*sigI_NRv(Er)**2

  #Get pure second derivatives
  T3 = (1/Er**4)*(1/2)*(4*qbar**2*chi**4*sigH_NRv(Er)**4 + (4*qbar**2*omega**4+8*qbar*omega**3+4*omega**2)*sigI_NRv(Er)**4)

  #third derivatives have a couple terms that are of the same order in 1/Er (mixed 3rd derivatives)
  T4 = (1/Er**4)*((1+qbar*omega)*(2*chi**2+6*qbar*chi**2*omega) + (qbar*chi)*(4*omega*chi+6*qbar*omega**2*chi))*sigH_NRv(Er)**2*sigI_NRv(Er)**2

  #third derivatives have a couple more terms that are of the same order in 1/Er (pure 3rd derivatives) 
  T5 = (1/Er**4)*((1+qbar*omega)*(6*omega**2+6*qbar*omega**3)*sigI_NRv(Er)**4 + (qbar*chi)*(6*chi**3*qbar)*sigH_NRv(Er)**4)

  #fourth derivatives have next order in 1/Er (pure 4th derivatives) 
  T6 = (1/Er**6)*(15/24.0)*((1+qbar*omega)*omega*(24*omega**3+24*qbar*omega**4)*sigI_NRv(Er)**6 + (2*qbar*chi**2)*(24*chi**4*qbar)*sigH_NRv(Er)**6)

  #print('TEdw: {}'.format(TEdw))
  #print('sqrt(TEdw): {}'.format(np.sqrt(TEdw)))
  #print('T1: {}'.format(T1))
  #print('Ta: {}'.format((1/Er**2)*(1/2)*qbar**2*chi**2*sigH_NRv(Er)**2)) 
  #print('Tb: {}'.format((1/Er**2)*(1/2)*(qbar**2*omega**2)*sigI_NRv(Er)**2)) 
  #print('Tc: {}'.format((1/Er**2)*(1/2)*(2*qbar*omega)*sigI_NRv(Er)**2)) 
  #print('T2: {}'.format(T2))
  #print('T3: {}'.format(T3))
  #print('T4: {}'.format(T4))
  #print('T5: {}'.format(T5))
  #print('T6: {}'.format(T6))

  #return TEdw+T2+T3 
  return TEdw+(T2+T3+T4+T5+T6)

#let's write a corrected version of this function
def series_NRQ_var_corr1(Er=10.0,F=0.0,V=4.0,aH=0.0381,alpha=(1/18.0),A=0.16,B=0.18,label='GGA3',corr1file='data/sigdiff_test.h5'):

    #set up return value so far
    sigr = np.sqrt(series_NRQ_var(Er=Er,F=F,V=V,aH=aH,alpha=alpha,A=A,B=B)) \
      + series_NRQ_sig_c1(Er=Er,F=F,V=V,aH=aH,A=A,B=B,alpha=alpha,label=label) \

    return sigr**2
def series_NRQ_var_corr2(Er=10.0,F=0.0,V=4.0,aH=0.0381,alpha=(1/18.0),A=0.16,B=0.18,label='GGA3',corr1file='data/sigdiff_test.h5',verbose=False):

    #set up return value so far
    sigr = np.sqrt(series_NRQ_var(Er=Er,F=F,V=V,aH=aH,alpha=alpha,A=A,B=B)) \
      + series_NRQ_sig_c1(Er=Er,F=F,V=V,aH=aH,A=A,B=B,alpha=alpha,label=label) \
      + series_NRQ_sig_c2(Er=Er,F=F,V=V,aH=aH,A=A,B=B,alpha=alpha,label=label,verbose=verbose) 

    return sigr**2
def series_NRQ_var_corr(Er=10.0,F=0.0,V=4.0,aH=0.0381,alpha=(1/18.0),A=0.16,B=0.18,label='GGA3',corr1file='data/sigdiff_test.h5'):

    #set up return value so far
    sigr = np.sqrt(series_NRQ_var(Er=Er,F=F,V=V,aH=aH,alpha=alpha,A=A,B=B)) \
      + series_NRQ_sig_c1(Er=Er,F=F,V=V,aH=aH,A=A,B=B,alpha=alpha,label=label) \
      + series_NRQ_sig_c2(Er=Er,F=F,V=V,aH=aH,A=A,B=B,alpha=alpha,label=label) \
      + series_NRQ_sig_c3(Er=Er,F=F,V=V,aH=aH,A=A,B=B,alpha=alpha,label=label,corr1file=corr1file) 

    return sigr**2

def series_NRQ_sig_c1(Er=10.0,F=0.0,V=4.0,aH=0.0381,alpha=(1/18.0),A=0.16,B=0.18,label='GGA3'):

    #####First Correction
    #get sigQ for for nominal parameters
    #Enr,signr = fc.RWCalc(filename='data/res_calc.h5',alpha=1/18.0,aH=0.0381,band='NR')

    #spline those diffs
    #sig0 = inter.InterpolatedUnivariateSpline(Enr, signr , k=3)
    #sig_corr = sig0_glob(Er) - np.sqrt(series_NRQ_var(Er,V=4.0,F=0,aH=0.0381,A=0.16,B=0.18,alpha=(1/18.0)))
    #sig_corr = fc.sig0_glob(Er) - er.sigQnrv_glob(Er) 


    #8/1/20 update for best fit based on exact sigma Parameters: V = 4.0000 V; aH = 0.0381; A = 0.1537; B = 0.1703; scale = 0.9948
    #V=4.0,alpha=(1/18.0),aH=3.81134613e-02,A=1.53737587e-01,B=1.70327657e-01,scale=9.94778557e-01,maxEr=200)
    Vmod = 4.0*9.94778557e-1
    sig_corr = fc.sig0_glob(Er) - np.sqrt(series_NRQ_var(Er,V=Vmod,F=0,aH=3.81134613e-2,A=1.53737587e-1,B=1.70327657e-1,alpha=(1/18.0)))

    #set up return value so far
    #sigr = np.sqrt(series_NRQ_var(Er=Er,F=F,V=V,aH=aH,alpha=alpha,A=A,B=B)) + sig_corr 

    return sig_corr

def series_NRQ_sig_c2(Er=10.0,F=0.0,V=4.0,aH=0.0381,alpha=(1/18.0),A=0.16,B=0.18,label='GGA3',verbose=False):

    ######This Correction is technically only defined at the values of the E points: 24.5 keV, 34 keV, 44 keV, 58 keV, and 97 keV
    # it is based on a multi-linear regression to the differences from the exact computation
    #ER_data, NR_data = edu.getERNR()
    #NREr = np.asarray(NR_data['Erecoil'])
    #NREr = np.sort(NREr)

    #the stuff above is more general but it involves file I/O and may slow stuff down
    NREr = np.asarray([24.5012,34.2156,44.2627,58.4014,97.7172])

    #the 24.5 keV point should be the first element, so assume those
    didx = 0
    dist = np.abs(Er-NREr[didx])
    #coeffs in order: aH, scale, A, B
    coeff= np.asarray([0.00552431, 0.00133703, 0.01633244, 0.02117115])
    intercept= -0.007662520580006483

    for i,E in enumerate(NREr):
        newdist = np.abs(Er-E)
        if newdist<dist:
          dist = newdist
          didx = i

    
    switch_coeff={
            0:np.asarray([0.00552431, 0.00133703, 0.01633244, 0.02117115]),
            1:np.asarray([0.00891606, 0.00139528, 0.013981,   0.01819466]),
            2:np.asarray([0.01268575, 0.00128421, 0.01304974, 0.01676499]),
            3:np.asarray([0.0195214,  0.00114802, 0.01256059, 0.01603726]),
            4:np.asarray([0.02973745, 0.00088643, 0.01310807, 0.01668303])

            }
    switch_intercept={
            0:-0.007662520580006483,
            1:-0.006976876162577476,
            2:-0.006621463867453043,
            3:-0.00654459822082259,
            4:-0.006859726936860776

            }

    coeff = switch_coeff.get(didx,np.asarray([0.00552431, 0.00133703, 0.01633244, 0.02117115]))
    intercept = switch_intercept.get(didx,-0.007662520580006483)

    if verbose:
      print(coeff)
      print(intercept)

    corr2 = intercept
    if verbose:
      print('intercept: {}'.format(intercept))
    corr2+= coeff[0]*aH
    if verbose:
      print('coef X X0 = {:01.7f} X {:01.7f} = {:01.7f}'.format(coeff[0],aH,coeff[0]*aH))
    corr2+= coeff[1]*(V/4.0) #assuming! our fit done with nominal voltage 4V
    if verbose:
      print('coef X X0 = {:01.7f} X {:01.7f} = {:01.7f}'.format(coeff[1],(V/4.0),(V/4.0)*coeff[1]))
    corr2+= coeff[2]*A
    if verbose:
      print('coef X X0 = {:01.7f} X {:01.7f} = {:01.7f}'.format(coeff[2],A,coeff[2]*A))
    corr2+= coeff[3]*B
    if verbose:
      print('coef X X0 = {:01.7f} X {:01.7f} = {:01.7f}'.format(coeff[3],B,coeff[3]*B))
    
    return corr2


def series_NRQ_sig_c3(Er=10.0,F=0.0,V=4.0,aH=0.0381,alpha=(1/18.0),A=0.16,B=0.18,label='GGA3',corr1file='data/sigdiff_test.h5'):

    ######Next Correction (can only do it if Er is near the Edw data values)
    f = h5py.File(corr1file,'r')
    E = np.zeros((0,))
    for i in f['NR/']:
      a = np.asarray([float(i)])
      E = np.concatenate((E,a))

    E = np.sort(E)
    #print(E)
    #f.close()
    #resNR_data = pd.read_csv("data/edelweiss_NRwidth_GGA3_data.txt", skiprows=1, \
    #              names=['E_recoil', 'sig_NR', 'E_recoil_err', 'sig_NR_err'], \
    #              delim_whitespace=True)

    #NR_data = {'Erecoil': resNR_data["E_recoil"][2::], 'sigma': resNR_data["sig_NR"][2::], 'sigma_err': resNR_data["sig_NR_err"][2::]}

    #E = np.sort(NR_data['Erecoil'])

    Enear = find_nearest(E,Er)

    if (np.abs(Enear-Er)/Er) < 0.01:
      #print('highly accurate')
      path='{}/{:3.1f}/'.format('NR',Enear)
      #f = h5py.File(corr1file,'r')
      output = np.asarray(f[path+'output'])
      pars = np.asarray(f[path+'pars'])
      corr_func = inter.NearestNDInterpolator(pars,output)
      return corr_func([A,B,aH,(V/4.0)])
    else:
      return 0.0

#helper function 
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

# The function below will compute the HPD interval. 
# The idea is that we rank-order the MCMC trace. 
# We know that the number of samples that are included in the HPD is 
# 0.95 times the total number of MCMC sample. We then consider all intervals 
# that contain that many samples and find the shortest one.
# From http://bebi103.caltech.edu.s3-website-us-east-1.amazonaws.com/2015/tutorials/l06_credible_regions.html
# pymc3 also has an hpd function defined 
def hpd(trace, mass_frac) :
    """
    Returns highest probability density region given by
    a set of samples.

    Parameters
    ----------
    trace : array
        1D array of MCMC samples for a single variable
    mass_frac : float with 0 < mass_frac <= 1
        The fraction of the probability to be included in
        the HPD.  For example, `massfrac` = 0.95 gives a
        95% HPD.
        
    Returns
    -------
    output : array, shape (2,)
        The bounds of the HPD
    """
    # Get sorted list
    d = np.sort(np.copy(trace))

    # Number of total samples taken
    n = len(trace)
    #print (n)
    
    # Get number of samples that should be included in HPD
    n_samples = np.floor(mass_frac * n).astype(int)
    #print (n_samples)
    
    # Get width (in units of data) of all intervals with n_samples samples
    int_width = d[n_samples:] - d[:n-n_samples]
    
    # Pick out minimal interval
    min_int = np.argmin(int_width)
    #print (min_int)
    
    # Return interval
    return np.array([d[min_int], d[min_int+n_samples]])

#map out the difference between our corrected series approx and the numerical integration in sigmomEdw
def diffmap(pars,Etest=25.0,outfile='data/diffmap.h5',diff0file='data/sig_diffs.h5'):
    """
    Returns and saves a structure holding computed differences
    between sigmomEdw and series approx.
    

    Parameters
    ----------
    pars : array
        (N,4) numpy array holding 4 parameters in each row:
        A,B,aH,mu (fractional voltage variation). 
    Etest : double 
        Energy test point in keV. 
        default: 25.0 keV 
    outfile : string 
        fullpath and filename of output file. 
        default: data/diffmap.h5' 
    diff0file : string 
        fullpath and filename of file storing zeroth order diffs. 
        default: data/diffmap.h5' 
        
    Returns
    -------
    output : array, shape (N,1)
        The differences for each row of params 
    """

    #if parameters not supplied correctly
    if np.shape(pars)[1] != 4:
      print('Too few columns, need 4 params')
      return np.zeros((0,1))

    output = np.zeros((np.shape(pars)[0],1))
    frac = np.zeros((np.shape(pars)[0],1))

    #get sigQ for for nominal parameters
    Enr,signr = fc.RWCalc(filename='data/res_calc.h5',alpha=1/18.0,aH=0.0381,band='NR')

    #spline those diffs
    sig0 = inter.InterpolatedUnivariateSpline(Enr, signr , k=3)
    sig_corr = inter.InterpolatedUnivariateSpline(Enr, sig0(Enr) - np.sqrt(series_NRQ_var(Enr,V=4.0,F=0,aH=0.0381,A=0.16,B=0.18,alpha=(1/18.0))), k=3)
    sig_corr_v = np.vectorize(sig_corr)
    print(sig_corr_v(150))

    for i in np.arange(np.shape(output)[0]):
      p = pars[i,:]
      start = time.time()
      sig_res = sigmomEdw(Etest,band='NR',label='GGA3',F=0.000001,V=p[3]*4.0,aH=p[2],alpha=(1/18.0),A=p[0],B=p[1])
      end = time.time()
      print('Normalization and Integration: {:1.5f} sec.'.format(end-start))
      sig_res_func = np.sqrt(series_NRQ_var(Etest,V=p[3]*4.0,F=0,aH=p[2],A=p[0],B=p[1],alpha=(1/18.0))) + sig_corr(Etest)

      output[i] = sig_res-sig_res_func
      frac[i] = (sig_res-sig_res_func)/sig_res


    #write output file
    path='{}/{:3.1f}/'.format('NR',Etest)
    
    f = h5py.File(outfile,'a')
    
    diffoutput = path+'output' in f
    difffrac = path+'frac' in f
    diffpars = path+'pars' in f
    
    
    if diffoutput:
      del f[path+'output']
    if difffrac:
      del f[path+'frac']
    if diffpars:
      del f[path+'pars']
    
    
    
    dset = f.create_dataset(path+'output',np.shape(output),dtype=np.dtype('float64').type, \
    compression="gzip",compression_opts=9)
    dset[...] = output 

    dset = f.create_dataset(path+'frac',np.shape(frac),dtype=np.dtype('float64').type, \
    compression="gzip",compression_opts=9)
    dset[...] = frac 
    
    dset = f.create_dataset(path+'pars',np.shape(pars),dtype=np.dtype('float64').type, \
    compression="gzip",compression_opts=9)
    dset[...] = pars 
    
    
    f.close() 

    return output,frac
