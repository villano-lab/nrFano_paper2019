import numpy as np
from scipy.special import erf
import math 
from scipy.integrate import quad
import resfuncRead as rfr
import scipy.optimize as so
import EdwRes as er




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
def sigmomEdw(Er,band='ER',label='GGA3',F=0.000001,V=4.0,aH=0.0381,alpha=(1/100)):

    #get the resolutions
    sigHv,sigIv,sigQerv,sigH_NRv,sigI_NRv,sigQnrv = \
    er.getEdw_det_res(label,V,'data/edw_res_data.txt',aH,C=None)


    #energy constant
    eps=3.0/1000.0 #Edelweiss used 3 in the early publication


    #g0 = YErSpec_v2_2D(f0)
    #crude check for ER band
    if band is 'ER':
      fF = QEr_v2_2D_fast(sigHv,sigIv,V,eps,F,Qbar=lambda x: 1)
      gF = expband_2D(fF,alpha,3)
      mean = 1
    else:
      fF = QEr_v2_2D_fast(sigHv,sigIv,V,eps,F,Qbar=lambda x: 0.16*x**0.18)
      gF = expband_2D(fF,alpha,1.5)
      mean = 0.16*Er**0.18

    norm = quad(gF,-1,4,args=(Er,))[0]
    #print(norm)


    Qdist = lambda Q: (1/norm)*gF(Q,Er)
    #print(Qdist(1))

    #get the mean 
    #NOTE: actually I found the calculation of variance from the definition (NOT 68% containment)
    #to be VERY sensitive to the mean, for ER band if I use a mean of 1 it was off by a factor
    #of two compared to the 68% region. But if I used the true mean as calculated above (which was
    #generally less than 1% off from 1--it came back to being in line
    #I guess this is because <x^2> and <x>^2 are both "large" and you have to subtract them to get
    #the answer--this amplifies numerical uncertainties.
    meanfun = lambda Q: Q*Qdist(Q)
    mean = quad(meanfun,-1,4)[0]
    #print(mean)

    intyF = lambda a: quad(Qdist,mean-a,mean+a,limit=100)[0]


    #minsigF = lambda a: intyF(a) - 0.6827 #one sigma
    #rootF = so.brentq(minsigF,0,1,rtol=0.001,maxiter=100)

    #by integration
    sigfun = lambda Q: Q**2*Qdist(Q)
    q2 = quad(sigfun,-1,4)[0]

    #print(sigQerv(Er))
    #return norm,rootF,(np.sqrt(q2-mean**2)) 
    return (np.sqrt(q2-mean**2)) 

