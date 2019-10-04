import numpy as np
import EdwRes as er
import fano_calc as fc
from scipy.integrate import quad
import scipy.interpolate as inter
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import h5py

def getBinSys(C,m,filename='data/res_calc.h5',bins = [5, 10, 20, 30, 40, 50, 70, 150],Qbar=lambda x: 0.16*x**0.18):

    bins = np.asarray(bins)
    Ebase,sigbase = fc.RWCalc(filename=filename,band='NR',alpha=(1/18.0),F=000)
    #remove vars
    f = h5py.File(filename,'r')

    f.close()

    Ec = np.zeros((np.shape(bins)[0]-1,))
    delta = np.zeros((np.shape(bins)[0]-1,))

    #calculate the binning correction
    sig_ss = inter.InterpolatedUnivariateSpline(Ebase, sigbase, k=3)
    sig_final = lambda Er: np.sqrt(sig_ss(Er)**2 + (C+m*Er)**2)

    alpha = (1/18.0)
    pnr = lambda Er: (1/alpha)*np.exp(-alpha*Er)

    pqe = lambda Q,Er: (1/np.sqrt(2*np.pi*sig_final(Er)**2))*np.exp(-(Q-Qbar(Er))**2/(2*sig_final(Er)**2))

    #integrate
    igrnd = lambda Er,Q: pqe(Q,Er)*pnr(Er)
    pq = lambda Q,El,Eh: quad(igrnd,El,Eh,limit=100,args=(Q,))[0]

    for i,Eloop in enumerate(bins):

      if i==np.shape(bins)[0]-1:
        break
        
      l = Eloop
      h = bins[i+1]
        
      #print('calculating binning systematic for bin El: {}; Eh: {}'.format(l,h))
      norm = quad(pq,-5,5,limit=100,args=(l,h,))[0]    
      pq_norm = lambda Q,El,Eh: pq(Q,El,Eh)/norm
      pq_mean = lambda Q,El,Eh: Q*pq_norm(Q,El,Eh)
      pq_std = lambda Q,El,Eh: Q**2*pq_norm(Q,El,Eh)
    
      a = quad(pq_norm,-5,5,limit=100,args=(l,h,))
      b = quad(pq_mean,-5,5,limit=100,args=(l,h,))
      c = quad(pq_std,-5,5,limit=100,args=(l,h,))
    
      Emid = (h+l)/2.0
      Ec[i] = Emid
      
      sigmid = sig_final(Emid)
      #print(Qbar(Emid))
      sigcalc = np.sqrt(c[0] - b[0]**2)
   
      delta[i] = sigmid/sigcalc
      cmid = np.sqrt(sigmid**2 - sig_ss(Emid)**2)
      ccalc = np.sqrt(sigcalc**2-sig_ss(Emid)**2)
      #print('percent error: {:1.2f}%'.format(((sigcalc-sigmid)/sigmid)*100))
      #print('percent error on C: {:1.2f}%'.format(((ccalc-cmid)/cmid)*100)) 
    

    return Ec, delta
