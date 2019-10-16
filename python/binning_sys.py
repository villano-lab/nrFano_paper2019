import numpy as np
import EdwRes as er
import fano_calc as fc
from scipy.integrate import quad
import scipy.interpolate as inter
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import h5py

def getBinSys(C,m,aH=None,scale=None,V=None,filename='data/res_calc.h5',bins = [5, 10, 20, 30, 40, 50, 70, 150],Qbar=lambda x: 0.16*x**0.18):

    #extract some parameters
    A = Qbar(1)
    B = np.log(Qbar(np.exp(1))/A)
    #print(A)
    #print(B)
    bins = np.asarray(bins)
    if (aH is None) and (scale is None) and (V is None):
      print('here')
      Ebase,sigbase = fc.RWCalc(filename=filename,band='NR',alpha=(1/18.0),F=000)
    elif (aH is not None) and (scale is not None) and (V is not None):
      f = er.get_sig_nuc_func_alt(1.3,1.5,3.1,0.4,2.7,pars={'V':scale*V,'eps_eV':3.0,'a':A,'b':B},aH=aH,C=C,m=m)
      Ebase = np.arange(0.1,200,0.1)
      sigbase = f(Ebase)
    else:
      raise Exception('getBinSys: parameters not supplied')
      return -999.999
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
    
      a = quad(pq_norm,-1,5,limit=100,args=(l,h,))
      b = quad(pq_mean,-1,5,limit=100,args=(l,h,))
      c = quad(pq_std,-1,5,limit=100,args=(l,h,))
    
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

def saveBinSys(Nsamp=100,filename='data/mcmc_fits.h5',outfile=None,bins = [5, 10, 20, 30, 40, 50, 70, 150]):

    if outfile is None:
      outfile = filename


    #get some mcmc info:
    f = h5py.File(filename,'r')
    
    #save the results for the Edw fit
    path='{}/{}/'.format('mcmc','edwdata')
    
    Cms = np.asarray(f[path+'Cms'])
    slope = np.asarray(f[path+'m'])
    samples = np.asarray(f[path+'samples'])
    sampsize = np.asarray(f[path+'sampsize'])
    xl = np.asarray(f[path+'Er'])
    upvec = np.asarray(f[path+'Csig_u'])
    dnvec = np.asarray(f[path+'Csig_l'])
    Sigss = np.asarray(f[path+'Sigss'])
    
    f.close()

    print(bins)
    delta = lambda C,m: getBinSys(C,m,bins=bins)[1]
    #deltav = np.vectorize(delta)
    
    #Csamp = np.zeros((100,))
    #msamp = np.zeros((100,))
    print(np.shape(samples[np.random.randint(len(samples), size=Nsamp)]))
    Candm = samples[np.random.randint(len(samples), size=Nsamp)]
    
    #print(Candm[:,0])
    #print(Candm[:,1])
    deli = [deltav(Candm[i,0],Candm[i,1]) for i in range(Nsamp)]
    deli = np.array(deli)
    print(deli)

    #save the results for the MS fit
    path='{}/{}/'.format('mcmc','edwdata_binsys')
    
    
    #remove vars
    f = h5py.File(outfile,'a')
    exbins = path+'bins_binsys' in f
    exsys = path+'binsys' in f

    if exbins:
      del f[path+'bins_binsys']
    if exsys:
      del f[path+'binsys']
    
    dset = f.create_dataset(path+'bins_binsys',np.shape(bins),dtype=np.dtype('float64').type, \
    compression="gzip",compression_opts=9)
    dset[...] = bins 
    dset = f.create_dataset(path+'binsys',np.shape(deli),dtype=np.dtype('float64').type, \
    compression="gzip",compression_opts=9)
    dset[...] = deli 
    
    f.close() 

    return

def saveBinSys_6par(Nsamp=100,filename='data/systematic_error_fits.h5',outfile=None,bins = [5, 10, 20, 30, 40, 50, 70, 150]):

    if outfile is None:
      outfile = filename


    #get some mcmc info:
    f = h5py.File(filename,'r')
    
    #save the results for the Edw fit
    path='{}/{}/'.format('mcmc','edwdata_sys_error')
    
    Cms = np.asarray(f[path+'Cms'])
    slope = np.asarray(f[path+'m'])
    a_yield = np.asarray(f[path+'A'])
    b_yield = np.asarray(f[path+'B'])
    aH = np.asarray(f[path+'aH'])
    scale = np.asarray(f[path+'scale'])
    samples = np.asarray(f[path+'samples'])
    sampsize = np.asarray(f[path+'sampsize'])
    xl = np.asarray(f[path+'Er'])
    upvec = np.asarray(f[path+'Csig_u'])
    dnvec = np.asarray(f[path+'Csig_l'])
    Sigtot = np.asarray(f[path+'Sigss'])
    Sigss = np.sqrt(Sigtot**2 - (Cms+slope*xl)**2)
    
    f.close()

    print(bins)
    delta = lambda C,m: getBinSys(C,m,aH=aH,scale=scale,V=4.0,bins=bins,Qbar=lambda x: a_yield*x**b_yield)[1]
    #deltav = np.vectorize(delta)
    
    #Csamp = np.zeros((100,))
    #msamp = np.zeros((100,))
    print(np.shape(samples[np.random.randint(len(samples), size=Nsamp)]))
    Candm = samples[np.random.randint(len(samples), size=Nsamp)]
    
    #print(Candm[:,0])
    #print(Candm[:,1])
    #print(Candm[0,1])
    #print(Candm[0,2])
    deli = [delta(Candm[i,1],Candm[i,2]) for i in range(Nsamp)]
    deli = np.array(deli)
    print(deli)

    #save the results for the MS fit
    path='{}/{}/'.format('mcmc','edwdata_binsys')
    
    
    #remove vars
    f = h5py.File(outfile,'a')
    exbins = path+'bins_binsys' in f
    exsys = path+'binsys' in f

    if exbins:
      del f[path+'bins_binsys']
    if exsys:
      del f[path+'binsys']
    
    dset = f.create_dataset(path+'bins_binsys',np.shape(bins),dtype=np.dtype('float64').type, \
    compression="gzip",compression_opts=9)
    dset[...] = bins 
    dset = f.create_dataset(path+'binsys',np.shape(deli),dtype=np.dtype('float64').type, \
    compression="gzip",compression_opts=9)
    dset[...] = deli 
    
    f.close() 

    return
