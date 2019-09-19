#this file gives some functions for computation of Lindhard function 
import numpy as np
import dataPython as dp #my text file library
import scipy.interpolate as inter
import scipy.integrate as integrate
import scipy.optimize as so
import scipy.special as special
from scipy.signal import savgol_filter


#try to make a version of f(t^1/2) because I'll need it
#return a callable
def getft12(version='s2'):

  if version=='s2':
    f = lambda x: 0.343
  elif version=='TF_approx' :
    f = lambda x: 1.43*x**0.35
  else:
    raise ValueError('getft12: invalid request for stopping funtion')

  return f
#try to get the Thomas-Fermi potential in several limits 
#return a callable
def getphi0(version='LT',file=None):

  p0pr = -1.5880464 #seen N-MISC-18-001

  if version=='LT':
    f = lambda x: 1 + p0pr*x + (4/3)*x**(3/2) + (2/5)*p0pr*x**(5/2) \
         + (1/3)*x**3 + (3/70)*p0pr**2*x**(7/2) + (2/15)*p0pr*x**4 \
         + (4/63)*((7/6)-(1/16)*p0pr**3)*x**(9/2)
  elif version=='HT' :
    f = lambda x: 144/x**3 
  elif version=='matched' :
    fl = getphi0('LT')
    fh = getphi0('HT')
    xi = 200 
    xip = 200 
    sig = 100 
    sigp = 100
    #f = lambda x: fl(x)*special.erf((xi-x)/(np.sqrt(2)*sig)) + fh(x)*special.erf((x-xip)/(np.sqrt(2)*sig))
    f1 = lambda x: fl(x)*((1/2) + (1/2)*special.erf((xi-x)/(np.sqrt(2)*sig))) 
    f2 = lambda x: fh(x)*((1/2) + (1/2)*special.erf((x-xip)/(np.sqrt(2)*sigp))) 
    #f = lambda x: (fl(x)*special.erf((xi-x)/(np.sqrt(2)*sig)) + fh(x)*special.erf((x-xip)/(np.sqrt(2)*sig)))
    f = lambda x: f1(x) + f2(x)
  elif version=='numeric':
    #get the data
    if(file==None) :
      raise ValueError('getphi0: data file does not exist')
      return None

    data = dp.getXYdata(file)
    #print(data.keys())

    #convert to numpy arrays
    data['xx']= np.asarray(data['xx'])
    data['yy']= np.asarray(data['yy'])

    #print(np.min(data['yy']))

    #spline fit
    #f = inter.InterpolatedUnivariateSpline (data['xx'], data['yy'], k=2)
    #f = inter.UnivariateSpline (data['xx'], data['yy'], k=3,s=0)
    yhat = savgol_filter(data['yy'], 3, 2) # window size 51, polynomial order 2
    f = inter.UnivariateSpline (data['xx'], yhat, k=3,s=0)
  else:
    raise ValueError('getphi0: invalid request for stopping funtion')

  return f
#try to get the Thomas-Fermi potential's derivative 
#return a callable
def getgradphi0(version='LT',file=None):

  p0pr = -1.5880464 #seen N-MISC-18-001

  if version=='HT':
    fpr = lambda x: -144*3*(1/x**4)
    return fpr
  elif version=='LT':
    fpr = lambda x: p0pr + (2)*x**(1/2) + p0pr*x**(3/2) \
         + x**2 + (3/20)*p0pr**2*x**(5/2) + (8/15)*p0pr*x**3 \
         + (18/63)*((7/6)-(1/16)*p0pr**3)*x**(7/2)
    return fpr

  f = getphi0(version,'data/phi0_NACI_format_mod.txt')

  #make a grid of x, and calculate the derivative on the grid
  dx=2
  X  = np.arange(0.001,1000,dx)
  #X  = np.logspace(0.001,1000,1000)
  #print(X)
  y = np.gradient(f(X),dx)

  #spline fit
  fpr = inter.InterpolatedUnivariateSpline (X, y, k=1)
  #fpr = inter.UnivariateSpline (X, y, k=5,s=9)

  return fpr
#calculate the function g(xi) see N-MISC-18-002 pg 21
def g(xi,version='LT',sol=None):

  #get u and derivative
  if version!='smooth':
    f = getphi0(version,'data/phi0_NACI_format_mod.txt')
    fpr = getgradphi0(version,'data/phi0_NACI_format_mod.txt')
  elif sol is None:
    sol = getTFScreeningFunction()
    f = lambda x: sol(x)[0]
    fpr = lambda x: sol(x)[1]
  else:
    f = lambda x: sol(x)[0]
    fpr = lambda x: sol(x)[1]

  #make a callable for integrand
  integrand = lambda x: np.cos(x)*(f(xi/np.cos(x)) - (xi/np.cos(x))*fpr(xi/np.cos(x)))

  #integrate
  result = integrate.quad(integrand, 0.0, np.pi/2.0,epsrel=0.001)

  return result
#construct the function lambda(t^1/2) see N-MISC-18-002 pg 25
def lam(t12,version='LT',sol=None):

  if version!='gotg':
    gxi = lambda x: g(x,version,sol)[0]
  elif sol is not None:
    gxi = lambda x: sol(x)
  else:
    raise ValueError('lam: invalid input for g(xi)')
   
  #print(np.shape(gxi(1)))
  func = lambda x: t12 - 1/(2*x)*np.float(gxi(x))

  #print(np.shape(func(1)))
  root = so.brentq(func,1e-6,100,rtol=0.001,maxiter=100) #come within 1% of exact root

  return root

#construct f(t^1/2)
def ft12(version='LT',sol=None,xmin=0.001,dx=1e-2):

  lam2 = lambda x: lam(x,version,sol)**2

  #calc derivative
  #make a grid of x, and calculate the derivative on the grid
  #dx=1e-2
  #X  = np.arange(xmin,10,dx)
  X  = np.logspace(np.log10(xmin),np.log10(10),np.int(10/dx))
 
  #print(X)
  #print(np.int(10/dx))
  #print(np.logspace(np.log10(xmin),np.log10(10),np.int(10/dx)))
  lam2v = np.vectorize(lam2)
  #y = np.gradient(lam2v(X),dx)
  #print(np.shape(X))
  fval = lam2v(X)
  #print(np.shape(fval))
  y = np.gradient(fval,X)

  #spline fit
  lam2pr = inter.UnivariateSpline (X, y, k=3,s=0)

  f = lambda x:-(x**2)*lam2pr(x)
  

  return f

#solve for the TF screening function with solve_bvp
#define the mesh
xmax = 10000
dx = 0.1
xmesh = np.arange(1e-3,xmax,dx)

def getTFScreeningFunction():

  y1 = getgradphi0('numeric','data/phi0_NACI_format_mod.txt')
  y0 = getphi0('numeric','data/phi0_NACI_format_mod.txt')
  y1v = np.vectorize(y0)
  y0v = np.vectorize(y1)
  yguess = np.stack((np.asarray(y0(xmesh),dtype=np.float64),np.asarray(y1(xmesh),dtype=np.float64)),axis=0)

  a = integrate.solve_bvp(TFdiffeqsys,TFbc,xmesh,yguess,max_nodes=5000000,verbose=1)
  print(a.status)
  print(a.success)


  return a.sol


def TFdiffeqsys(x,y):
  
  x = np.asarray(x,dtype=np.complex)
  y = np.asarray(y,dtype=np.complex)
  trow = y[1]
  brow = y[0]**(3/2)*x**(-1/2)
  trow=np.real(trow)
  brow=np.real(brow)
  out = np.stack((trow,brow))
  
  return out

def TFbc(ya,yb):

  y1 = getgradphi0('numeric','data/phi0_NACI_format_mod.txt')
  y0 = getphi0('numeric','data/phi0_NACI_format_mod.txt')
  y1v = np.vectorize(y0)
  y0v = np.vectorize(y1)
  out = np.asarray([ya[0]-y0(xmesh[0]),yb[0]-(144/xmax**3)],dtype=np.float64)
    
  return out

def error(f,g,x):

  #check for vectorization
  fv = np.vectorize(f)
  gv = np.vectorize(g)

  return np.sum((fv(x)-gv(x))**2)
