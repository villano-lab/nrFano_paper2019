#this file gives some functions for computation of Lindhard function 
import numpy as np


#get the equivalent charge energy for a recoil of energy E with parameters in the structure par
#def getLindhard(E,par=None):
def getLindhard(par=None,calck=False):
    #units of E should be eV
 
    #if E==None:
    #  raise ArgumentTypeError('getLindhard: you need an E variable')

    #check that par is right
    if not calck:
      if not set({'Z', 'k', 'a', 'b', 'c', 'd'}).issubset(par):
        raise ArgumentTypeError('getLindhard: one or more parameters (Z,k,a,b,c,d) missing')
    else:
      if not set({'Z', 'A', 'a', 'b', 'c', 'd'}).issubset(par):
        raise ArgumentTypeError('getLindhard: one or more parameters (Z,A,a,b,c,d) missing, set to calculate k by pure Lindhard')


    #par is a dictionary with Z, A, k, a, b, c, d defined as doubles
    #see pg. 89 of Scott Fallows' thesis
    #eps = 11.5 ER[keV] Z**(-(7/3))
    #g(eps) = a(eps)**b + c(eps)**d + (eps)
    Z = par['Z']
    k = 0.0
    A = 0.0
    if calck:
      A = par['A']
      k = 0.133*Z**(2.0/3.0)*A**(-(1.0/2.0))
    else:
      k = par['k']
    a = par['a']
    b = par['b']
    c = par['c']
    d = par['d']
    #Ekev = E/1000.0
    #eps = 11.5*Ekev*Z**(-(7.0/3.0))
    eps = lambda x: 11.5*(x/1000.0)*Z**(-(7.0/3.0))
    #g = a*eps**b + c*eps**d + eps
    g = lambda x: a*eps(x)**b + c*eps(x)**d + eps(x)

    #return k*g/(1+k*g) 
    return lambda x: k*g(x)/(1+k*g(x)) 
#function to get par lists for various materials
def getLindhardPars(mat='Ge',calck = False):

    #check that the material is supported 
    if  mat not in set({'Ge', 'Si'}):
      raise ArgumentTypeError('getLindhardPars: do not have requested material use (Ge,Si)')

    par = {}
    if mat=='Ge':
      par['Z'] = 32
      par['A'] = 73
      if not calck:
        par['k'] = 0.159
      else:
        par['k'] = 0.133*par['Z']**(2.0/3.0)*par['A']**(-(1.0/2.0))
      par['a'] = 3.0
      par['b'] = 0.15
      par['c'] = 0.7
      par['d'] = 0.6
    elif mat=='Si':
      par['Z'] = 14 
      par['A'] = 28 
      if not calck:
        par['k'] = 0.146 #not sure of this variable, essentially same as calc value
      else:
        par['k'] = 0.133*par['Z']**(2.0/3.0)*par['A']**(-(1.0/2.0))
      par['a'] = 3.0
      par['b'] = 0.15
      par['c'] = 0.7
      par['d'] = 0.6

    return par
#shortened function for Ge Lindhard
def getLindhardGe(calck=False):

      pars = getLindhardPars('Ge',calck)
      return getLindhard(pars) #only have to specify calck in one place
#shortened function for Ge Lindhard varying k only
def getLindhardGe_k(k):

      pars = getLindhardPars('Ge',False)
      pars['k'] = k
      return getLindhard(pars) #default of calck is false
#shortened function for Si Lindhard
def getLindhardSi(calck=False):

      pars = getLindhardPars('Si',calck)
      return getLindhard(pars) #only have to specify calck in one place
#shortened function for Si Lindhard varying k only
def getLindhardSi_k(k):

      pars = getLindhardPars('Si',False)
      pars['k'] = k
      return getLindhard(pars) #default of calck is false
