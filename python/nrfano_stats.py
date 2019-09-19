import gammapy.stats as gstats
from scipy import stats
import numpy as np

#get the Feldman-Cousins 68.27% (1sigma) confidence interval on counts with a bknd
#only good for numers up to MaxN 
def getFCErr(n,bkg=0,MaxN=25):

  Nbknd = bkg
  x_bins = np.arange(0, 50)
  mu_bins = np.linspace(0, MaxN, np.int(np.round(MaxN / 0.005 + 1,0)), endpoint=True)
  matrix = [stats.poisson(mu + Nbknd).pmf(x_bins) for mu in mu_bins]
  
  
  acceptance_intervals = gstats.fc_construct_acceptance_intervals_pdfs(matrix, 0.6827)
  LowerLimitNum, UpperLimitNum, _ = gstats.fc_get_limits(mu_bins, x_bins, acceptance_intervals)
  
  #print(matrix)
  #print(mu_bins)
  #print(LowerLimitNum)
  #print(UpperLimitNum)
  N=n
  ul=gstats.fc_find_limit(N, UpperLimitNum, mu_bins)
  ll=gstats.fc_find_limit(N, LowerLimitNum, mu_bins)
  #print(ll)
  #print(ul)

  return ll,ul

#only good for numers up to MaxN 
def largestErr(n,bknd=0,MaxN=25): 

  n=np.asarray(n)
  lims = [getFCErr(x,0,MaxN) for x in n]
  lls = [x[0] for x in lims]
  uls = [x[1] for x in lims]
  lls = np.asarray(lls)
  uls = np.asarray(uls)

  lls[lls==None]=0
  uls[uls==None]=0

  n = np.reshape(n,(np.shape(n)[0],1))
  lls = np.reshape(lls,(np.shape(lls)[0],1))
  uls = np.reshape(uls,(np.shape(lls)[0],1))

  mat = np.zeros((np.shape(lls)[0],0))
 
  mat = np.append(mat,n-lls,axis=1)
  mat = np.append(mat,uls-n,axis=1)
  #print(mat)

  return np.amax(mat,1)

#it takes impossibly long to calculate these things with the above library so precompute it
def largestErr_fast():

  err = [1.29, 1.75, 2.25, 2.3, 2.7750000000000004, 2.8050000000000006, \
       3.2750000000000004, 3.3000000000000007, 3.3149999999999995, \
        3.790000000000001, 3.8049999999999997, 3.8149999999999995, 4.289999999999999, \
         4.300000000000001, 4.315000000000001, 4.32, 4.754999999999999, \
          4.635000000000002, 4.425000000000001, 4.275000000000002, 4.17]

  err = np.asarray(err)
  return err

#function to tell if two vectors of values are in range of a "true" value
def inRange(val,mu,sig):

  if (np.shape(mu) != np.shape(sig)):
    raise Exception('mu and val vectors must have the same shape')

  diff = np.abs(mu-val)

  truth = sig>=diff
  perc = np.float(np.sum(truth==1))/np.float(np.shape(truth)[0])

  return truth,perc

