import numpy as np

"""read resolution functions from text files"""
def getRFunc(infile):

  #open the file
  f = open(infile)

  #setup output
  out = {}

  for i,line in enumerate(f.readlines()):
    #print(line)
    det = np.uint32(line.split()[0])
    out[det] = {}
    sqrt_vec = np.ones((6,),dtype=np.float64)
    sqrt_vec[0] = np.float64(line.split()[1])
    sqrt_vec[1] = np.float64(line.split()[2])
    sqrt_vec[2] = np.float64(line.split()[3])
    sqrt_vec[3] = np.float64(line.split()[4])
    sqrt_vec[4] = np.float64(line.split()[5])
    sqrt_vec[5] = np.float64(line.split()[6])
    out[det]['sqrt'] = sqrt_vec 
    lin_vec = np.ones((6,),dtype=np.float64)
    lin_vec[0] = np.float64(line.split()[7])
    lin_vec[1] = np.float64(line.split()[8])
    lin_vec[2] = np.float64(line.split()[9])
    lin_vec[3] = np.float64(line.split()[10])
    out[det]['lin'] = lin_vec 

  f.close()
  return out

def makeRFunc(vec,islin = False):

  if not islin:
    f = lambda x: np.sqrt(vec[0] + x*vec[2] + x**2*vec[4]) 
  else:
    f = lambda x: vec[0] + x*vec[2]

  return f
"""read band functions from text files"""
def getBandFunc(prefix):

  #open the file
  f = open(prefix+'_mu.txt')
  g = open(prefix+'_sig.txt')

  #setup output
  out = {}

  for i,line in enumerate(f.readlines()):
    #print(line)
    det = np.uint32(line.split()[0])
    out[det] = {}
    mu_vec = np.ones((2,),dtype=np.float64)
    mu_vec[0] = np.float64(line.split()[1])
    mu_vec[1] = np.float64(line.split()[2])
    out[det]['mu'] = mu_vec 

  for i,line in enumerate(g.readlines()):
    #print(line)
    det = np.uint32(line.split()[0])
    sig_vec = np.ones((4,),dtype=np.float64)
    sig_vec[0] = np.float64(line.split()[1])
    sig_vec[1] = np.float64(line.split()[2])
    sig_vec[2] = np.float64(line.split()[3])
    sig_vec[3] = np.float64(line.split()[4])
    out[det]['sig'] = sig_vec 

  f.close()
  g.close()
  return out
def makeBFunc(vec,issig = False):

  if not issig:
    f = lambda x: vec[0]*(x**vec[1])
  else:
    f = lambda x: np.piecewise(x, [x <= vec[2], x > vec[2]], [lambda t: np.sqrt(vec[1]*t**vec[3] + vec[0])/t, lambda t: np.sqrt(vec[1]*vec[2]**vec[3]+vec[0])/vec[2]]) 

  return f
