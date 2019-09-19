#save multiple-scatter data as in notebook: nrFano_Constraint/ms_simulation_yield.ipynb
import numpy as np
import pandas as pd
import sima2py as sapy
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import h5py
warnings.resetwarnings()



#get the equivalent charge energy for a recoil of energy E with parameters in the structure par
#def getLindhard(E,par=None):
def saveMS(file='/data/chocula/villaa/k100Sim_Data/252Cf/Cf252_0x0002.h5',outfile='k100_252Cf_shield_cdmsII_NRs.h5'):

    f = h5py.File(file,"r")

    data = f['geant4/hits']

    #first do some cuts:
    #first some hit-level cuts
    cHVDet = np.zeros(np.shape(data)[0],dtype=bool)
    cZeroEdep = np.zeros(np.shape(data)[0],dtype=bool)
    cNeutron = np.zeros(np.shape(data)[0],dtype=bool)
    cGamma = np.zeros(np.shape(data)[0],dtype=bool)
    cNR = np.zeros(np.shape(data)[0],dtype=bool)

    cHVDet[data[:,1]==1] = True
    cZeroEdep[data[:,6]==0] = True
    cNeutron[data[:,4]==2112] = True
    cGamma[data[:,4]==22] = True
    cNR[data[:,4]>3000] = True

    #now make a dataframe with the restricted data
    nr_data = data[:,[0,4,5,6,21]]
    nr_data = nr_data[cHVDet&~cZeroEdep&cNR,:]
    nr_dataframe = pd.DataFrame(data=nr_data)

    groupbyvec=[0]
    #print(np.max(nr_dataframe.groupby([0,1],axis=0).size()))
    max_vec = np.max(nr_dataframe.groupby(groupbyvec,axis=0).size())
    
    evec = np.zeros((0,max_vec))
    nhit = np.zeros((0,1))
    
    for i in nr_dataframe.groupby(groupbyvec,axis=0)[3].apply(list):
      #print(i)
      #print(np.shape(i))
      vector = np.zeros((1,max_vec))
      #print(np.shape(vector[0,0:np.shape(i)[0]]))
      vector[0,0:np.shape(i)[0]] = np.transpose(np.asarray(i))
      evec = np.append(evec,vector,0)
      nhit = np.append(nhit,np.shape(i)[0])

    #open and write file
    of = h5py.File(outfile, 'w')
    
    d = evec 
    #hits dataset
    dset_hits = of.create_dataset("nr_Fano/nr_energies", np.shape(d), dtype=np.dtype('float64').type, compression="gzip", compression_opts=9)
    dset_hits[...] = d
    
    d = nhit
    dset_hits = of.create_dataset("nr_Fano/nr_hits", np.shape(d), dtype=np.dtype('float64').type, compression="gzip", compression_opts=9)
    dset_hits[...] = d
    
    of.close()

    return 
