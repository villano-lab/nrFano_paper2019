#this file gives some functions for computation of DAMIC-measured Si yield
#Phys. Rev. D 94 082007 (2016)
import numpy as np
import dataPython as dp #my text file library
import scipy.interpolate as inter




#get the equivalent charge energy for a recoil of energy E with parameters in the structure par
#def getLindhard(E,par=None):
def getDAMICy(file='data/DAMIC_siyield_allerr.txt'):
    #units of E should be eV

    #get the data
    damic_data = dp.getXYdata_wXYerr(file)
    #print(damic_data.keys())

    #convert to numpy arrays
    damic_data['xx']= np.asarray(damic_data['xx'])*1000 #make units eV
    damic_data['yy']= np.asarray(damic_data['yy'])*1000 #make units eV
    damic_data['ex']= np.asarray(damic_data['ex'])*1000 #make units eV
    damic_data['ey']= np.asarray(damic_data['ey'])*1000 #make units eV

    #get the yield stuff
    damic_data['yy_yield'] = damic_data['yy']/damic_data['xx']
    damic_data['ey_yield'] = damic_data['yy_yield'] * np.sqrt((damic_data['ey']/damic_data['yy'])**2 + \
		                                     (damic_data['ex']/damic_data['xx'])**2)


    #spline fit
    damic_y = inter.UnivariateSpline (damic_data['xx'], damic_data['yy_yield'], s=1.5)

    return damic_y 
