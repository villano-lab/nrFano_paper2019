import pandas as pd
from edelweiss_fit import *

# import data from Edelweiss
def getERNR():
    resNR_data = pd.read_csv("data/edelweiss_NRwidth_GGA3_data.txt", skiprows=1, \
                        names=['E_recoil', 'sig_NR', 'E_recoil_err', 'sig_NR_err'], \
                        delim_whitespace=True)

    resER_data = pd.read_csv("data/edelweiss_ERwidth_GGA3_data.txt", skiprows=1, \
                            names=['E_recoil', 'sig_ER', 'sig_ER_err'], \
                            delim_whitespace=True)

    # the sorting is necessary!
    # otherwise the mask defined below will select the wrong data
    resER_data = resER_data.sort_values(by='E_recoil')

    #print (res_data.head(4))

    # set the data up for the fits
    # Edelweiss discards ER points near peaks
    # and first two NR points since they're affected by the threshold
    mask = [True, True, False, False, True, True, True, True, True]
    ER_data = {'Erecoil': resER_data["E_recoil"][mask], 'sigma': resER_data["sig_ER"][mask], 'sigma_err': resER_data["sig_ER_err"][mask]}
    NR_data = {'Erecoil': resNR_data["E_recoil"][2::], 'sigma': resNR_data["sig_NR"][2::], 'sigma_err': resNR_data["sig_NR_err"][2::]}

    return ER_data, NR_data