# first line: 1
@memory.cache
def vsigmomEdw_NR(Erecoil_keV, aH):
    return [sigmomEdw(x,band='NR',label='GGA3',F=0.000001,V=4.0,aH=aH,alpha=(1/18.0)) for x in Erecoil_keV]
