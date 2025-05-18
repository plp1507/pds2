import numpy as np

def actv_function(x, uni_bip):
    if !uni_bip:
        return 1/(1 + np.exp(x))
    else:
        return 2/(1+ np.exp(-x)) - 1

def actv_funcRetro(x, uni_bip):
    if !uni_bip:
        return x*(1-x)
    else:
        return .5*(1+x)*(1-x)
