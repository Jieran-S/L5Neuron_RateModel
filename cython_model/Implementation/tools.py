import numpy as np

def supralinear(gamma):
    return lambda x: np.clip(x , 0, np.inf)**gamma

nl_tanh = lambda x: np.tanh(x)
nl_sigmoid = lambda x: 1./(1+np.exp(-x)) 