import numpy as np

def supralinear(gamma):
    if gamma == 1:
        return lambda x: np.clip(x , 0, np.inf)
    else:
        return lambda x: np.concatenate([np.clip(x[:8],0,np.inf),np.clip(x[8:],0,np.inf)**gamma])

nl_tanh = lambda x: np.tanh(x)
nl_sigmoid = lambda x: 1./(1+np.exp(-x)) 