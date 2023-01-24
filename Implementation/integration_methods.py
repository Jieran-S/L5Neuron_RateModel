import numpy as np
from scipy import integrate 

def forward_euler(fprime, x, **kwargs):
    '''
    fprime: new act generation methods, i.e. update_network function
    fprime: can also be learning rules when updating the W_rec matrix
    x: Current parameter
    commonly use this one
    '''
    delta_t=kwargs['delta_t']
    return (x + delta_t*fprime(x, kwargs))
    
def runge_kutta_explicit(fprime, x, **kwargs):
    '''
    fprime: new act generation methods, i.e. update_network function
    x: Current parameter
    '''
    delta_t=kwargs['delta_t']
    k1 = fprime(x, kwargs)
    k2 = fprime(x + 0.5*k1*delta_t, kwargs)
    k3 = fprime(x + 0.5*k2*delta_t, kwargs)
    k4 = fprime(x + k3*delta_t, kwargs)
    x_new = x + delta_t*( (1./6.)*k1 + (1./3.)*k2 + (1./3.)*k3 + (1./6.)*k4 )
    # return x_new
    return np.clip(x_new, 0, None)
 
def update_network(x, kwargs):
    '''
    nonlinearity: nonlin function, either supralinear(ReLu), tanh or Sigmoid
    
    For each step, the activity is updated based on the 
    1. the current weight (w_rec) and current neuron activity (x)
    2. the input stimulus (Input), as it directly affecting the post-syn neurons, it doesn't need to be applied weights
    '''    
    # Define the timestep to jump and nonlinear function
    timescale=1/(kwargs['tau'])
    nonlinearity=kwargs['nonlinearity']    

    # W-rec should be changed and w_input should be steady
    update= -x + nonlinearity(np.dot(kwargs['w_rec'], x) + np.dot(kwargs['w_input'], kwargs['Input'])) 
    
    # linearized change amount
    return timescale*update

def nonlearning_weights(x, kwargs):
    return np.zeros_like(x)

def Simple_test_learn(x, kwargs):
    '''
    Testing if the learning rule works and the system is actually updating.

    N = np.array([45, 275, 46, 34])
    w_cs = x[:, :N[0]] * 1.01
    w_cc = x[:, sum(N[:1]):sum(N[:2])] * 0.98
    w_pv = x[:, sum(N[:2]):sum(N[:3])] * 1.05
    w_sst = x[:, sum(N[:3]):sum(N)] * 0.95
    '''
    return x*1.04

def BCM_rule(weights_project, kwargs):
    """
    Run reference: http://www.scholarpedia.org/article/BCM_theory
    inputs: input stimulus each time
    N: number of different neuron types 
    neurons: referring to which type of neurons' weight is updated
    train_mode: at this time step which typo of neuron is trained
    w_struct_mask: an eye matrix making sure the output dimension fits

    """

    timescale_learn=1./(kwargs['tau_learn'])
    activity_all=np.asarray(kwargs['prev_act'])
    activity_current=activity_all[-1]
    inputs=kwargs['Input']
    N = kwargs['N']
    
    tresholds=np.var(activity_all, axis=0) + np.mean(activity_all, axis=0)**2
    activity_presyn = activity_current*(activity_current-tresholds)
    weight_change=np.dot(activity_presyn[:,None], inputs[None,:])/(tresholds[:,None])

    ######## Config which neurons are trianed and their patterns ########
    #trimming only for excitory neurons(CC, CS)
    if kwargs['neurons'] == 'excit_only': 
        weight_change[:, -np.sum(N[2:]):]= 0
        weight_change[-np.sum(N[2:]):, :]= 0
        
    elif kwargs['neurons'] == 'inhibit_only': 
        weight_change[:, :np.sum(N[:2])]= 0
        weight_change[:np.sum(N[2:]), :]= 0

    # If specific training patterns are sepecified
    if kwargs['train_mode'] == 'CS':
        # Only updating the weight of CS as post-synaptic neurons (rows)
        weight_change[-np.sum(N[1:]):, :]= 0

    elif kwargs['train_mode'] == 'CC':
        # only updating the weight of CC as post-synaptic neurons (rows)
        weight_change[:N[0],:] = 0
        weight_change[-np.sum(N[2:]):, :]= 0
    
    elif kwargs['train_mode'] == 'Rest':
        weight_change[:,:] = 0
    
    # return a N x N matrix, row is post-syn, col is pre-syn
    # print(timescale_learn*weight_change*kwargs['w_struct_mask'])
    return timescale_learn*weight_change*kwargs['w_struct_mask']

def BCM_rule_sliding_th(weights_project, kwargs):
    """
    rule reference: http://www.scholarpedia.org/article/BCM_theory
    Same documentation as BCM rules. sliding BCM changes threshold for learning all the time
    """

    def threshold_old(activity_all):
        return np.var(activity_all, axis=0) + np.mean(activity_all, axis=0)**2    
    
    def threshold_function(activity_all, dt, timescale):
        number_timepoints=len(activity_all)
        #integral_factors  
        prefactors_inside_integral_pre=np.arange(-1*number_timepoints+1,1)*dt
        prefactors_inside_integral=np.exp(prefactors_inside_integral_pre*timescale)
        thresholds=timescale*integrate.simps(y=prefactors_inside_integral[:,None]*activity_all**2,x=None, dx=dt, axis=0)

        return thresholds
    
    timescale_learn=1./(kwargs['tau_learn'])
    activity_all=np.asarray(kwargs['prev_act'])
    inputs=kwargs['Input']
    N = kwargs['N']
    activity_current=activity_all[-1]
    thresholds=threshold_function(activity_all=np.asarray(activity_all), 
                                dt=kwargs['delta_t'],
                                timescale=1./(kwargs['tau_threshold']))

    activity_postsyn = activity_current*(activity_current-thresholds)
    # post-syn as row and pre-syn as column
    weight_change=np.dot(activity_postsyn[:,None], inputs[None,:])/(thresholds[:,None])

    # Weight change for inihibitory neuron as pre-syn should be reveresed 
    # Row: Postsyn; Col: Presyn
    weight_change[:, -np.sum(N[2:]):] *= -1

    ######## Config which neurons are trianed and their patterns ########

    #trimming only for excitory neurons(CC, CS)
    if kwargs['neurons'] == 'excit_only': 
        weight_change[:, -np.sum(N[2:]):]= 0
        weight_change[-np.sum(N[2:]):, :]= 0
        
    elif kwargs['neurons'] == 'inhibit_only': 
        weight_change[:, :np.sum(N[:2])]= 0
        weight_change[:np.sum(N[2:]), :]= 0

    # If specific training patterns are sepecified
    if kwargs['train_mode'] == 'CS':
        # Only updating the weight of CS as post-synaptic neurons (rows)
        weight_change[-np.sum(N[1:]):, :]= 0

    elif kwargs['train_mode'] == 'CC':
        # only updating the weight of CC as post-synaptic neurons (rows)
        weight_change[:N[0],:] = 0
        weight_change[-np.sum(N[2:]):, :]= 0
    
    elif kwargs['train_mode'] == 'Rest':
        weight_change[:,:] = 0

    # return a N x N matrix, row is post-syn, col is pre-syn
    return timescale_learn*weight_change*kwargs['w_struct_mask']
 
def Oja_rule(weight_project, kwargs):
    """
    Reference: https://neuronaldynamics.epfl.ch/online/Ch19.S2.html#Ch19.F5
    same construction for other learning rules
    """
    timescale_learn=1./(kwargs['tau_learn'])
    activity_all=np.asarray(kwargs['prev_act'])
    inputs=kwargs['Input']
    N = kwargs['N']
    activity_postsyn= activity_all[-1]

    # Weight_change = postsyn[activity_presyn - w_rec*postsyn]
    activity_presyn = activity_postsyn + inputs - np.dot(weight_project, activity_postsyn)
    # post-syn as row and pre-syn as column
    weight_change=np.dot(activity_postsyn[:,None], activity_presyn[None,:])

    # Weight change for inihibitory neuron as pre-syn should be reveresed 
    # Row: Postsyn; Col: Presyn
    weight_change[:, -np.sum(N[2:]):] *= -1

    ######## Config which neurons are trianed and their patterns ########
    #trimming only for excitory neurons(CC, CS)
    if kwargs['neurons'] == 'excit_only': 
        weight_change[:, -np.sum(N[2:]):]= 0
        weight_change[-np.sum(N[2:]):, :]= 0
        
    elif kwargs['neurons'] == 'inhibit_only': 
        weight_change[:, :np.sum(N[:2])]= 0
        weight_change[:np.sum(N[2:]), :]= 0

    # If specific training patterns are sepecified
    if kwargs['train_mode'] == 'CS':
        # Only updating the weight of CS as post-synaptic neurons (rows)
        weight_change[-np.sum(N[1:]):, :]= 0

    elif kwargs['train_mode'] == 'CC':
        # only updating the weight of CC as post-synaptic neurons (rows)
        weight_change[:N[0],:] = 0
        weight_change[-np.sum(N[2:]):, :]= 0
    
    elif kwargs['train_mode'] == 'Rest':
        weight_change[:,:] = 0

    # return a N x N matrix, row is post-syn, col is pre-syn
    return timescale_learn*weight_change*kwargs['w_struct_mask']

def Cov_rule(weight_project, kwargs):
    """
    reference: https://neuronaldynamics.epfl.ch/online/Ch19.S2.html#Ch19.F5 
    other setting is the same
    """
    # Covariance-driven learning rules
    timescale_learn=1./(kwargs['tau_learn'])
    activity_all=np.asarray(kwargs['prev_act'])
    inputs=kwargs['Input']
    N = kwargs['N']

    activity_average = np.mean(activity_all, axis=0).reshape(-1,)
    activity_current = activity_all[-1] - activity_average

    # Taking the covariance of the current activity based on the its difference between the past average
    weight_change=np.dot(activity_current[:, None], activity_current[None,:])
    
    # Weight change for inihibitory neuron as pre-syn should be reveresed 
    # Row: Postsyn; Col: Presyn
    weight_change[:, -np.sum(N[2:]):] *= -1

    ######## Config which neurons are trianed and their patterns ########
    #trimming only for excitory neurons(CC, CS)
    if kwargs['neurons'] == 'excit_only': 
        weight_change[:, -np.sum(N[2:]):]= 0
        weight_change[-np.sum(N[2:]):, :]= 0
        
    elif kwargs['neurons'] == 'inhibit_only': 
        weight_change[:, :np.sum(N[:2])]= 0
        weight_change[:np.sum(N[2:]), :]= 0

    # If specific training patterns are sepecified
    if kwargs['train_mode'] == 'CS':
        # Only updating the weight of CS as post-synaptic neurons (rows)
        weight_change[-np.sum(N[1:]):, :]= 0

    elif kwargs['train_mode'] == 'CC':
        # only updating the weight of CC as post-synaptic neurons (rows)
        weight_change[:N[0],:] = 0
        weight_change[-np.sum(N[2:]):, :]= 0
    
    elif kwargs['train_mode'] == 'Rest':
        weight_change[:,:] = 0

    # return a N x N matrix, row is post-syn, col is pre-syn
    return timescale_learn*weight_change*kwargs['w_struct_mask']