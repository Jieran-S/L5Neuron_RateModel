import numpy as np
from scipy import integrate

def forward_euler(fprime, x, **kwargs):
    '''
    fprime: new act generation methods, i.e. update_network function
    fprime: can also be learning rules when updating the W_rec matrix
    x: Current parameter
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
    '''    
    # Define the timestep to jump and nonlinear function
    timescale=1/(kwargs['tau'])
    nonlinearity=kwargs['nonlinearity']    

    # 1st term: inner product of connectivity matrix and current PSP intensity + random input 
    #           resulting in the current input stage
    # 2nd term: time step input, only return the input value from the time step?? There's no 2nd input 
    # from the system??
    update= -x + nonlinearity( np.dot(kwargs['w_rec'], x) + np.dot(kwargs['w_input'], kwargs['Input']) )
    
    # linearized change amount
    return timescale*update


def nonlearning_weights(x, kwargs):
    return np.zeros_like(x)

def BCM_rule(weights_project, kwargs):
    timescale_learn=1./(kwargs['tau_learn'])
    activity_all=kwargs['prev_act']
    activity_current=activity_all[-1]
    inputs=kwargs['Input']
    
    tresholds=np.var(activity_all, axis=0) + np.mean(activity_all, axis=0)**2
    activity_presyn = activity_current*(activity_current-tresholds)
    weight_change=np.dot(activity_presyn[:,None], inputs[None,:])/(tresholds[:,None])
    return timescale_learn*weight_change*kwargs['w_struct_mask']


def BCM_rule_sliding_th(weights_project, kwargs):
    def threshold_old(activity_all):
        return np.var(activity_all, axis=0) + np.mean(activity_all, axis=0)**2    
    
    def threshold_function(activity_all, dt, timescale):
        number_timepoints=len(activity_all)
        #integral_factors  
        prefactors_inside_integral_pre=np.arange(-1*number_timepoints+1,1)*dt
        prefactors_inside_integral=np.exp(prefactors_inside_integral_pre*timescale)
        thresholds=timescale*integrate.simps(y=prefactors_inside_integral[:,None]*activity_all**2,x=None, dx=dt, axis=0)
        return thresholds
    
    timescale_learn=kwargs['tau_learn']
    activity_all=kwargs['prev_act']    
    inputs=kwargs['Input']
    activity_current=activity_all[-1]
    thresholds=threshold_function(activity_all=np.asarray(activity_all), 
                                dt=kwargs['delta_t'],
                                timescale=1./(kwargs['tau_threshold']))
    activity_postsyn = activity_current*(activity_current-thresholds)
    weight_change=np.dot(activity_postsyn[:,None], inputs[None,:])/(thresholds[:,None])
    return timescale_learn*weight_change*kwargs['w_struct_mask']
    





