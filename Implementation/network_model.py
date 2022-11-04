from os.path import abspath, sep, pardir 
import sys
sys.path.append(abspath('') + sep + pardir + sep )
import numpy as np
import time
import Implementation.tools as snt
import Implementation.integration_methods as im
import Implementation.helper as helper

class SimpleNetwork:
    def __init__(self,
                W_rec,
                W_project,
                nonlinearity_rule,
                integrator='forward_euler',
                delta_t=0.015, 
                number_steps_before_learning = 250,
                tau=5.,
                tau_learn=1000,
                tau_threshold=1000,
                Ttau=6,
                update_function='version_normal',
                learning_rule = 'Simple_test',
                gamma=1,
                W_structure=None):
        self.W_rec =W_rec
        self.W_input=W_project
        if W_structure is not None:
            self.W_structure=W_structure
        else:
            # N x N matrix (postsynaptic neuron no. x total neuron?)
            self.W_structure=np.ones((W_rec.shape[0], W_project.shape[-1]))
        self.delta_t = delta_t
        
        # Changing the rate of activity and learning update (threshold update: tau_threshold)
        self.tau=tau
        self.tau_learn=tau_learn
        self.tau_threshold=tau_threshold

        # Tuning: Plotting and simulation parameter. Controlling tsteps = 2000 and learning_step = 250
        self.tsteps=int((Ttau*tau)/delta_t)
        self.number_steps_before_learning = number_steps_before_learning
        self.number_timepoints_plasticity = int(-1*(self.tau_threshold/self.delta_t)*np.log(0.1))
        self.update_function=update_function
        self.learning_rule=learning_rule
        self.gamma=gamma
        
        self.nonlinearity_rule = nonlinearity_rule
        self.integrator = integrator    
        self.inputs = None        
        self.start_activity = 0.
        self._init_nonlinearity()
        self._init_update_function()
        self._init_learningrule()
        self._init_integrator()
        
    def _init_nonlinearity(self):
        if self.nonlinearity_rule=='supralinear':
            self.np_nonlinearity = snt.supralinear(self.gamma)
        elif self.nonlinearity_rule == 'sigmoid':
            self.t_nonlinearity = snt.nl_sigmoid
        elif self.nonlinearity_rule == 'tanh':
            self.t_nonlinearity = snt.nl_tanh
            
    def _init_update_function(self):
        if self.update_function=='version_normal':
            self.update_act = im.update_network
            
    def _init_learningrule(self):
        if self.learning_rule=='None':
            self.learningrule = im.nonlearning_weights  
        if self.learning_rule=='BCM':
            self.learningrule = im.BCM_rule
        if self.learning_rule=='Slide_BCM':
            self.learningrule = im.BCM_rule_sliding_th
        if self.learning_rule=='Simple_test':
            self.learningrule = im.Simple_test_learn
            
    def _init_integrator(self):        
        if self.integrator == 'runge_kutta':
            self.integrator_function = im.runge_kutta_explicit    
        elif self.integrator == 'forward_euler':
            self.integrator_function = im.forward_euler
        else:
            raise Exception("Unknown integrator ({0})".format(self.integrator))
    
    def _check_input(self, inputs):
        '''
        Make sure the input is of the same type and structure.
        '''
        Ntotal = self.tsteps
        assert inputs.shape[-1]==self.W_input.shape[-1]
        if len(inputs.shape)==1:            
            inputs_time=np.tile(inputs, (Ntotal,1))
        if len(inputs.shape)==2:
            assert inputs.shape[0]==Ntotal
            inputs_time=inputs
        return inputs_time

    def check_convergence(self, activities):
        a1 = activities[-100:-50, :]
        a2 = activities[-50:, :]
        mean1 = np.mean(a1, axis=0)
        mean2 = np.mean(a2, axis=0)
        check_eq = np.sum(np.where(mean1 - mean2 < 0.05, np.zeros(self.W_rec.shape[0]), 1))
        if check_eq < int(self.W_rec.shape[0]* 0.1):
            return True
            
    def run(self, inputs, start_activity, simulate_till_converge):
        Ntotal = self.tsteps
        all_act=[]
        all_act.append(start_activity)
        all_weights=[]
        # all_weights.append(self.W_input)
        all_weights.append(self.W_rec)
        
        inputs_time = self._check_input(inputs)
        for step in range(Ntotal):
            # For each update, first update the act such that the neuron activation level get updated
            new_act=self.integrator_function(self.update_act,  #intergation method
                              all_act[-1],  #general parameters     
                              delta_t=self.delta_t,
                              tau=self.tau, w_rec= all_weights[-1] , w_input=self.W_input, #kwargs the input should be the same while keeping
                              Input=inputs_time[step],            
                              nonlinearity=self.np_nonlinearity, )
            all_act.append(new_act) # append new activity to use for learning
            
            # check if the neuron get to the stationary point
            if step<self.number_steps_before_learning:# or not(step%100==0): # added not(step%100...)
                # if not just use the current weight
                new_weights = all_weights[-1]
            else:
                # if reached, update the weight function with the weight function learning rule
                new_weights = self.integrator_function(self.learningrule,   #learning rule
                                         all_weights[-1], #general parameters 
                                         delta_t=self.delta_t, #kwargs
                                         tau=self.tau, tau_learn=self.tau_learn, 
                                         tau_threshold=self.tau_threshold,
                                         w_rec=self.W_rec , w_input=self.W_input,
                                         w_struct_mask=self.W_structure,
                                         Input=inputs_time[step], 
                                         prev_act=all_act[-self.number_timepoints_plasticity:], 
                                         nonlinearity=self.np_nonlinearity,)
            all_weights.append(new_weights)
        if simulate_till_converge == True: 
            # Adding convergence check for last 100 steps
            while self.check_convergence(activities=np.array(all_act)) != True: 
                step = step+1
                '''
                Remark: For steady input, we can just reuse the previous input.
                But for moving input related to t, we need to find the timing for the cycle to match the smooth transition 
                np.cos(temporalF * t) -> Find the position for np.cos(temporalF * step)
                Closest number: remainder of step divided by 2 pi
                '''
                input_step = int(Ntotal%(2*np.pi))+step-Ntotal+1
                
                new_act=self.integrator_function(self.update_act,  #intergation method
                                all_act[-1],  #general parameters     
                                delta_t=self.delta_t,
                                tau=self.tau, w_rec= all_weights[-1] , 
                                w_input=self.W_input, #kwargs the input should be the same while keeping
                                Input=inputs_time[input_step],            
                                nonlinearity=self.np_nonlinearity, )
                all_act.append(new_act) # append new activity to use for learning
                
                # if reached, update the weight function with the weight function learning rule
                new_weights = self.integrator_function(self.learningrule,   #learning rule
                                            all_weights[-1], #general parameters 
                                            delta_t=self.delta_t, #kwargs
                                            tau=self.tau, tau_learn=self.tau_learn, 
                                            tau_threshold=self.tau_threshold,
                                            w_rec=self.W_rec , w_input=self.W_input,
                                            w_struct_mask=self.W_structure,
                                            Input=inputs_time[input_step], 
                                            prev_act=all_act[-self.number_timepoints_plasticity:], 
                                            nonlinearity=self.np_nonlinearity,)
                all_weights.append(new_weights)

        return all_act, all_weights, step
            
        
        
        
    