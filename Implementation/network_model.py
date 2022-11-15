from os.path import abspath, sep, pardir 
import os
import sys
sys.path.append(abspath('') + sep + pardir + sep )
import numpy as np
import time
import Implementation.tools as snt
import Implementation.integration_methods as im
import Implementation.helper as helper
import matplotlib.pyplot as plt
from datetime import datetime


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
                W_structure=None,
                N = np.array([45, 275, 46, 34]),
                excit_only = True):
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
        self.N = N
        self.excit_only = excit_only

        # Tuning: Plotting and simulation parameter. Controlling tsteps = 2000 and learning_step = 250
        self.tsteps=int((Ttau*tau)/delta_t)
        self.Ttau = Ttau
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
                                         nonlinearity=self.np_nonlinearity,
                                         N = self.N, excit_only = self.excit_only)
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
                input_step = int(Ntotal%(2*np.pi)) + step - Ntotal
                
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
                                            nonlinearity=self.np_nonlinearity,
                                            N = self.N, excit_only = self.excit_only)
                all_weights.append(new_weights)
                if step >= int(Ntotal*2 - int(Ntotal%(2*np.pi)) - 1):
                    break
        
        self.activity = all_act
        self.weights = all_weights
        return all_act, all_weights, step
            
        
    def plot_activity(self, activity, sim, saving = False):
        '''
        activity: 3d matrix with infomraiton on the activation of different neurons
        '''
        N = self.N
        Ttau = self.Ttau
        learningrule = self.learning_rule

        if len(activity) == 0:
            return(0)
        # Extract the connectivity data for each cell population? 
        activity = activity[sim]
        activity_cs = activity[:, :N[0]]
        activity_cc = activity[:, sum(N[:1]):sum(N[:2])]
        activity_pv = activity[:, sum(N[:2]):sum(N[:3])]
        activity_sst = activity[:, sum(N[:3]):sum(N)]
        activity_vec = [activity_cs, activity_cc, activity_pv, activity_sst]
        namelist = ['CS', 'CC', 'PV', 'SST']

        fig,axes = plt.subplots(2,2)
        for ind, act in enumerate(activity_vec):
            axs = axes.flatten()[ind]
            for i in range(act.shape[1]):
                axs.plot(np.linspace(0, Ttau, act.shape[0]), act[:,i],c='grey',alpha=0.5)
            axs.set_title(namelist[ind])

        fig.tight_layout(pad=2.0)

        now = datetime.now() # current date and time
        DateFolder = now.strftime('%m_%d')
        if os.path.exists(f'data/{DateFolder}') == False:
            os.makedirs(f'data/{DateFolder}')
        time_id = now.strftime("%m%d_%H:%M")

        time_id = datetime.now().strftime("%m%d_%H:%M")
        title_save = f'data/{DateFolder}/{learningrule}_{sim}_{time_id}_act.png'
        if saving == True:
            fig.savefig(title_save)   
        
    def plot_weights(self, weights, sim, saving = False):

        '''
        weights: a 3D matrix (4D if all simulation taken into account): Tstep x N(post-syn) x N(pre-syn)
        '''
        N = self.N
        Ttau = self.Ttau
        learningrule = self.learning_rule

        weights = weights[sim]
        weight_cs = weights[:, :N[0], :]
        weight_cc = weights[:, sum(N[:1]):sum(N[:2]), :]
        weight_pv = weights[:, sum(N[:2]):sum(N[:3]), :]
        weight_sst = weights[:, sum(N[:3]):sum(N), :]
        weights_vector = [weight_cs, weight_cc, weight_pv, weight_sst]

        fig, axes = plt.subplots(2,2)
        for j,wei in enumerate(weights_vector):
            # return the average weight from one cell to a specific responses
            w_to_cs = np.mean(wei[:, :, :N[0]], axis=-1)
            w_to_cc = np.mean(wei[:, :, sum(N[:1]):sum(N[:2])], axis= -1 )
            w_to_pv = np.mean(wei[:, :, sum(N[:2]):sum(N[:3])], axis= -1 )
            w_to_sst = np.mean(wei[:, :, sum(N[:3]):sum(N)], axis= -1 )
            x_length = w_to_cc.shape[0]
            # see full colortable: https://matplotlib.org/3.1.0/gallery/color/named_colors.html
            color_list = ['blue', 'salmon', 'lightseagreen', 'mediumorchid']
            label_list = ['cs pre', 'cc pre', 'pv pre', 'sst pre']
            # Specify the graph
            axs = axes.flatten()[j]
            # Different weight type
            for ind,plotwei in enumerate([w_to_cs,w_to_cc, w_to_pv, w_to_sst]):
                # Different cell numbers
                for i in range(plotwei.shape[1]):
                    axs.plot(np.linspace(0, Ttau, x_length), plotwei[:, i], c = color_list[ind], label = label_list[ind], alpha = 0.5)
            
            name = ['CS', 'CC', 'PV','SST']
            axs.set_title(f"postsyn(col):{name[j]}")
        
        # Set legend content and location
        handles, labels = axs.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        fig.legend(by_label.values(), by_label.keys(),loc = 'lower center', ncol = 4, bbox_to_anchor=(0.5, 0))

        #save graph
        fig.tight_layout(pad=2.0)

        now = datetime.now() # current date and time
        DateFolder = now.strftime('%m_%d')
        if os.path.exists(f'data/{DateFolder}') == False:
            os.makedirs(f'data/{DateFolder}')
        time_id = now.strftime("%m%d_%H:%M")

        time_id = datetime.now().strftime("%m%d_%H:%M")
        title_save = f'data/{DateFolder}/{learningrule}_{sim}_{time_id}_weight.png'
        if saving == True:
            fig.savefig(title_save)