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
                neurons = 'excit_only',
                phase_list = np.repeat(['CS','CC','Rest'], 10) ):
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
        self.neurons = neurons
        self.phase_list = phase_list

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
        # if self.learning_rule=='Simple_test':
        #     self.learningrule = im.Simple_test_learn
        if self.learning_rule=='Oja':
            self.learningrule = im.Oja_rule
            
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
        #evaluating only the mean value for the activity
        if activities.shape[0]%25 !=0:
            return False
        else:
            activities = np.mean(activities.reshape(-1,25, activities.shape[1]), axis = 1)

            a1 = activities[-20:-10, :]
            a2 = activities[-10:, :]
            mean1 = np.mean(a1, axis=0)
            mean2 = np.mean(a2, axis=0)
            check_eq = np.sum(np.where(mean1 - mean2 < 0.05, np.zeros(self.W_rec.shape[0]), 1))
            if check_eq < int(self.W_rec.shape[0]* 0.1):
                return True
            else:
                return False
            
    def run(self, inputs, start_activity, simulate_till_converge):
        Ntotal = self.tsteps
        all_act=[]
        all_act.append(start_activity)
        all_weights=[]
        Latest_weight = self.W_rec
        
        inputs_time = self._check_input(inputs)
        for step in range(Ntotal):
            # For each update, first update the act such that the neuron activation level get updated
            new_act=self.integrator_function(self.update_act,  #intergation method
                              all_act[-1],  #general parameters     
                              delta_t=self.delta_t,
                              tau=self.tau, w_rec= Latest_weight , w_input=self.W_input, #kwargs the input should be the same while keeping
                              Input=inputs_time[step],            
                              nonlinearity=self.np_nonlinearity, )
            
            all_act.append(new_act) # append new activity to use for learning

            # check if the neuron get to the stationary point
            if step<self.number_steps_before_learning:# or not(step%100==0): # added not(step%100...)
                # if not just use the current weight
                new_weights = Latest_weight
            else:
                train_mode = self.phase_list[step%len(self.phase_list)]
                if train_mode == "Rest":
                    # if the system is resting, no training is happening
                    new_weights = Latest_weight
                else:
                    # Either training CC or CS repeatedly
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
                                            N = self.N, 
                                            neurons = self.neurons,
                                            train_mode = train_mode)
            
            # attach only every 1 in 5 weight matrix to save space
            if step%1 == 0:
                all_weights.append(new_weights)
            Latest_weight = new_weights

        if simulate_till_converge == True: 
            # Adding convergence check for last 100 steps
            while self.check_convergence(activities=np.array(all_act)) != True: 
                '''
                Remark: For steady input, we can just reuse the previous input.
                But for moving input related to t, we need to find the timing for the cycle to match the smooth transition 
                np.cos(temporalF * t) -> Find the position for np.cos(temporalF * step)
                Closest number: remainder of step divided by 2 pi
                '''
                step = step+1
                input_step = 10 + step - Ntotal
                # int(Ntotal%(2*np.pi)) + step - Ntotal
                
                new_act=self.integrator_function(self.update_act,  #intergation method
                                all_act[-1],  #general parameters     
                                delta_t=self.delta_t,
                                tau=self.tau, w_rec= Latest_weight,
                                w_input=self.W_input, #kwargs the input should be the same while keeping
                                Input=inputs_time[input_step],            
                                nonlinearity=self.np_nonlinearity, )
                all_act.append(new_act) # append new activity to use for learning
                
                Train_mode = self.phase_list[step%len(self.phase_list)]
                if Train_mode == "Rest":
                    # if the system is resting, no training is happening
                    new_weights = Latest_weight
                else:
                    # Either training CC or CS repeatedly
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
                                            N = self.N, 
                                            neurons = self.neurons,
                                            train_mode = train_mode)
                if step%1 == 0:
                    all_weights.append(new_weights)
                Latest_weight = new_weights

                if step > int(Ntotal*2 - 10): # int(Ntotal%(2*np.pi)) - 1):
                    break
        
        self.activity = all_act[-Ntotal:]
        self.weights = all_weights[-Ntotal:]
        self.step = step
        return self.activity, self.weights, step
            
    ############### Plotting and visualization ################## 
    def plot_activity(self, activity, sim, saving = False):
        '''
        activity: 3d matrix with infomraiton on the activation of different neurons
        '''
        N = self.N
        Ttau = self.Ttau
        learningrule = self.learning_rule
        step = self.step

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
                n = 25
                # axs.plot(np.arange(act.shape[0])*5, act[:,i],c='grey',alpha=0.5)
                axs.plot(np.linspace(0, Ttau, act.shape[0]), act[:,i],c='grey',alpha=0.3)

            for i in range(act.shape[1]):
                mean_act = np.average(act[:,i].reshape(-1, n), axis=1)
                axs.plot(np.linspace(0, Ttau, act.shape[0]), np.repeat(mean_act, n),c='orange',alpha=0.8)
            LearningTime = (self.number_steps_before_learning - (self.step+1- self.tsteps))*self.delta_t/self.tau
            axs.axvline( x= LearningTime,linewidth=2, color='green')
            axs.set_title(namelist[ind])

        fig.tight_layout(pad=2.0)

        now = datetime.now() # current date and time
        DateFolder = now.strftime('%m_%d')
        if os.path.exists(f'data/{DateFolder}') == False:
            os.makedirs(f'data/{DateFolder}')
            
        if saving == True:
            time_id = now.strftime("%m%d_%H:%M")

            time_id = datetime.now().strftime("%m%d_%H:%M")
            title_save = f'data/{DateFolder}/{learningrule}_{sim}_{time_id}_act.png'
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

        if self.neurons == 'excit_only':
            postsyn_weights = [weight_cs, weight_cc]
            fig, axes = plt.subplots(1,2)
        elif self.neurons == 'inhibit_only':
            postsyn_weights = [weight_pv, weight_sst]
            fig, axes = plt.subplots(1,2)
        else:
            postsyn_weights = [weight_cs, weight_cc, weight_pv, weight_sst]
            fig, axes = plt.subplots(2,2)
        
        target_weight = np.array([  [ 0.01459867,  0.        ,  0.06143608,  0.00388622],
                                    [ 0.00577864,  0.00486622,  0.03568564,  0.00790761],
                                    [-0.04649947, -0.06677541, -0.07941407, -0.02081662],
                                    [-0.0333877 , -0.00483243, -0.01764006, -0.00642071]])
        Wref_col = [ 'dodgerblue','deeppink','seagreen','fuchsia']
        
        for j,wei in enumerate(postsyn_weights):
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
            if self.neurons == 'excit_only':
                presyn_weights = [w_to_cs,w_to_cc]
            elif self.neurons == 'inhibit_only':
                presyn_weights = [w_to_pv, w_to_sst]
            else:
                presyn_weights = [w_to_cs,w_to_cc, w_to_pv, w_to_sst]
            
            for ind,plotwei in enumerate(presyn_weights):
                # Different cell numbers
                for i in range(plotwei.shape[1]):
                    # axs.plot(np.arange(x_length)*5, plotwei[:, i], c = color_list[ind], label = label_list[ind], alpha = 0.5)
                    axs.plot(np.linspace(0, Ttau, x_length), 
                            plotwei[:, i], 
                            c = color_list[ind], 
                            label = label_list[ind], 
                            alpha = 0.5)
                
                if self.neurons == 'inhibit_only':
                    ind_tar = ind+2
                    j_tar = j+2
                else:
                    ind_tar = ind 
                    j_tar = j
                axs.axhline(y = target_weight[ind_tar, j_tar], linewidth=2, color=Wref_col[ind])
            
            name = ['CS', 'CC', 'PV','SST']
            LearningTime = max((self.number_steps_before_learning - 
                                (self.step+1- self.tsteps))*self.delta_t/self.tau, 
                                0)
            axs.axvline( x= LearningTime,linewidth=2, color='green')
            axs.set_title(f"postsyn(col):{name[j]}")
        
        # Set legend content and location
        handles, labels = axs.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        fig.legend( by_label.values(), 
                    by_label.keys(),
                    loc = 'lower center', 
                    ncol = 4, bbox_to_anchor=(0.5, 0))

        #save graph
        fig.tight_layout(pad=2.0)

        now = datetime.now() # current date and time
        DateFolder = now.strftime('%m_%d')
        if os.path.exists(f'data/{DateFolder}') == False:
            os.makedirs(f'data/{DateFolder}')

        if saving == True:
            time_id = now.strftime("%m%d_%H:%M")

            time_id = datetime.now().strftime("%m%d_%H:%M")
            title_save = f'data/{DateFolder}/{learningrule}_{sim}_{time_id}_weight.png'
            fig.savefig(title_save)
