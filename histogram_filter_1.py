# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 16:50:47 2020

@author: sonit
"""

import numpy as np


class HistogramFilter(object):
    """
    Class HistogramFilter implements the Bayes Filter on a discretized grid space.
    """

    def histogram_filter(self, cmap, belief, actions, observations):
        '''
        Takes in a prior belief distribution, a colormap, action, and observation, and returns the posterior
        belief distribution according to the Bayes Filter.
        :param cmap: The binary NxM colormap known to the robot.
        :param belief: An NxM numpy ndarray representing the prior belief.
        :param action: The action as a numpy ndarray. [(1, 0), (-1, 0), (0, 1), (0, -1)]
        :param observation: The observation from the color sensor. [0 or 1].
        :return: The posterior distribution.
        Use starter.npz data given in /data to debug and test your code. starter.npz has sequence of actions and 
        corresponding observations. True belief is also given to compare your filters results with actual results. 
        cmap = arr_0, actions = arr_1, observations = arr_2, true state = arr_3
    
        ### Your Algorithm goes Below.
        '''
        self.cmap = np.rot90(cmap,-1)
        self.belief = belief
        self.actions = actions
        self.observations = observations
        
        if np.isscalar(actions):
            n = 1
        else:
            n = len(self.actions)
        
        self.max_idxs = []
        
        for i in range(0,n):
            
            if n == 1:
                act = self.actions
                obs = self.observations
            else:
                act = self.actions[i]
                obs = self.observations[i]
            
            if self.actions[i][0] != 0:
                ax = 0
                shift = self.actions[i][0] 
            elif self.actions[i][1] != 0:
                ax = 1
                shift = self.actions[i][1]
            
            T1 = np.roll(self.belief, shift, axis=ax)
            T2 = self.belief*1
            if ax == 0:
                if shift > 0:
                    T1[(shift-1)] = 0
                    T2[(shift-1):(np.shape(T2)[ax]-shift)] = 0
                elif shift < 0:
                    T1[shift] = 0
                    T2[-shift:] = 0
            elif ax == 1:
                if shift > 0:
                    T1[:,(shift-1)] = 0
                    T2[:,(shift-1):(np.shape(T2)[ax]-shift)] = 0
                elif shift < 0:
                    T1[:,shift] = 0
                    T2[:,-shift:] = 0            
                
            belief_act = 0.1*self.belief + 0.9*T1 + (1-0.1)*T2
            
            if len(self.observations) == 1:
                obs = self.observations
            else:
                obs = self.observations[i]
            
            M_obs = np.where(self.cmap == obs, 0.9, 0.1)
            #M_obs = np.where(self.cmap == self.observations[i], 0.9, 0.1)
            
            belief_next = belief_act*M_obs
            belief_next = belief_next/np.sum(belief_next)
            self.belief = belief_next
            max_idx = np.unravel_index(np.argmax(self.belief), self.belief.shape)
            self.max_idxs.append(max_idx)
            
        return np.array(self.max_idxs[-1]) , self.belief