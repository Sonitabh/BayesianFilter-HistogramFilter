# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 21:31:48 2020

@author: sonit
"""

import numpy as np

belief = np.full((20,20),(1/400))
#belief[19,0] =0.0043
cmap = np.rot90(cmap,-1)
max_idxs = []


#Move right

for i in range(0,len(actions)):
    if actions[i][0] != 0:
        ax = 0
        shift = actions[i][0] 
    elif actions[i][1] != 0:
        ax = 1
        shift = actions[i][1] 
    else:
        ax = 999
    
    
    #M = np.roll(test, shift, axis=ax)
    T1 = np.roll(belief, shift, axis=ax)
    T2 = belief*1
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
    
        
    belief_act = 0.1*belief + 0.9*T1 + (1-0.1)*T2
    
    M_obs = np.where(cmap == observations[i], 0.9, 0.1)
    
    belief_next = belief_act*M_obs
    belief_next = belief_next/np.sum(belief_next)
    belief = belief_next
    max_idx = np.unravel_index(np.argmax(belief), belief.shape)
    max_idxs.append(max_idx)

#idx1 = np.where(belief == np.max(belief))