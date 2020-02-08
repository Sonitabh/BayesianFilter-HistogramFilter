import numpy as np
import matplotlib.pyplot as plt
from histogram_filter import HistogramFilter
import random


if __name__ == "__main__":

    # Load the data
    data = np.load(open('data/starter.npz', 'rb'))
    cmap = data['arr_0']
    #actions = data['arr_1']
    actions = np.array([1,0])
    
    #observations = data['arr_2']
    observations = 0
    belief_states = data['arr_3']
    
    h,w = np.shape(cmap)
    belief = np.ones((h,w))
    belief = (1/(h*w))*belief
    obj = HistogramFilter()
    ind, bel = obj.histogram_filter(cmap,belief,actions,observations)
    #print(belief_states)


    #### Test your code here
