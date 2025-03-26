import numpy as np #type: ignore
import pandas as pd #type: ignore

def array_to_pandas(arr, stride, t_offset=0):
    '''Converts a multi-dimensional array into a pandas dataframe without
    value repetitions, assumig that `n_targets` are sampled every `stride`.
    The input array is indexed according to the timestep from the first entry
    (as opposed to the location along the array).
    An offset can be specified so to shift the whole timestep series to a
    given starting point.
    
    Parameters
    ----------
    arr : np.ndarray
        Input sequence lacking timestamps, and with possible overlaps.
    stride : int
        Timestep stride that have been used to create `arr`.
    t_offset : int, optional (default: 0)
        Timestep offset.  Can be used to set the ouput dataframe timesteps to
        a given starting point.
    
    Returns
    -------
    df_arr : pd.DataFrame
        Dataframe of values without repetitions, indexed by timesteps.
    '''
    
    # Retrieving window target size from the data themselves:
    try:
        n_targets = arr.shape[1]
    except:
        n_targets = 1 

    arr_flat  = []
    idxs_flat = []
    keep = min(n_targets, stride)
    # actual number of items to keep along the 2nd dimension, at each stride
    # iteration

    if n_targets == 1: arr = arr.reshape(-1, 1)
    # forcefully reshaping array if it is 1D

    for i in range(0, len(arr), 1):
        timestep = i*stride
        # timestep at the beginning of the window

        if stride >= n_targets:
             idx_min = timestep
        else:
            try:
                idx_min = idxs_flat[-1] + 1
            except:
            # initialization case
                idx_min = 0
        
        idx_MAX = timestep + n_targets

        idxs_flat.extend(np.arange(idx_min, idx_MAX))
        if i == len(arr)-1:
        # on last iteration, keep all entries
            keep = None
        arr_flat.extend(arr[i,:keep].flatten())

    idsx_flat = np.array(idxs_flat)
    arr_flat  = np.array(arr_flat)
    
    idsx_flat += t_offset
    # adding timestep offset
    
    df_arr = pd.DataFrame(data=arr_flat.T, columns=['value'], index=idsx_flat)
    df_arr.index.name = 'idx_t'

    return df_arr
