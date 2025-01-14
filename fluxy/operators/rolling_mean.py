import numpy as np



def calc_rolling_mean_v0(list_data,rolling_mean=2):
    """
    Calculate rolling mean of a list of numpy array using numpy.convolv
    (see https://stackoverflow.com/questions/14313510/how-to-calculate-rolling-moving-average-using-python-numpy-scipy).
    
    Args:
        list_data : list of numpy array of dtype float or numpy.datetime64
        rolling_mean : rolling_mean period
        
    Return
        averaged_data : list of numpy array containing the averaged data.
    """
    if rolling_mean is None :
        return list_data
    
    elif rolling_mean>=list_data[0].size:
        raise ValueError(f'rolling_mean value ({rolling_mean}) should be inferior to the size of the data ({list_data[0].size}).')
        
    else :
        averaged_data = list()
        for data in list_data:
            if np.issubdtype(data.dtype, np.datetime64):
                averaged_data.append((np.convolve(data.astype(int), np.ones(rolling_mean), 'valid') / rolling_mean
                                     ).astype(data.dtype))
            else : 
                averaged_data.append(np.convolve(data, np.ones(rolling_mean), 'valid') / rolling_mean)
                
        return averaged_data
 
def calc_rolling_mean(list_data):
    """
    Calculate rolling mean of a list of numpy array using numpy.convolv
    (see https://stackoverflow.com/questions/14313510/how-to-calculate-rolling-moving-average-using-python-numpy-scipy).
    
    Args:
        list_data : list of numpy array of dtype float or numpy.datetime64
        
    Return
        averaged_data : list of numpy array containing the averaged data.
    """
    rolling_mean = 3
    averaged_data = list()
    for data in list_data:
        if np.issubdtype(data.dtype, np.datetime64):
            tmp = (np.convolve(data.astype(int), np.ones(rolling_mean), 'valid') / rolling_mean
                  ).astype(data.dtype)
            tmp = np.concatenate([[data[0],],tmp,[data[-1],]])
        else : 
            tmp = np.convolve(data, np.ones(rolling_mean), 'valid') / rolling_mean
            tmp = np.concatenate([[np.mean(data[:2]),],tmp,[np.mean(data[-2:]),]])
        averaged_data.append(tmp)
    return averaged_data
 
def calc_3yr_rolling_mean_v2(list_data):
    """
    Calculate rolling mean of a list of numpy array using numpy.convolv
    (see https://stackoverflow.com/questions/14313510/how-to-calculate-rolling-moving-average-using-python-numpy-scipy).
    
    Args:
        list_data : list of numpy array of dtype float or numpy.datetime64
        
    Return
        averaged_data : list of numpy array containing the averaged data.
    """
    rolling_mean = 3
    averaged_data = list()
    for data in list_data:
        print(data,type(data))
        if np.issubdtype(data.dtype, np.datetime64):
            tmp = (np.convolve(data.astype(int), np.ones(rolling_mean), 'valid') / rolling_mean
                  ).astype(data.dtype)
            tmp = np.concatenate([[data[0],],tmp,[data[-1],]])
        else : 
            tmp = np.convolve(data, np.ones(rolling_mean), 'valid') / rolling_mean
            tmp = np.concatenate([[data[0],],tmp,[data[-1],]])
        averaged_data.append(tmp)
    return averaged_data

