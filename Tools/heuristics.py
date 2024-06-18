import numpy as np
def compute_heuristics(data: np.ndarray) -> np.ndarray:
    heuristics = np.zeros((data.shape[0],data.shape[1],3))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            #peak_index, peak_value = (np.argmax(data[i,j]), np.max(data[i,j]))
            #peak_index = 1 if peak_index == 0 else peak_index
            peak_index = 79
            peak_value = data[i,j][peak_index]
            mean = np.mean(data[i,j][:peak_index]) - np.mean(data[i,j][peak_index:])
            sd = np.std(data[i,j][:peak_index]) - np.std(data[i,j][peak_index:])
            slope = (peak_value - data[i,j][0])/(len(data[i,j])/2) - (peak_value - data[i,j][-1])/(len(data[i,j])/2)
            
            #check for nans 
            assert not np.isnan(mean)
            assert not np.isnan(sd)
            assert not np.isnan(slope)

            heuristics[i,j] = np.array([mean, sd, slope])
    print(heuristics.shape)
    return heuristics