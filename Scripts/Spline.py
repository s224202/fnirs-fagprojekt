
# Moving Standard deviation

import numpy as np
from scipy.signal import butter, filtfilt
from scipy.interpolate import CubicSpline

def moving_std(data, window_size):
    """
    Calculate the moving standard deviation of a signal.
    
    Parameters
    ----------
    data : np.ndarray
        The signal to calculate the moving standard deviation of.
    window_size : int
        The size of the window to use for the moving standard deviation.
        
    Returns
    -------
    np.ndarray
        The moving standard deviation of the signal.
    """
    return np.array([np.std(data[i-window_size:i+window_size]) for i in range(window_size, len(data)-window_size)])

# Motion Artifact detection

def motion_artifact_detection(data,T):
    Motion_Artifacts = []
    for i in range(len(data)):
        if data[i] > T:
            Motion_Artifact = []
            while data[i] > T and i < (len(data) - 1):
                i += 1
                Motion_Artifact.append(i)
            Motion_Artifacts.append(Motion_Artifact)
    # Hacky but probably fine
    i = 0
    while i < len(Motion_Artifacts):
        if len(Motion_Artifacts[i]) < 3:
            Motion_Artifacts.pop(i)
            i -= 1
        i += 1
    return Motion_Artifacts

# Motion Artifact correction

def motion_artifact_correction(data):
    
    for i in range(len(data)):
        MSTD = moving_std(data[i], 10)
        MAD = motion_artifact_detection(MSTD, 0.1)
        for j in range(len(MAD)):
            myspline = CubicSpline(MAD[j], data[i][MAD[j]])
            data[i][MAD[j]] = data[i][MAD[j]]-myspline(MAD[j])
    return data
    

