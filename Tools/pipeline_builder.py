from sklearn.pipeline import Pipeline, make_pipeline
from Tools.function_wrappers import wiener_wrapper
# The big pipeline builder function
def build_pipeline(systemic:str, motion:str, phys:str, classifier:str)-> Pipeline:
    ''' 
    Build a pipeline based on the input parameters.
    Systemic: The type of systemic filtering to be applied
    Motion: The type of motion artifact removal to be applied
    Phys: The type of physiological noise removal to be applied
    Classifier: The type of classifier to be applied

    Supported types:
    Systemic: 'Low pass', 'Wiener',
    Motion: 'ICA', 'PCA', 'Spline', 'TDDR',
    Phys: 'Bandpass', 'PCA', 'bPCA',

    '''

    
    # Check for motion artifact removal type
    if motion == 'ICA':
        print('We should be doing some ICA here')
    elif motion == 'PCA':
        print('We should be doing some PCA here')
    elif motion == 'Spline':
        print('We should be doing some Spline here')
    elif motion == 'TDDR':
        print('We should be doing some TDDR here')
    elif motion == 'None':
        print('No motion artifact removal')
    else:
        raise ValueError('Motion artifact removal not recognized, please check your input')
    
    # Check for physiological noise removal type
    if phys == 'Bandpass':
        print('We should be doing some Bandpass filtering here')
    elif phys == 'PCA':
        print('We should be doing some PCA here')
    elif phys == 'bPCA':
        print('We should be doing some bPCA here')
    elif phys == 'None':
        print('No physiological noise removal')
    else:
        raise ValueError('Physiological noise removal not recognized, please check your input')
    

def systemic_function(systemic:str)-> function:
    if systemic == 'Low pass':
        print('We should be doing some low pass filtering here')
    elif systemic == 'Wiener':
        return wiener_wrapper
    elif systemic == 'Regression':
        print('We should be doing some regression here')
    elif systemic == 'None':
        print('No systemic filtering')
    else:
        raise ValueError('Systemic filtering not recognized, please check your input')