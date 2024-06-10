from sklearn.pipeline import Pipeline, FunctionTransformer
from sklearn.svm import SVC
from Tools.function_wrappers import wiener_wrapper, nirs_od_wrapper, nirs_beer_lambert_wrapper, event_splitter_wrapper, butter_bandpass_wrapper, ICA_wrapper, PCA_wrapper, bPCA_wrapper, spline_wrapper, TDDR_wrapper, short_channel_regression_wrapper
from sklearn.preprocessing import StandardScaler
from Tools.heuristics import compute_heuristics
from Tools.Array_transformers import arrayflattener
# The big pipeline builder function
# TODO: Figure out removing bads

def build_pipeline(systemic:str, motion:str, phys:str, classifier:str, split_epochs:bool)-> Pipeline:
    ''' 
    Build a pipeline based on the input parameters. Can be used as a filter by setting the all the parameters to 'None', except for the desired filter.
    Systemic: The type of systemic filtering to be applied
    Motion: The type of motion artifact removal to be applied
    Phys: The type of physiological noise removal to be applied
    Classifier: The type of classifier to be applied

    Supported types(not actually implemented yet):
    Systemic: 'Low pass', 'Wiener',
    Motion: 'ICA', 'PCA', 'Spline', 'TDDR',
    Phys: 'Bandpass', 'PCA', 'bPCA',

    Classifier: 'SVM', 'LDA', 'KNN'

    '''
    estimator_list = []

    #Optional filters
    systemic_func = systemic_function(systemic)
    if systemic_func is not None:
        estimator_list.append(('systemic', FunctionTransformer(systemic_func)))
    motion_func = motion_function(motion)
    if motion_func is not None:
        estimator_list.append(('motion', FunctionTransformer(motion_func)))

    #Mandatory conversion to HbO and HbR    
    estimator_list.append(('nirs_od', FunctionTransformer(nirs_od_wrapper)))
    estimator_list.append(('short_channel_regression', FunctionTransformer(short_channel_regression_wrapper)))
    estimator_list.append(('nirs_beer_lambert', FunctionTransformer(nirs_beer_lambert_wrapper)))

    # Optional filters
    phys_func = phys_function(phys)
    if phys_func is not None:
        estimator_list.append(('phys', FunctionTransformer(phys_func)))
    if split_epochs:
    
    # Mandatory conversion to correct data format
        estimator_list.append(('event_splitter', FunctionTransformer(event_splitter_wrapper)))
    estimator_list.append(('heuristics', FunctionTransformer(compute_heuristics)))
    estimator_list.append(('array_flattener', FunctionTransformer(arrayflattener)))                      
    estimator_list.append(('scaler', StandardScaler()))

    # Optional classifier
    classifier_func = classifier_function(classifier)
    if classifier_func is not None:
        estimator_list.append(('classifier', classifier_func))
    return Pipeline(estimator_list)
    

# Funtions to find the correct function based on the input
def systemic_function(systemic:str):
    if systemic == 'Low pass':
        return butter_bandpass_wrapper
    elif systemic == 'Wiener':
        return wiener_wrapper
    elif systemic == 'Regression':
        print('We should be doing some regression here')
    elif systemic == 'None':
        print('No systemic filtering')
        return None
    else:
        raise ValueError('Systemic filtering not recognized, please check your input')
    
def motion_function(motion:str):
    if motion == 'ICA':
        return ICA_wrapper
    elif motion == 'PCA':
        return PCA_wrapper
    elif motion == 'Spline':
        return spline_wrapper
    elif motion == 'TDDR':
        return TDDR_wrapper
    elif motion == 'None':
        print('No motion artifact removal')
        return None
    else:
        raise ValueError('Motion artifact removal not recognized, please check your input')
    
def phys_function(phys:str):
    # Check for physiological noise removal type
    if phys == 'Bandpass':
        print('We should be doing some Bandpass filtering here')
    elif phys == 'PCA':
        print('We should be doing some PCA here')
    elif phys == 'bPCA':
        return bPCA_wrapper
    elif phys == 'None':
        print('No physiological noise removal')
        return None
    else:
        raise ValueError('Physiological noise removal not recognized, please check your input')
    
def classifier_function(classifier:str):
    if classifier == 'SVM':
        return SVC()
    elif classifier == 'LDA':
        print('We should be doing some LDA here')
    elif classifier == 'KNN':
        print('We should be doing some KNN here')
    elif classifier == 'None':
        print('No classifier')
        return None
    else:
        raise ValueError('Classifier not recognized, please check your input')
    
