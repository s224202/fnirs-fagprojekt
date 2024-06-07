from sklearn.pipeline import Pipeline, FunctionTransformer
from sklearn.svm import SVC
from Tools.function_wrappers import wiener_wrapper, nirs_od_wrapper, nirs_beer_lambert_wrapper, classifier_wrapper, event_splitter_wrapper
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
    estimator_list = []
    systemic_func = systemic_function(systemic)
    if systemic_func is not None:
        estimator_list.append(('systemic', systemic_func))
    motion_func = motion_function(motion)
    if motion_func is not None:
        estimator_list.append(('motion', motion_func))
    estimator_list.append(('nirs_od', FunctionTransformer(nirs_od_wrapper)))
    estimator_list.append(('nirs_beer_lambert', FunctionTransformer(nirs_beer_lambert_wrapper)))
    phys_func = phys_function(phys)
    if phys_func is not None:
        estimator_list.append(('phys', phys_func))
    estimator_list.append(('event_splitter', FunctionTransformer(event_splitter_wrapper)))
    classifier_func = classifier_function(classifier)
    if classifier_func is not None:
        estimator_list.append(('classifier', classifier_func))
    return Pipeline(estimator_list)
    

# Funtions to find the correct function based on the input
def systemic_function(systemic:str):
    if systemic == 'Low pass':
        print('We should be doing some low pass filtering here')
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
        print('We should be doing some ICA here')
    elif motion == 'PCA':
        print('We should be doing some PCA here')
    elif motion == 'Spline':
        print('We should be doing some Spline here')
    elif motion == 'TDDR':
        print('We should be doing some TDDR here')
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
        print('We should be doing some bPCA here')
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