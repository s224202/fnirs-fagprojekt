from scipy.signal import wiener
from Tools.Array_transformers import arrayflattener
import mne

# TODO: Figure out classifier function
def wiener_wrapper(x):
    return wiener(x, mysize=5)

def nirs_od_wrapper(x: mne.io.Raw) -> mne.io.Raw:
    return mne.preprocessing.nirs.optical_density(x)

def nirs_beer_lambert_wrapper(x):
    return mne.preprocessing.nirs.beer_lambert_law(x, ppf=0.1)

def classifier_wrapper(x):
    return 

def event_splitter_wrapper(x):
    events, event_dict = mne.events_from_annotations(x)
    X = mne.Epochs(x, events, event_dict, tmin=-5, tmax=15, baseline=None, preload=True)
    return arrayflattener(X.get_data())