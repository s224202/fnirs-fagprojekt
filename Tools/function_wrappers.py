from scipy.signal import wiener
from Tools.Array_transformers import arrayflattener
import mne

def wiener_wrapper(x):
    data = x.copy().get_data()
    annotaions = x.annotations
    info = x.info
    data = wiener(data, mysize=5)
    filtered_raw = mne.io.RawArray(data, info)
    filtered_raw.set_annotations(annotaions)
    return filtered_raw

def nirs_od_wrapper(x: mne.io.Raw) -> mne.io.Raw:
    return mne.preprocessing.nirs.optical_density(x)

def nirs_beer_lambert_wrapper(x):
    return mne.preprocessing.nirs.beer_lambert_law(x, ppf=0.1)

def event_splitter_wrapper(x):
    events, event_dict = mne.events_from_annotations(x)
    print(x.copy().get_data().shape)
    X = mne.Epochs(x, events, event_dict, tmin=-5, tmax=15, baseline=None, preload=True)
    return arrayflattener(X.get_data())