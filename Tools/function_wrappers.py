from scipy.signal import wiener
from scipy.signal import butter, filtfilt
from Tools.Array_transformers import arrayflattener
from mne_nirs.signal_enhancement import short_channel_regression
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
    X = mne.Epochs(x, events, event_dict, tmin=-5, tmax=15, baseline=None, preload=True)
    return arrayflattener(X.get_data())

def butter_bandpass_wrapper(x):
    data = x.copy().get_data()
    annotaions = x.annotations
    info = x.info
    lowcut = 0.01
    highcut = 0.1
    nyquist = 0.5 * info['sfreq']
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(2, [low, high], btype='band')
    data = filtfilt(b, a, data)
    filtered_raw = mne.io.RawArray(data, info)
    filtered_raw.set_annotations(annotaions)
    return filtered_raw

def ICA_wrapper(x):
    data = x.copy().get_data()
    annotaions = x.annotations
    info = x.info
    ica = mne.preprocessing.ICA(n_components=20)
    ica.fit(data)
    data = ica.apply(data)
    filtered_raw = mne.io.RawArray(data, info)
    filtered_raw.set_annotations(annotaions)
    return filtered_raw

def PCA_wrapper(x):
    data = x.copy().get_data()
    annotaions = x.annotations
    info = x.info
    pca = mne.preprocessing.PCA(n_components=20)
    pca.fit(data)
    data = pca.apply(data)
    filtered_raw = mne.io.RawArray(data, info)
    filtered_raw.set_annotations(annotaions)
    return filtered_raw

def TDDR_wrapper(x):
    data = x.copy().get_data()
    annotaions = x.annotations
    info = x.info
    tddr = mne.preprocessing.TDDR()
    tddr.fit(data)
    data = tddr.apply(data)
    filtered_raw = mne.io.RawArray(data, info)
    filtered_raw.set_annotations(annotaions)
    return filtered_raw

def spline_wrapper(x):
    # This one does not work, blame gunnar
    data = x.copy().get_data()
    annotaions = x.annotations
    info = x.info
    spline = mne.preprocessing.Spline()
    spline.fit(data)
    data = spline.apply(data)
    filtered_raw = mne.io.RawArray(data, info)
    filtered_raw.set_annotations(annotaions)
    return filtered_raw

def bPCA_wrapper(x):
    # This one does not work, blame gunnar
    data = x.copy().get_data()
    annotaions = x.annotations
    info = x.info
    bpca = mne.preprocessing.bPCA()
    bpca.fit(data)
    data = bpca.apply(data)
    filtered_raw = mne.io.RawArray(data, info)
    filtered_raw.set_annotations(annotaions)
    return filtered_raw

def short_channel_regression_wrapper(x):
    return short_channel_regression(x, max_dist=0.01)