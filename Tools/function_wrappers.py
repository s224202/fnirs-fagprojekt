from scipy.signal import wiener
from scipy.signal import butter, filtfilt
from Tools.Array_transformers import arrayflattener, bPCA
from mne_nirs.signal_enhancement import short_channel_regression
from Scripts.TDDR import TDDR
from Scripts.Spline import motion_artifact_correction
import mne
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neural_network import MLPClassifier
from itertools import compress

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
    X = mne.Epochs(x, events, event_dict, tmin=-5, tmax=15, baseline=(None, 0), preload=True, proj=True, detrend=None,verbose=True)
    return X.get_data()

#  def butter_bandpass_wrapper(x):
    # data = x.copy().get_data()
    # annotaions = x.annotations
    # info = x.info
    # lowcut = 0.01
    # highcut = 0.1
    # nyquist = 0.5 * info['sfreq']
    # low = lowcut / nyquist
    # high = highcut / nyquist
    # b, a = butter(2, [low, high], btype='band')
    # data = filtfilt(b, a, data)
    # filtered_raw = mne.io.RawArray(data, info)
    # filtered_raw.set_annotations(annotaions)
    # return filtered_raw

def bandpass_wrapper(x):
    x.filter(0.05, 0.7, h_trans_bandwidth=0.2, l_trans_bandwidth=0.02)
    return x

def ICA_wrapper(x):
    ica = mne.preprocessing.ICA(n_components=0.8, random_state=42, max_iter='auto')
    ica.fit(x.copy().filter(l_freq = 1, h_freq= None ))
    x = ica.apply(x, exclude=ica.find_bads_muscle(inst=x, l_freq=0.5, h_freq=5)[0])
    return x

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
    data = TDDR(data, info['sfreq'])
    filtered_raw = mne.io.RawArray(data, info)
    filtered_raw.set_annotations(annotaions)
    return filtered_raw


def spline_wrapper(x):
    data = x.copy().get_data()
    annotaions = x.annotations
    info = x.info
    data = motion_artifact_correction(data)
    filtered_raw = mne.io.RawArray(data, info)
    filtered_raw.set_annotations(annotaions)
    return filtered_raw


def bPCA_wrapper(x):
    data = x.copy().get_data()
    annotaions = x.annotations
    info = x.info
    data = bPCA(data, len(data)-2)
    filtered_raw = mne.io.RawArray(data, info)
    filtered_raw.set_annotations(annotaions)
    return filtered_raw


def short_channel_regression_wrapper(x):
    return short_channel_regression(x, max_dist=0.01)

def feature_selection_wrapper(x,labels):
    model = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=10000)
    sfs = SequentialFeatureSelector(model, n_features_to_select='auto', cv=3)
    return sfs.fit_transform(X=x,y=labels)

def bads_wrapper(x):
    scis = mne.preprocessing.nirs.scalp_coupling_index(x)
    x.info['bads'] = list(compress(x.ch_names, scis < 0.8))
    print(x.info['bads'])
    return x.pick(picks=None, exclude=x.info['bads'])

def shorts_wrapper(x):
    picks = mne.pick_types(x.info, meg=False, fnirs=True, exclude='bads')
    dists = mne.preprocessing.nirs.source_detector_distances(x.info, picks=picks)
    longs = [x.ch_names[i] for i in range(len(dists)) if dists[i] > 0.01]
    return x.pick(picks=longs, exclude='bads')