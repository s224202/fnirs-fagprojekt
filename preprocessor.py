# %% 
from scipy.signal import find_peaks
import pywt
from itertools import compress
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from scipy.interpolate import CubicSpline
from sklearn.pipeline import make_pipeline
from mne.datasets import sample
from Scripts.TDDR import TDDR
from mne_nirs.signal_enhancement import short_channel_regression
from scipy.stats import studentized_range as student_dist
import mne
from mne_nirs.channels import get_long_channels
from mne_bids import (
    BIDSPath,
    read_raw_bids,
    print_dir_tree,
    make_report,
    find_matching_paths,
    get_entity_vals,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
sessions = get_entity_vals("./Rob Luke Tapping dataset", "session")
datatype = "nirs"
extension = [".snirf"]
bids_paths = find_matching_paths("./Rob Luke Tapping dataset", datatypes=datatype, extensions=extension)
data = [read_raw_bids(bids_path) for bids_path in bids_paths]

# %%
raw0 = data[0].pick(picks="all")
raw1 = data[1].pick(picks="all")
raw2 = data[2].pick(picks="all")
raw3 = data[3].pick(picks="all")
raw4 = data[4].pick(picks="all")
# %%
raw0.annotations.set_durations(5)
raw1.annotations.set_durations(5)
raw2.annotations.set_durations(5)
raw3.annotations.set_durations(5)
raw4.annotations.set_durations(5)

# %%
raw0.annotations.delete(np.nonzero(raw0.annotations.description == "15.0"))
raw1.annotations.delete(np.nonzero(raw1.annotations.description == "15.0"))
raw2.annotations.delete(np.nonzero(raw2.annotations.description == "15.0"))
raw3.annotations.delete(np.nonzero(raw3.annotations.description == "15.0"))
raw4.annotations.delete(np.nonzero(raw4.annotations.description == "15.0"))

# %% 
picks0 = mne.pick_types(raw0.info, meg=False, eeg=False, fnirs=True)
picks1 = mne.pick_types(raw1.info, meg=False, eeg=False, fnirs=True)
picks2 = mne.pick_types(raw2.info, meg=False, eeg=False, fnirs=True)
picks3 = mne.pick_types(raw3.info, meg=False, eeg=False, fnirs=True)
picks4 = mne.pick_types(raw4.info, meg=False, eeg=False, fnirs=True)

# %%

dists0 = mne.preprocessing.nirs.source_detector_distances(raw0.info,picks=picks0)
dists1 = mne.preprocessing.nirs.source_detector_distances(raw1.info,picks=picks1)
dists2 = mne.preprocessing.nirs.source_detector_distances(raw2.info,picks=picks2)
dists3 = mne.preprocessing.nirs.source_detector_distances(raw3.info,picks=picks3)
dists4 = mne.preprocessing.nirs.source_detector_distances(raw4.info,picks=picks4)

# %%

# Short channels
short_channels0 = mne.preprocessing.nirs.short_channels(raw0.info, threshold=0.01)
short_channels1 = mne.preprocessing.nirs.short_channels(raw1.info, threshold=0.01)
short_channels2 = mne.preprocessing.nirs.short_channels(raw2.info, threshold=0.01)
short_channels3 = mne.preprocessing.nirs.short_channels(raw3.info, threshold=0.01)
short_channels4 = mne.preprocessing.nirs.short_channels(raw4.info, threshold=0.01)

short_channels00 = mne_nirs.channels.get_short_channels(raw, max_dist=0.01)
# Long channels
long_channels0 = get_long_channels(raw0, min_dist=0.015, max_dist=0.045)
long_channels1 = get_long_channels(raw1, min_dist=0.015, max_dist=0.045)
long_channels2 = get_long_channels(raw2, min_dist=0.015, max_dist=0.045)
long_channels3 = get_long_channels(raw3, min_dist=0.015, max_dist=0.045)
long_channels4 = get_long_channels(raw4, min_dist=0.015, max_dist=0.045)

#%%

# Visualize the short and long channels (only on the first subject)

brain = mne.viz.Brain("fsaverage", subjects_dir=raw0, background="w", cortex="0.5")
brain.add_sensors(raw0.info, trans="fsaverage", fnirs=["channels", "pairs", "sources", "detectors"],)
brain.show_view(azimuth=20, elevation=60, distance=400)

#%%

# Visualize motion artifacts (only on the first subject)
# Large jumps in light intensities

raw = read_raw_bids(bids_paths[0])
raw.plot(n_channels=56, duration=60)
plt.show()

# Physiological artifacts (heartbeat)
fig, axes = plt.subplots(2, 1, figsize=(15, 10))
long_channels0.plot_psd(axes=axes[0], show=False)
short_channels0.plot_psd(axes=axes[1], show=False)
plt.show()

#%%

# Convert from raw intensity to optical density

# With short and long channels
raw_od0 = mne.preprocessing.nirs.optical_density(raw0)
raw_od1 = mne.preprocessing.nirs.optical_density(raw1)
raw_od2 = mne.preprocessing.nirs.optical_density(raw2)
raw_od3 = mne.preprocessing.nirs.optical_density(raw3)
raw_od4 = mne.preprocessing.nirs.optical_density(raw4)

# Without short channels
raw_od0_long = mne.preprocessing.nirs.optical_density(long_channels0)
raw_od1_long = mne.preprocessing.nirs.optical_density(long_channels1)
raw_od2_long = mne.preprocessing.nirs.optical_density(long_channels2)
raw_od3_long = mne.preprocessing.nirs.optical_density(long_channels3)
raw_od4_long = mne.preprocessing.nirs.optical_density(long_channels4)

#%%

# Evaluating data quality, calculating scalp coupling index (SCI) for all channels (a version of SNR)
sci0 = mne.preprocessing.nirs.scalp_coupling_index(raw_od0)
sci1 = mne.preprocessing.nirs.scalp_coupling_index(raw_od1)
sci2 = mne.preprocessing.nirs.scalp_coupling_index(raw_od2)
sci3 = mne.preprocessing.nirs.scalp_coupling_index(raw_od3)
sci4 = mne.preprocessing.nirs.scalp_coupling_index(raw_od4)

# Plot the SCI values (only on the first subject)

fig, ax = plt.subplots(layout="constrained")
ax.hist(sci0)
ax.set(xlabel="Scalp Coupling Index", ylabel="Count", xlim=[0, 1])

# Plot optical density before removal of bad channels (only on the first subject)
raw_od0.plot(n_channels=90, duration=4000, show_scrollbars=False, clipping=None)


# %%

# Remove bad channels, eg. channels with SCI < 0.8

# With short and long channels
raw_od0.info['bads'] = list(compress(raw_od0.ch_names, sci0 < 0.8))
raw_od1.info['bads'] = list(compress(raw_od1.ch_names, sci1 < 0.8))
raw_od2.info['bads'] = list(compress(raw_od2.ch_names, sci2 < 0.8))
raw_od3.info['bads'] = list(compress(raw_od3.ch_names, sci3 < 0.8))
raw_od4.info['bads'] = list(compress(raw_od4.ch_names, sci4 < 0.8))

# Without short channels
raw_od0_long.info['bads'] = list(compress(raw_od0_long.ch_names, sci0 < 0.8))
raw_od1_long.info['bads'] = list(compress(raw_od1_long.ch_names, sci1 < 0.8))
raw_od2_long.info['bads'] = list(compress(raw_od2_long.ch_names, sci2 < 0.8))
raw_od3_long.info['bads'] = list(compress(raw_od3_long.ch_names, sci3 < 0.8))
raw_od4_long.info['bads'] = list(compress(raw_od4_long.ch_names, sci4 < 0.8))

# print how many bad channels were removed and which ones
print('Number of bad channels removed (subject 0):', len(raw_od0.info['bads']))
print(raw_od0.info['bads'])

# Plot optical density after removal of bad channels (only on the first subject)
raw_od0.plot(n_channels=90, duration=4000, show_scrollbars=False, clipping=None)

# Plot montage (only on the first subject)
raw_od0.plot_sensors()

#%%

# Motion Artifact Correction

r = 42

# bPCA (baseline PCA, last comps.)
def bPCA(x, n):
    pca = PCA()
    pca.fit(x)
    #print(pca.explained_variance_ratio_.shape)
    X_reduced = pca.transform(x)
    #print(pca.explained_variance_ratio_)
    return pca.inverse_transform(X_reduced)[:,n:]


bPCA1 = FunctionTransformer(bPCA, kw_args={'n': 39})
bPCA2 = FunctionTransformer(bPCA, kw_args={'n': 38})

# print(lastCompsPCATransformer1.fit_transform(long_channels0.get_data()).shape)

bPCALogisticPipeline1 = make_pipeline(bPCA1, LogisticRegression(random_state = r))
bPCACALogisticPipeline2 = make_pipeline(bPCA2, LogisticRegression(random_state = r))

# Wiener filter
from scipy.signal import wiener
def wienerPreprocessor(x, n=5):
    return wiener(x, mysize=n)

wienerLogisticPipeline = make_pipeline(FunctionTransformer(wienerPreprocessor), LogisticRegression(random_state = r))

# Cubic spline interpolation 
def cubicSplineInterpolation(x):
    n = x.shape[1]
    t = np.arange(n)
    cs = CubicSpline(t, x, axis = 1)
    return cs(t)

cubicSplineLogisticPipeline = make_pipeline(FunctionTransformer(cubicSplineInterpolation), LogisticRegression(random_state = r))

# Wavelet denoising
def waveletPreprocessor(x, wavelet='db1'):
    cA, cD = pywt.dwt(x, 'bior1.3')
    return cA




# TDDR (needs to be used along with a low-pass filter and sampling frequency above 1 Hz according to the study by Fishburn et al. (2019))
def tddr(signals, sample_rate):
    return TDDR(signals, sample_rate)

# Short-channel regression (mne)
def shortChannelRegression(x):
    return short_channel_regression(x, max_dist=0.01)


# %%
plt.plot(long_channels0.get_data()[0], label='Original')
plt.plot(waveletPreprocessor(long_channels0.get_data()[0]), label='Wavelet denoising')
# plt.plot(cubicSplineInterpolation(long_channels0.get_data()), label='Cubic spline interpolation')
plt.plot(wienerPreprocessor(long_channels0.get_data()[0]), label='Wiener filter')
# plt.plot(lastCompsPCA(long_channels0.get_data()[0], 39), label='PCA last comps.')
plt.legend()
plt.show()
# %%
# Random Forest Classifier

# Assuming X is your feature set and y is your target variable
events, event_dict = mne.events_from_annotations(raw0)
X = mne.Epochs(raw0, events,event_id=event_dict, tmin=0, tmax=15, baseline=None, preload=True).get_data()


def arrayflattener(x):
    Xflat = np.zeros((x.shape[0], x.shape[1]*x.shape[2]))
    for i in range(x.shape[0]):
        Xflat[i] = np.reshape(x[i], (x.shape[1]*x.shape[2]))
    return Xflat
X = arrayflattener(X)
print(X.shape)
# Create a binary target variable for raw0
y = raw0.annotations.to_data_frame()
y = y['description'].to_numpy()

y = LabelEncoder().fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the RandomForestClassifier
clf = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1000, random_state=42)

from sklearn.model_selection import cross_val_score


# Perform cross-validation
scores = cross_val_score(clf, X, y, cv=5)

print(f'Cross-validation scores: {scores}')
print(f'Average score: {scores.mean()}')

#%%
print(X.shape)
# %%
