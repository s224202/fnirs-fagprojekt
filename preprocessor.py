# %% 
from scipy.signal import find_peaks
from scipy.stats import skew
import pywt
from itertools import compress
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
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
from scipy.signal import butter, filtfilt
from mne_nirs.signal_enhancement import short_channel_regression
import mne
from mne_nirs.channels import get_long_channels
from mne.preprocessing.nirs import temporal_derivative_distribution_repair
from mne_bids import (
    BIDSPath,
    read_raw_bids,
    print_dir_tree,
    make_report,
    find_matching_paths,
    get_entity_vals,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SequentialFeatureSelector
sessions = get_entity_vals("./Rob Luke Tapping dataset", "session")
datatype = "nirs"
extension = [".snirf"]
bids_paths = find_matching_paths("./Rob Luke Tapping dataset", datatypes=datatype, extensions=extension)
data = [read_raw_bids(bids_path) for bids_path in bids_paths]

plotting = False
# %%
def arrayflattener(x):
    Xflat = np.zeros((x.shape[0], x.shape[1]*x.shape[2]))
    for i in range(x.shape[0]):
        Xflat[i] = np.reshape(x[i], (x.shape[1]*x.shape[2]))
    return Xflat
# %%
raws = []
for i in range(5):
    raws.append(data[i].pick(picks="all"))
# %%
for i in raws:
    i.annotations.set_durations(5)
# %%
for i in raws:
    i.annotations.delete(np.nonzero(i.annotations.description == "15.0"))
# %% 
picks = []
for i in raws:
    picks.append(mne.pick_types(i.info, meg=False, eeg=False, fnirs=True))
# %%
dists = []
for i in range(5):
    dists.append(mne.preprocessing.nirs.source_detector_distances(raws[i].info,picks=picks[i]))
# %%
#short channels
short_channels = []
for raw in raws:
    short_channels.append(mne.preprocessing.nirs.short_channels(raw.info, threshold=0.01))
# Long channels
long_channels = []
for raw in raws:
    long_channels.append(get_long_channels(raw, min_dist=0.015, max_dist=0.045))

#%%

# Visualize the short and long channels (only on the first subject)
# VIRKER IKKE!!

#subjects_dir = "./Rob Luke Tapping dataset" 

#brain = mne.viz.Brain("fsaverage", subjects_dir = subjects_dir, background="w", cortex="0.5")
#brain.add_sensors(raw0.info, trans="fsaverage", fnirs=["channels", "pairs", "sources", "detectors"],)
#brain.show_view(azimuth=20, elevation=60, distance=400)

#%% Convert from raw intensity to optical density
raw_ods = []
for raw in raws:
    raw_ods.append(mne.preprocessing.nirs.optical_density(raw))

# Without short channels
raw_od_longs = []
for long_channel in long_channels:
    raw_od_longs.append(mne.preprocessing.nirs.optical_density(long_channel))

#%% Resample the data to 3 Hz and show artifacts in plot (only on the first subject)
if plotting:
    raw_od0_resampled = raw_ods[0].copy().resample(3, npad="auto")
    raw_od0_resampled.plot(n_channels=28, duration=250, show_scrollbars=False, clipping=None)

    new_annotations = mne.Annotations(
        [450, 1075, 2150, 2650], [50, 50, 50, 50], ["Spike", "Spike", "Movement", "Baseline"]
    )
    raw_od0_resampled_new_annotations = raw_od0_resampled.set_annotations(new_annotations)
    raw_od0_resampled_new_annotations.plot(n_channels=28, duration=4000, show_scrollbars=False, clipping=None)
#raw_od1_resampled = raw_od1.copy().resample(3, npad="auto")
#raw_od1.plot(n_channels=28, duration=4000, show_scrollbars=False, clipping=None)

#raw_od2_resampled = raw_od2.copy().resample(3, npad="auto")
#raw_od2.plot(n_channels=28, duration=4000, show_scrollbars=False, clipping=None)

#raw_od3_resampled = raw_od3.copy().resample(3, npad="auto")
#raw_od3.plot(n_channels=28, duration=4000, show_scrollbars=False, clipping=None)

#raw_od4_resampled = raw_od4.copy().resample(3, npad="auto")
#raw_od4.plot(n_channels=28, duration=4000, show_scrollbars=False, clipping=None)

#%% Evaluating data quality, calculating scalp coupling index (SCI) for all channels (a version of SNR) (only on the first subject)
scis = []
for raw_od in raw_ods:
    scis.append(mne.preprocessing.nirs.scalp_coupling_index(raw_od))    

# Plot the SCI values (only on the first subject)
if plotting:
    fig, ax = plt.subplots(layout="constrained")
    ax.hist(scis[0])
    ax.set(xlabel="Scalp Coupling Index", ylabel="Count", xlim=[0, 1])

# %%

# The whole plot of optical density before bad channels removal (only on the first subject) (for comparison)
if plotting:
    raw_ods[0].plot(n_channels=56, duration=4000, show_scrollbars=False, clipping=None)

# Remove bad channels, eg. channels with SCI < 0.8

# With short and long channels
for raw_od, sci in zip(raw_ods, scis):
    raw_od.info['bads'] = list(compress(raw_od.ch_names, sci < 0.8))


# Without short channels
for raw_od_long, sci in zip(raw_od_longs, scis):
    raw_od_long.info['bads'] = list(compress(raw_od_long.ch_names, sci < 0.8))

# print how many bad channels were removed and which ones
print('Number of bad channels removed (subject 0):', len(raw_ods[0].info['bads']))
print(raw_ods[0].info['bads'])
# BAD CHANNELS ER IKKE FJERNET; BRUG exclude='bads' OG SØRG FOR DE IKKE ER MED I PREPROCESSING

# Plot optical density after removal of bad channels (only on the first subject)
if plotting:
    raw_od0=raw_ods[0].copy()
    raw_od0.plot(n_channels=56, duration=4000, show_scrollbars=False, clipping=None)

# Plot montage (only on the first subject) VIRKER IKKE!!! ikke rigtigt resultat
    raw_od0.plot_sensors()

#%%

#Standardize the data

# Instrumental Noise Correction

#Low-pass filter

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






# %%
if plotting:
    plt.plot(long_channels[0].get_data()[0], label='Original', alpha=1)
    #plt.plot(waveletPreprocessor(long_channels[0].get_data()[0]), label='Wavelet denoising')
    #plt.plot(cubicSplineInterpolation(long_channels[0].get_data()), label='Cubic spline interpolation')
    plt.plot(bPCA(long_channels[0].get_data(), 38)[0], label='PCA last comps', alpha=0.5)
    plt.plot(wienerPreprocessor(long_channels[0].get_data()[0]), label='Wiener filter', alpha=0.25)
    plt.plot(temporal_derivative_distribution_repair(long_channels[0]).get_data()[0], label='Temporal derivative distribution repair')
    plt.legend()
    plt.show()

#%%
# Converting from optical density to hemoglobin concentration

# With short and long channels
raw_haemos = []
for raw_od in raw_ods:
    raw_haemos.append(mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=0.1))

# Without short channels
raw_haemos_long = []
for raw_od_long in raw_od_longs:
    raw_haemos_long.append(mne.preprocessing.nirs.beer_lambert_law(raw_od_long, ppf=0.1))

oxy_haemos_long = []
deoxy_haemos_long = []
for raw_haemo_long in raw_haemos_long:
    oxy = raw_haemo_long.copy().pick(picks="hbo")
    deoxy = raw_haemo_long.copy().pick(picks="hbr")
    oxy_haemos_long.append(oxy)
    deoxy_haemos_long.append(deoxy)

oxy_haemos = []
deoxy_haemos = []
for raw_haemo in raw_haemos:
    oxy = raw_haemo.copy().pick(picks="hbo")
    deoxy = raw_haemo.copy().pick(picks="hbr")
    oxy_haemos.append(oxy)
    deoxy_haemos.append(deoxy)

# %%
# Unpack data to numpy arrays

#%% Physiological Noise Correction

# Band-pass filter
def bandPassFilter(x, lowcut, highcut, sfreq = data):
    return mne.filter.filter_data(x, lowcut, highcut)

# our sampling frequency from data

# Short-channel regression (mne)
def shortChannelRegression(x):
    return short_channel_regression(x, max_dist=0.01)

# PCA

# ICA

#%%

# Visualize example of heartrate artifacts before and after bandpass filter (only on the first subject)
raw_haemo0_unfiltered = raw_haemos[0].copy()
# BRUG BANDPASS NÅR DEN VIRKER?
# raw_haemo0_filtered = bandPassFilter(raw_haemo0.copy(), 0.05, 0.7)
raw_haemo0_filtered = raw_haemos[0].copy().filter(0.05, 0.7, h_trans_bandwidth = 0.2, l_trans_bandwidth = 0.02) # fra mne
if plotting:
    print("Unfiltered")
    raw_haemo0_unfiltered.compute_psd().plot(average=True, amplitude=False, picks="data", exclude="bads")
    print("Filtered")
    raw_haemo0_filtered.compute_psd().plot(average=True, amplitude=False, picks="data", exclude="bads")

#%%
# This is for channels (we need to check whether it should be implemented, doesnt seem to be in the literature)
# This is stupid
subject_datas = []
for i in range(5):
    events, event_dict = mne.events_from_annotations(raws[i])
    dat = mne.Epochs(long_channels[i], events, event_id=event_dict, tmin=0, tmax=15, baseline=None, preload=True).get_data()
    subject_datas.append(dat)
labels = []
for i in range(5):
    y = long_channels[i].annotations.to_data_frame()
    y = y['description'].to_numpy()
    y = LabelEncoder().fit_transform(y)
    labels.append(y)
# FEATURE SELECTION
K = 10
tol = 1e-3
participants = len(data)
test_MSE = np.zeros((participants, K))
train_MSE = np.zeros((participants, K))
CV = KFold(n_splits=K, shuffle=True, random_state=r)
best_features = [0,0,0,0,0]
best_scores = [0,0,0,0,0]
sfs_features = [[] for _ in range(participants)]
for subject in range(participants):
    subject_data = arrayflattener(subject_datas[subject])
    label = labels[subject]
    for k, (train, test) in enumerate(CV.split(subject_data)):
        best_mse = np.inf
        X_train = subject_data[train]
        X_test = subject_data[test]
        y_train = label[train]
        y_test = label[test]
        for i in range(1, X_train.shape[1]+1):
            sfs = SequentialFeatureSelector(MLPClassifier(hidden_layer_sizes=(10,10,8,5), random_state=r))
            sfs.fit(X_train, y_train)
            model= LogisticRegression(random_state=r)
            model.fit(X_train[:,sfs.get_support()], y_train)
            est_y = model.predict(X_test[:, sfs.get_support()])
            error = mean_squared_error(est_y, y_test)
            improvement = best_mse - error
            if improvement > tol:
                best_mse = error
                test_MSE[subject, k] = error
                train_MSE[subject, k] = mean_squared_error(model.predict(X_train[:, sfs.get_support()]), y_train)
                best_scores[subject] = model.score(X_test[:, sfs.get_support()], y_test)
                best_features[subject] = sfs.get_support()
            else:
                sfs_features[subject] = sfs.get_support()
                break
print("Best features for each subject: " + str(sfs_features))
print("Best scores: " + str(best_scores))
print("Best features: " + str(best_features))
# %%
# Random Forest Classifier
raw0=raws[0].copy() # using the first subject for now
# Assuming X is your feature set and y is your target variable
events, event_dict = mne.events_from_annotations(raw0)
#X = mne.Epochs(raw0, events,event_id=event_dict, tmin=0, tmax=15, baseline=None, preload=True).get_data()
X = mne.Epochs(raw_haemos_long[0], events,event_id=event_dict, tmin=0, tmax=15, baseline=None, preload=True).get_data()


X = arrayflattener(X)
print(X.shape)
# Create a binary target variable for raw0
y = raw_haemos_long[0].annotations.to_data_frame()
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

# %%
# Heuristic for selecting the best features
heurisic_data_oxy = []
heurisic_data_deoxy = []
for oxy, deoxy in zip(oxy_haemos_long, deoxy_haemos_long):
    heurisic_data_oxy.append(mne.Epochs(oxy, events, event_id=event_dict, tmin=0, tmax=15, baseline=None, preload=True).get_data())
    heurisic_data_deoxy.append(mne.Epochs(deoxy, events, event_id=event_dict, tmin=0, tmax=15, baseline=None, preload=True).get_data())

#signal means for oxy and deoxy
means_oxy = []
for person in heurisic_data_oxy:
    per_means = np.zeros((len(person), len(person[0])))
    for i in range(len(person)):
        for j in range(len(person[i])):
            per_means[i][j] = np.mean(person[i][j])
    means_oxy.append(per_means)

means_deoxy = []
for person in heurisic_data_deoxy:
    per_means = np.zeros((len(person), len(person[0])))
    for i in range(len(person)):
        for j in range(len(person[i])):
            per_means[i][j] = np.mean(person[i][j])
    means_deoxy.append(per_means)

means = []
for i in range(len(means_oxy)):
    means.append(np.hstack((means_oxy[i], means_deoxy[i])))


print(means[0].shape, len(means))

# signal peaks for oxy and deoxy
peaks_oxy = []
for person in heurisic_data_oxy:
    per_peaks = np.zeros((len(person), len(person[0])))
    for i in range(len(person)):
        for j in range(len(person[i])):
            per_peaks[i][j] = max(person[i][j])
    peaks_oxy.append(per_peaks)

peaks_deoxy = []
for person in heurisic_data_deoxy:
    per_peaks = np.zeros((len(person), len(person[0])))
    for i in range(len(person)):
        for j in range(len(person[i])):
            per_peaks[i][j] = max(person[i][j])
    peaks_deoxy.append(per_peaks)

peaks = []
for i in range(len(peaks_oxy)):
    peaks.append(np.hstack((peaks_oxy[i], peaks_deoxy[i])))

print(peaks[0].shape, len(peaks))

# signal skewness for oxy and deoxy
skewness_oxy = []
for person in heurisic_data_oxy:
    per_skewness = np.zeros((len(person), len(person[0])))
    for i in range(len(person)):
        for j in range(len(person[i])):
            per_skewness[i][j] = skew(person[i][j])
    skewness_oxy.append(per_skewness)

skewness_deoxy = []
for person in heurisic_data_deoxy:
    per_skewness = np.zeros((len(person), len(person[0])))
    for i in range(len(person)):
        for j in range(len(person[i])):
            per_skewness[i][j] = skew(person[i][j])
    skewness_deoxy.append(per_skewness)

skewness = []
for i in range(len(skewness_oxy)):
    skewness.append(np.hstack((skewness_oxy[i], skewness_deoxy[i])))

print(skewness[0].shape, len(skewness))

# signal slope for oxy and deoxy
slope_oxy = []
for person in heurisic_data_oxy:
    per_slope = np.zeros((len(person), len(person[0])))
    for i in range(len(person)):
        for j in range(len(person[i])):
            per_slope[i][j] = np.polyfit(np.arange(len(person[i][j])), person[i][j], 1)[0]
    slope_oxy.append(per_slope)

slope_deoxy = []
for person in heurisic_data_deoxy:
    per_slope = np.zeros((len(person), len(person[0])))
    for i in range(len(person)):
        for j in range(len(person[i])):
            per_slope[i][j] = np.polyfit(np.arange(len(person[i][j])), person[i][j], 1)[0]
    slope_deoxy.append(per_slope)

slope = []
for i in range(len(slope_oxy)):
    slope.append(np.hstack((slope_oxy[i], slope_deoxy[i])))

print(slope[0].shape, len(slope))

# print example of the heuristics (virker legit?)
print(f'Mean, peak, skewness and slope of the first epoch of the first channel of the first person: {means[0][0][0]}, {peaks[0][0][0]}, {skewness[0][0][0]}, {slope[0][0][0]}')

# other heuristics (could be used)
# kurtosis
# variance
# std
# median

# %%
# SFFS for heuristics
subject_datas = []
for i in range(len(data)):
    subject_datas.append(np.hstack((means[i], peaks[i], skewness[i], slope[i])))

labels = []
for i in range(len(data)):
    y = long_channels[i].annotations.to_data_frame()
    y = y['description'].to_numpy()
    y = LabelEncoder().fit_transform(y)
    labels.append(y)
# FEATURE SELECTION
K = 10
tol = 1e-3
participants = len(data)
test_MSE = np.zeros((participants, K))
train_MSE = np.zeros((participants, K))
CV = KFold(n_splits=K, shuffle=True, random_state=r)
best_scores = [0,0,0,0,0]
sfs_features = [[] for _ in range(participants)]
best_features = [0,0,0,0,0]
for subject in range(participants):
    subject_data = subject_datas[subject]
    label = labels[subject]
    for k, (train, test) in enumerate(CV.split(subject_data)):
        best_mse = np.inf
        X_train = subject_data[train]
        X_test = subject_data[test]
        y_train = label[train]
        y_test = label[test]
        for i in range(1, X_train.shape[1]+1):
            sfs = SequentialFeatureSelector(MLPClassifier(hidden_layer_sizes=(10,10,8,5), random_state=r), n_features_to_select=i)
            sfs.fit(X_train, y_train)
            model= LogisticRegression(random_state=r)
            model.fit(X_train[:,sfs.get_support()], y_train)
            est_y = model.predict(X_test[:, sfs.get_support()])
            error = mean_squared_error(est_y, y_test)
            improvement = best_mse - error
            if improvement > tol:
                best_mse = error
                best_scores[subject] = model.score(X_test[:, sfs.get_support()], y_test)
                test_MSE[subject, k] = error
                train_MSE[subject, k] = mean_squared_error(model.predict(X_train[:, sfs.get_support()]), y_train)
                best_features[subject] = sfs.get_support()
            else:
                sfs_features[subject] = sfs.get_support()
                break
print("Selected features for each subject: " + str(best_features))
print("Best scores: " + str(best_scores))

# %%
oxy_haemos_long[0].get_data().shape

# %% 
if plotting:
    test_data = raw_haemos_long[0].copy()
    test_data.filter(0.05,0.7, h_trans_bandwidth=0.2,l_trans_bandwidth=0.02)

    events, event_dict = mne.events_from_annotations(test_data)
    X = mne.Epochs(test_data, events,event_id=event_dict, tmin=-5, tmax=15, baseline=None, preload=True)
    X["Control"].plot_image(
    combine="mean",
    vmin=-30,
    vmax=30,
    ts_args=dict(ylim=dict(hbo=[-15, 20], hbr=[-15, 15])),
    )
# %%
