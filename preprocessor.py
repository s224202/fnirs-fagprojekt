# %% 
from scipy.signal import find_peaks
import pywt
from itertools import compress
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from scipy.interpolate import CubicSpline
from sklearn.pipeline import make_pipeline
from mne.datasets import sample
import mne
from mne_bids import (
    BIDSPath,
    read_raw_bids,
    print_dir_tree,
    make_report,
    find_matching_paths,
    get_entity_vals,
)
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

short_channels0 = raw0.copy().pick(picks0[dists0 <= 0.01])
short_channels1 = raw1.copy().pick(picks1[dists1 <= 0.01])
short_channels2 = raw2.copy().pick(picks2[dists2 <= 0.01])
short_channels3 = raw3.copy().pick(picks3[dists3 <= 0.01])
short_channels4 = raw4.copy().pick(picks4[dists4 <= 0.01])

# Long channels

long_channels0 = raw0.copy().pick(picks0[dists0 > 0.01])
long_channels1 = raw1.copy().pick(picks1[dists1 > 0.01])
long_channels2 = raw2.copy().pick(picks2[dists2 > 0.01])
long_channels3 = raw3.copy().pick(picks3[dists3 > 0.01])
long_channels4 = raw4.copy().pick(picks4[dists4 > 0.01])
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

# Motion Artifact Correction

r = 42

# PCA last comps.
def lastCompsPCA(x, n):
    pca = PCA()
    pca.fit(x)
    #print(pca.explained_variance_ratio_.shape)
    X_reduced = pca.transform(x)
    #print(pca.explained_variance_ratio_)
    return pca.inverse_transform(X_reduced)[:,n:]


lastCompsPCATransformer1 = FunctionTransformer(lastCompsPCA, kw_args={'n': 39})
lastCompsPCATransformer2 = FunctionTransformer(lastCompsPCA, kw_args={'n': 38})

# print(lastCompsPCATransformer1.fit_transform(long_channels0.get_data()).shape)

lastCompsPCALogisticPipeline1 = make_pipeline(lastCompsPCATransformer1, LogisticRegression(random_state = r))
lastCompsPCALogisticPipeline2 = make_pipeline(lastCompsPCATransformer2, LogisticRegression(random_state = r))

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


# Kalman filtering


# Studentized residuals (outlier detection)
def studentizedResiduals(x, y):
    # Create a polynomial fit and apply the fit to data
    poly_order = 2
    coefs = np.polyfit(x, y, poly_order)
    y_pred = np.polyval(coefs, x)

    # Calculate hat matrix
    X_mat = np.vstack((np.ones_like(x), x)).T
    X_hat = X_mat @ np.linalg.inv(X_mat.T @ X_mat) @ X_mat.T
    hat_diagonal = X_hat.diagonal()

    # Calculate degrees of freedom
    n = len(y)
    dof = n - 3  # Using p = 2 from paper

    # Calculate standardised residuals 
    res = y - y_pred
    sse = np.sum(res ** 2)
    t_res = res * np.sqrt(dof / (sse * (1 - hat_diagonal) - res**2))

    # Return filtered dataframe with the anomalies removed using BC value
    alpha=0.05
    bc_relaxation = 1/6
    bc = student_dist.ppf(1 - alpha / (2 * n), df=dof) * bc_relaxation
    mask = np.logical_and(t_res < bc, t_res > - bc)

    return x[mask]



# %%
plt.plot(long_channels0.get_data()[0], label='Original')
plt.plot(waveletPreprocessor(long_channels0.get_data()[0]), label='Wavelet denoising')
# plt.plot(cubicSplineInterpolation(long_channels0.get_data()), label='Cubic spline interpolation')
plt.plot(wienerPreprocessor(long_channels0.get_data()[0]), label='Wiener filter')
# plt.plot(lastCompsPCA(long_channels0.get_data()[0], 39), label='PCA last comps.')
plt.legend()
plt.show()
# %%
# test
