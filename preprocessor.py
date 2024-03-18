# %% 
from itertools import compress
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.decomposition import PCA
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

# Visualize artifacts (only on the first subject) (mne overview of artifacts detection)
raw = read_raw_bids(bids_paths[0])

# Low-frequency drifts

mag_channels0 = mne.pick_types(raw.info, meg="mag")
raw.load_data()
mne.viz.plot_raw(raw, duration=60, n_channels = len(raw.ch_names), scalings="auto")


#%%
def lastCompsPCA(x, n):
    pca = PCA()
    pca.fit(x)
    #print(pca.explained_variance_ratio_.shape)
    X_reduced = pca.transform(x)
    #print(pca.explained_variance_ratio_)
    return pca.inverse_transform(X_reduced)[:,n:]


# # %%
# # Motion Artifact Correction

PCA_MAC = make_pipeline(PCA(n ))


