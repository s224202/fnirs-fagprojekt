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
