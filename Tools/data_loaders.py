from mne_bids import (BIDSPath,read_raw_bids,print_dir_tree,make_report,find_matching_paths,get_entity_vals)
import numpy as np

def load_individual(id):
    sessions = get_entity_vals("./Rob Luke Tapping dataset", "session")
    datatype = "nirs"
    extension = [".snirf"]
    bids_paths = find_matching_paths("./Rob Luke Tapping dataset", datatypes=datatype, extensions=extension)
    data = [read_raw_bids(bids_path) for bids_path in bids_paths]
    data = data[id]
    data = data.pick(picks='all')
    data.annotations.set_durations(5)
    data.annotations.delete(np.nonzero(data.annotations.description == "15.0"))
    return data
