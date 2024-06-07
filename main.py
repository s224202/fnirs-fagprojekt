from Tools.pipeline_builder import build_pipeline
from mne_bids import (BIDSPath,read_raw_bids,print_dir_tree,make_report,find_matching_paths,get_entity_vals)
import numpy as np
from sklearn.preprocessing import LabelEncoder
# Test the pipeline builder
# Get and clean the data, not the final version
sessions = get_entity_vals("./Rob Luke Tapping dataset", "session")
datatype = "nirs"
extension = [".snirf"]
bids_paths = find_matching_paths("./Rob Luke Tapping dataset", datatypes=datatype, extensions=extension)
data = [read_raw_bids(bids_path) for bids_path in bids_paths]
data = data[0]
data = data.pick(picks='all')
data.annotations.set_durations(5)
data.annotations.delete(np.nonzero(data.annotations.description == "15.0"))

# Test the pipeline builder
labels = data.annotations.to_data_frame()['description']
labels = LabelEncoder().fit_transform(labels)
print(len(labels))
pipeline = build_pipeline('None', 'None', 'None', 'SVM')
pipeline.fit(data, labels)
print("Pipeline built successfully")
print(pipeline.predict(data))