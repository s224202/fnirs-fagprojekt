from data_loaders import *
from function_wrappers import *

#load individual data
id = int
id_data = []
id_data.append(load_individual(id))

#transform individual data
id_data = nirs_od_wrapper(id_data)

#resample the data to 3 Hz and show artifacts
new_anno = mne.Annotations(
    [450,1075,2150,2650],[50,50,50,50],["Spike","Spike","Movement","Baseline"]
)
(id_data.copy().resample(3,npad="auto").set_annotations(new_anno)).plot(n_channels=28,duration=250,show_scrollbars=False,clipping=None)

