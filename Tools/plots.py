from data_loaders import *
from function_wrappers import *
import mne

#load individual data
id = int
id_data = Raw
id_data.append(load_individual(id))

#transform individual data
id_data = nirs_od_wrapper(id_data)

#resample the data to 3 Hz and show artifacts
new_anno = mne.Annotations([450,1075,2150,2650],[50,50,50,50],["Spike","Spike","Movement","Baseline"])
((id_data.copy().resample(3,npad="auto")).set_annotations(new_anno)).plot(n_channels=28,duration=250,show_scrollbars=False,clipping=None)

#TODO: Implement and plot the SCI values for the data
    
#ploting of the optical density before the removal of bad channels
id_data.plot(n_channels=56,duration=4000,show_scrollbars=False,clipping=None)

#plotting of the optical density after the removal of bad channels
id_data.plot(n_channels=56,duration=4000,show_scrollbars=False,clipping=None,exclude="bads")

#plotting montage, Check if the montage is correct

#hemolevels
id_data = nirs_beer_lambert_wrapper(id_data)

#plotting of the hemoglobin levels
id_data.plot(n_channels=56,duration=4000,show_scrollbars=False,clipping=None)

#plotting psd before transitional bandpass filter
id_data.compute_psd().plot()

#plotting psd after transitional bandpass filter
psdafter = nirs_bandpass_filter_wrapper(id_data).compute_psd().plot()

#TODO:visualize response of hemoglobin levels to the task(both tapping and toungue)

#TODO:visualize response of hemoglobin levels to control task

#TODO:visualize response of hemoglobin levels to imagery task


