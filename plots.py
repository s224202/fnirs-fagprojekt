#%%
import mne
from Tools.data_loaders import *
from Tools.function_wrappers import *
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, xlabel, ylabel, title
import seaborn as sns

#%%

# For Rob Luke Dataset

#load individual data
id = 0
id_data = load_individual(id)

#transform individual data
id_data = nirs_od_wrapper(id_data)

#%%

#resample the data to 3 Hz and show artifacts
new_anno = mne.Annotations([30, 185, 320, 440],[15,15,15,15],["Spike movement","Spike movement", "Spike movement", "Spike movement"])
id_data_resampled = id_data.copy().resample(3,npad='auto')
id_data_resampled.set_annotations(new_anno).plot(n_channels=28,duration=500,show_scrollbars=False,clipping=None)
# 
#%%
    
#plotting of the optical density before the removal of bad channels
id_data.plot(n_channels=56,duration=4000,show_scrollbars=False,clipping=None)

#plotting of the optical density after the removal of bad channels
sci = mne.preprocessing.nirs.scalp_coupling_index(id_data)

id_data_bads = id_data.info["bads"] = list(compress(id_data.ch_names, sci < 0.8))
id_data_bads[0].plot(n_channels=56,duration=4000,show_scrollbars=False,clipping=None)

#%%
#plotting overview of channels' sci
fig, ax = plt.subplots(layout="constrained")
ax.set_title("Participant 1 (Rob Luke Dataset) - Scalp Coupling Index Distribution")
ax.hist(sci)
ax.xaxis.set_major_locator(plt.MultipleLocator(0.1))
ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
ax.yaxis.grid(True)
ax.set(xlabel="Scalp Coupling Index", ylabel="Count", xlim=[0, 1])
fig.show()

print('Number of bad channels removed (subject 0):', len(id_data.info['bads']))
print(id_data.info['bads'])
#%%

#plotting montage, Check if the montage is correct
id_data.plot_sensors(ch_type='fnirs_od')
#%%

#transformation to hemoglobin concentration
id_data = nirs_beer_lambert_wrapper(id_data)

#plotting of the hemoglobin levels
id_data.plot(n_channels=56,duration=4000,show_scrollbars=False,clipping=None)

#%%

#plotting psd before transitional bandpass filter
id_data.compute_psd().plot()

#%%

#plotting psd after transitional bandpass filter
id_data_bandpass = bandpass_wrapper(id_data)
id_data_bandpass.compute_psd().plot()

#%%

# Manual epoch extraction
events, event_dict = mne.events_from_annotations(id_data)
epochs = mne.Epochs(
    id_data,
    events,
    event_id=event_dict,
    tmin=-5,
    tmax=15,
    reject_by_annotation=True,
    proj=True,
    baseline=(None, 0),
    preload=True,
    detrend=None,
    verbose=True)

#plotting mean response to tapping task
epochs["Tapping"].plot_image(
    combine="mean",
    vmin=-30,
    vmax=30,
    ts_args=dict(ylim=dict(hbo=[-15, 15], hbr=[-15, 15])))

#plotting mean response to control task
epochs["Control"].plot_image(
    combine="mean",
    vmin=-30,
    vmax=30,
    ts_args=dict(ylim=dict(hbo=[-15, 15], hbr=[-15, 15])))

#%%
#plotting standard fNIRS response image
evoked_dict = {
    "Tapping/HbO": epochs["Tapping"].average(picks="hbo"),
    "Tapping/HbR": epochs["Tapping"].average(picks="hbr"),
    "Control/HbO": epochs["Control"].average(picks="hbo"),
    "Control/HbR": epochs["Control"].average(picks="hbr"),
}

# Rename channels until the encoding of frequency in ch_name is fixed
for condition in evoked_dict:
    evoked_dict[condition].rename_channels(lambda x: x[:-4])

color_dict = dict(HbO="#AA3377", HbR="b")
styles_dict = dict(Control=dict(linestyle="dashed"))

mne.viz.plot_compare_evokeds(
    evoked_dict, combine="mean", ci=0.95, colors=color_dict, styles=styles_dict)

#%%
#plotting topomaps of right tapping, left tapping and control
topomap_args = dict(extrapolate="local")

evoked_left_hbo = epochs["Tapping/Left"].average(picks="hbo")
evoked_left_hbr = epochs["Tapping/Left"].average(picks="hbr")
evoked_right_hbo = epochs["Tapping/Right"].average(picks="hbo")
evoked_right_hbr = epochs["Tapping/Right"].average(picks="hbr")
evoked_control_hbo = epochs["Control"].average(picks="hbo")
evoked_control_hbr = epochs["Control"].average(picks="hbr")
evoked_diff_hbo = mne.combine_evoked([evoked_left_hbo, evoked_left_hbo], weights=[1, -1])
evoked_diff_hbr = mne.combine_evoked([evoked_left_hbr, evoked_left_hbr], weights=[1, -1])

ts = 9.0
vlim = (-8, 8)

print('Topomap of HbO and HbR for left tapping')
evoked_left_hbo.plot_topomap(times=ts, vlim=vlim, colorbar=False, **topomap_args)
evoked_left_hbr.plot_topomap(times=ts, vlim=vlim, colorbar=False, **topomap_args)
print('Topomap of HbO and HbR for right tapping')
evoked_right_hbo.plot_topomap(times=ts, vlim=vlim, colorbar=False, **topomap_args)
evoked_right_hbr.plot_topomap(times=ts, vlim=vlim, colorbar=False, **topomap_args)
print('Topomap of HbO and HbR for control task')
evoked_control_hbo.plot_topomap(times=ts, vlim=vlim, colorbar=False, **topomap_args)
evoked_control_hbr.plot_topomap(times=ts, vlim=vlim, colorbar=False, **topomap_args)
print('Topomap of HbO and HbR for difference between left and right tapping')
evoked_diff_hbo.plot_topomap(times=ts, vlim=vlim, colorbar=False, **topomap_args)
evoked_diff_hbr.plot_topomap(times=ts, vlim=vlim, colorbar=True, **topomap_args)
#%%

#plotting joint topomap of HbO and HbR for left tapping
times = np.arange(0, 12, 2)
print('Topomap of HbO and HbR for left tapping')
evoked_left_hbo.plot_joint(
    times=times, topomap_args=topomap_args
)
evoked_left_hbr.plot_joint(
    times=times, topomap_args=topomap_args
)

#plotting joint topomap of HbO and HbR for right tapping
print('Topomap of HbO and HbR for right tapping')
evoked_right_hbo.plot_joint(
    times=times, topomap_args=topomap_args
)
evoked_right_hbr.plot_joint(
    times=times, topomap_args=topomap_args
)

#plotting joint topomap of HbO and HbR for control task
print('Topomap of HbO and HbR for control task')
evoked_control_hbo.plot_joint(
    times=times, topomap_args=topomap_args
)
evoked_control_hbr.plot_joint(
    times=times, topomap_args=topomap_args
)

#plotting joint topomap of HbO and HbR for difference between left and right tapping
# print('Topomap of HbO and HbR for difference between left and right tapping')
# evoked_diff_hbo.plot_joint(
#     times=times, topomap_args=topomap_args
# )
# evoked_diff_hbr.plot_joint(
#     times=times, topomap_args=topomap_args
# )


#%%


#TODO:visualize response of hemoglobin levels to the task(both tapping and toungue)

#TODO:visualize response of hemoglobin levels to control task

#TODO:visualize response of hemoglobin levels to imagery task


