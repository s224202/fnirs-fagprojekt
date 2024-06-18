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

#%%
#plotting of the optical density before the removal of bad channels
id_data.plot(n_channels=56,duration=4000,show_scrollbars=False,clipping=None)

#plotting of the optical density after the removal of bad channels
sci = mne.preprocessing.nirs.scalp_coupling_index(id_data)

id_data_bads = id_data.info["bads"] = list(compress(id_data.ch_names, sci < 0.8))
id_data.plot(n_channels=56,duration=4000,show_scrollbars=False,clipping=None)

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
events, event_dict = mne.events_from_annotations(id_data_bandpass)
epochs = mne.Epochs(
    id_data_bandpass,
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
evoked_diff_hbo = mne.combine_evoked([evoked_left_hbo, evoked_right_hbo], weights=[1, -1])
evoked_diff_hbr = mne.combine_evoked([evoked_left_hbr, evoked_right_hbr], weights=[1, -1])

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

#%%
# For Rigets data
author_id = 7
paradigm = 'Healthy'
cuh_data = load_CUH_data(author_id, paradigm)

#transform data
cuh_data = nirs_od_wrapper(cuh_data)

#%%
#resample the data to 3 Hz and show artifacts
cuh_data_resampled = cuh_data.copy().resample(3,npad='auto')
cuh_data_resampled.plot(n_channels=28,duration=200,show_scrollbars=False,clipping=None)

#%%
#plotting of the optical density before the removal of bad channels
plot = cuh_data.plot(n_channels=32,duration=1000,show_scrollbars=False,clipping=None)
plot.show()
#plotting of the optical density after the removal of bad channels
sci = mne.preprocessing.nirs.scalp_coupling_index(cuh_data)

cuh_data_bads = cuh_data.info["bads"] = list(compress(cuh_data.ch_names, sci < 0.8))
cuh_data.plot(n_channels=32,duration=4000,show_scrollbars=False,clipping=None)

#%%
#plotting overview of channels' sci
fig, ax = plt.subplots(layout="constrained")
ax.set_title("Participant 7 (CUH Dataset) - Scalp Coupling Index Distribution")
ax.hist(sci, bins=np.arange(0, 1.1, 0.1),)
ax.xaxis.set_major_locator(plt.MultipleLocator(0.1))
ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
ax.yaxis.grid(True)
ax.set(xlabel="Scalp Coupling Index", ylabel="Count", xlim=[0, 1])
fig.show()

print('Number of bad channels removed (subject 7):', len(cuh_data.info['bads']))
print(cuh_data.info['bads'])
#%%
#plotting montage, Check if the montage is correct
cuh_data.plot_sensors(ch_type='fnirs_od')
#%%
#transformation to hemoglobin concentration
cuh_data = nirs_beer_lambert_wrapper(cuh_data)

#plotting of the hemoglobin levels
cuh_data.plot(n_channels=56,duration=4000,show_scrollbars=False,clipping=None)

#%%
#plotting psd before transitional bandpass filter
cuh_data.compute_psd().plot()

#%%
#plotting psd after transitional bandpass filter
cuh_data_bandpass = bandpass_wrapper(cuh_data)
cuh_data_bandpass.compute_psd().plot()

#%%
# Manual epoch extraction
events, event_dict = mne.events_from_annotations(cuh_data_bandpass)
epochs = mne.Epochs(
    cuh_data_bandpass,
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

#plotting mean response to tongue task
print('Mean response to tongue task')
epochs["1"].plot_image(
    combine="mean",
    vmin=-30,
    vmax=30,
    ts_args=dict(ylim=dict(hbo=[-25, 25], hbr=[-25, 25])))

#plotting mean response to control task
print('Mean response to control task')
epochs["2"].plot_image(
    combine="mean",
    vmin=-30,
    vmax=30,
    ts_args=dict(ylim=dict(hbo=[-25, 25], hbr=[-25, 25])))

#plotting mean response to imagery task
print('Mean response to imagery task')
epochs["3"].plot_image(
    combine="mean",
    vmin=-30,
    vmax=30,
    ts_args=dict(ylim=dict(hbo=[-25, 25], hbr=[-25, 25])))

#%%
# mean responses only with the channels of interest
phys_channels_hbo = mne.pick_channels(ch_names = cuh_data.info['ch_names'], include = ['S5_D4 hbo', 'S5_D12 hbo', 'S6_D5 hbo', 'S6_D13 hbo'])
print('Mean hbo response to tongue task (channel 5 and 6)')
epochs["1"].plot_image(
    combine="mean",
    picks=phys_channels_hbo,
    vmin=-30,
    vmax=30,
    ts_args=dict(ylim=dict(hbo=[-25,25])))

phys_channels_hbr = mne.pick_channels(ch_names = cuh_data.info['ch_names'], include = ['S5_D4 hbr', 'S5_D12 hbr', 'S6_D5 hbr', 'S6_D13 hbr'])
print('Mean hbr response to tongue task (channel 5 and 6)')
epochs["1"].plot_image(
    combine="mean",
    picks=phys_channels_hbr,
    vmin=-30,
    vmax=30,
    ts_args=dict(ylim=dict(hbr=[-25,25])))

imagery_channels_hbo = mne.pick_channels(ch_names = cuh_data.info['ch_names'], include = ['S1_D1 hbo', 'S1_D8 hbo', 'S2_D1 hbo', 'S2_D9 hbo'])
print('Mean hbo response to imagery task (channel 1 and 2)')
epochs["3"].plot_image(
    combine="mean",
    picks=imagery_channels_hbo,
    vmin=-30,
    vmax=30,
    ts_args=dict(ylim=dict(hbo=[-25,25])))

imagery_channels_hbr = mne.pick_channels(ch_names = cuh_data.info['ch_names'], include = ['S1_D1 hbr', 'S1_D8 hbr', 'S2_D1 hbr', 'S2_D9 hbr'])
print('Mean hbr response to imagery task (channel 1 and 2)')
epochs["3"].plot_image(
    combine="mean",
    picks=imagery_channels_hbr,
    vmin=-30,
    vmax=30,
    ts_args=dict(ylim=dict(hbr=[-25,25])))

#%%
#plotting standard fNIRS response image
evoked_dict = {
    "Motor/HbO": epochs["1"].average(picks="hbo"),
    "Motor/HbR": epochs["1"].average(picks="hbr"),
    "Control/HbO": epochs["2"].average(picks="hbo"),
    "Control/HbR": epochs["2"].average(picks="hbr"),
    "Imagery/HbO": epochs["3"].average(picks="hbo"),
    "Imagery/HbR": epochs["3"].average(picks="hbr"),
}

# Rename channels until the encoding of frequency in ch_name is fixed
for condition in evoked_dict:
    evoked_dict[condition].rename_channels(lambda x: x[:-4])

color_dict = dict(HbO="#AA3377", HbR="b")
styles_dict = dict(Control=dict(linestyle="dashed"), Imagery=dict(linestyle="dotted"))

mne.viz.plot_compare_evokeds(
    evoked_dict, combine="mean", ci=0.95, colors=color_dict, styles=styles_dict)

#%%
#plotting standard fnirs response image with only the channels of interest
print('Standard fNIRS response image (channels 5 and 6)')
evoked_dict = {
    "Motor/HbO": epochs["1"].average(picks=phys_channels_hbo),
    "Motor/HbR": epochs["1"].average(picks=phys_channels_hbr),
    "Control/HbO": epochs["2"].average(picks=phys_channels_hbo),
    "Control/HbR": epochs["2"].average(picks=phys_channels_hbr),
    "Imagery/HbO": epochs["3"].average(picks=phys_channels_hbo),
    "Imagery/HbR": epochs["3"].average(picks=phys_channels_hbr),
}

for condition in evoked_dict:
    evoked_dict[condition].rename_channels(lambda x: x[:-4])

color_dict = dict(HbO="#AA3377", HbR="b")
styles_dict = dict(Control=dict(linestyle="dashed"), Imagery=dict(linestyle="dotted"))

mne.viz.plot_compare_evokeds(
    evoked_dict, combine="mean", ci=0.95, colors=color_dict, styles=styles_dict)

print('Standard fNIRS response image (channels 1 and 2)')
evoked_dict = {
    "Motor/HbO": epochs["1"].average(picks=imagery_channels_hbo),
    "Motor/HbR": epochs["1"].average(picks=imagery_channels_hbr),
    "Control/HbO": epochs["2"].average(picks=imagery_channels_hbo),
    "Control/HbR": epochs["2"].average(picks=imagery_channels_hbr),
    "Imagery/HbO": epochs["3"].average(picks=imagery_channels_hbo),
    "Imagery/HbR": epochs["3"].average(picks=imagery_channels_hbr),
}

for condition in evoked_dict:
    evoked_dict[condition].rename_channels(lambda x: x[:-4])

color_dict = dict(HbO="#AA3377", HbR="b")
styles_dict = dict(Control=dict(linestyle="dashed"), Imagery=dict(linestyle="dotted"))

mne.viz.plot_compare_evokeds(
    evoked_dict, combine="mean", ci=0.95, colors=color_dict, styles=styles_dict)

#%%
#plotting topomaps of right tapping, left tapping and control
topomap_args = dict(extrapolate="local")

evoked_motor_hbo = epochs["1"].average(picks="hbo")
evoked_motor_hbr = epochs["1"].average(picks="hbr")
evoked_control_hbo = epochs["2"].average(picks="hbo")
evoked_control_hbr = epochs["2"].average(picks="hbr")
evoked_imagery_hbo = epochs["3"].average(picks="hbo")
evoked_imagery_hbr = epochs["3"].average(picks="hbr")
evoked_diff_hbo = mne.combine_evoked([evoked_control_hbo, evoked_imagery_hbo], weights=[1, -1])
evoked_diff_hbr = mne.combine_evoked([evoked_control_hbr, evoked_imagery_hbr], weights=[1, -1])

ts = 5
vlim = (-15, 15)

print('Topomap of HbO and HbR for motor')
evoked_motor_hbo.plot_topomap(times=ts, vlim=vlim, colorbar=False, **topomap_args)
evoked_motor_hbr.plot_topomap(times=ts, vlim=vlim, colorbar=False, **topomap_args)
print('Topomap of HbO and HbR for control')
evoked_control_hbo.plot_topomap(times=ts, vlim=vlim, colorbar=False, **topomap_args)
evoked_control_hbr.plot_topomap(times=ts, vlim=vlim, colorbar=False, **topomap_args)
print('Topomap of HbO and HbR for imagery')
evoked_imagery_hbo.plot_topomap(times=ts, vlim=vlim, colorbar=False, **topomap_args)
evoked_imagery_hbr.plot_topomap(times=ts, vlim=vlim, colorbar=False, **topomap_args)
print('Topomap of HbO and HbR for difference between control and imagery')
evoked_diff_hbo.plot_topomap(times=ts, vlim=vlim, colorbar=False, **topomap_args)
evoked_diff_hbr.plot_topomap(times=ts, vlim=vlim, colorbar=True, **topomap_args)
#%%

#plotting joint topomap of HbO and HbR for tongue
times = np.arange(0, 12, 2)
print('Topomap of HbO and HbR for motor')
evoked_motor_hbo.plot_joint(
    times=times, topomap_args=topomap_args
)
evoked_left_hbr.plot_joint(
    times=times, topomap_args=topomap_args
)

#plotting joint topomap of HbO and HbR for control
print('Topomap of HbO and HbR for control')
evoked_control_hbo.plot_joint(
    times=times, topomap_args=topomap_args
)
evoked_control_hbr.plot_joint(
    times=times, topomap_args=topomap_args
)

#plotting joint topomap of HbO and HbR for imagery
print('Topomap of HbO and HbR for imagery')
evoked_imagery_hbo.plot_joint(
    times=times, topomap_args=topomap_args
)
evoked_imagery_hbr.plot_joint(
    times=times, topomap_args=topomap_args
)

#%%
#plotting response across all channels
clims = dict(hbo=[-20, 20], hbr=[-20, 20])
print('Response across all channels for tongue')
epochs["1"].average().plot_image(clim=clims)
print('Response across all channels for control')
epochs["2"].average().plot_image(clim=clims)
print('Response across all channels for imagery')
epochs["3"].average().plot_image(clim=clims)

# %%
#plotting mean and sd of responses for all channels (crazy ass plot)
fig, ax = plt.subplots()
ax.set_title("Participant 7 (CUH Dataset) - Mean and Standard Deviation of Responses")
ax.plot(epochs["1"].average().times, epochs["1"].average().data.T, color='blue', alpha=0.1)
ax.plot(epochs["2"].average().times, epochs["2"].average().data.T, color='red', alpha=0.1)
ax.plot(epochs["3"].average().times, epochs["3"].average().data.T, color='green', alpha=0.1)
ax.plot(epochs["1"].average().times, epochs["1"].average().data.mean(axis=0), color='blue', label='Tongue Task')
ax.fill_between(epochs["1"].average().times, epochs["1"].average().data.mean(axis=0) - epochs["1"].average().data.std(axis=0), epochs["1"].average().data.mean(axis=0) + epochs["1"].average().data.std(axis=0), color='blue', alpha=0.5)
ax.plot(epochs["2"].average().times, epochs["2"].average().data.mean(axis=0), color='red', label='Control Task')
ax.fill_between(epochs["2"].average().times, epochs["2"].average().data.mean(axis=0) - epochs["2"].average().data.std(axis=0), epochs["2"].average().data.mean(axis=0) + epochs["2"].average().data.std(axis=0), color='red', alpha=0.5)
ax.plot(epochs["3"].average().times, epochs["3"].average().data.mean(axis=0), color='green', label='Imagery Task')
ax.fill_between(epochs["3"].average().times, epochs["3"].average().data.mean(axis=0) - epochs["3"].average().data.std(axis=0), epochs["3"].average().data.mean(axis=0) + epochs["3"].average().data.std(axis=0), color='green', alpha=0.5)
ax.legend()
ax.set(xlabel='Time (s)', ylabel='Amplitude (a.u.)')
fig.show()

#%%
cuh_data.info