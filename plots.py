#%%
import mne
from Tools.data_loaders import *
from Tools.function_wrappers import *
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, xlabel, ylabel, title
import seaborn as sns
import pandas as pd

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
# 
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
#violinplot for pipelines results
df = pd.read_csv('Results/resultsrobFinal.csv', sep=',', header=None)
none = df.iloc[0, 1::2]
bandpass = df.iloc[1, 1::2]
ica = df.iloc[2, 1::2]
bpca = df.iloc[3, 1::2]
regression = df.iloc[4, 1::2]
tddr = df.iloc[5, 1::2]
wiener = df.iloc[6, 1::2]
spline = df.iloc[7, 1::2]
ica_tddr = df.iloc[8, 1::2]
ica_wiener = df.iloc[9, 1::2]
ica_spline = df.iloc[10, 1::2]
bpca_tddr = df.iloc[11, 1::2]
bpca_wiener = df.iloc[12, 1::2]
bpca_spline = df.iloc[13, 1::2]
regression_tddr = df.iloc[14, 1::2]
regression_wiener = df.iloc[15, 1::2]
regression_spline = df.iloc[16, 1::2]
baseline = 0.3333

data = [none, bandpass, ica, bpca, regression, tddr, wiener, spline, ica_tddr, ica_wiener, ica_spline, bpca_tddr, bpca_wiener, bpca_spline, regression_tddr, regression_wiener, regression_spline]

sns.violinplot(data=data, scale='width', inner='point', linewidth=1.5, cut=0, bw=0.5, palette='Set2')
plt.show()

#%%
#versus boxplot 
pipelines = ['Control', 'Band pass', 'ICA', 'bPCA', 'SCR', 'TDDR', 'Wiener', 'Spline', 'ICA-TDDR', 'ICA-Wiener', 'ICA-Spline', 'bPCA-TDDR', 'bPCA-Wiener', 'bPCA-Spline', 'SCR-TDDR', 'SCR-Wiener', 'SCR-Spline']
plt.figure(figsize=(10,6))
baseline_line = plt.hlines(baseline, -1, len(data), color='r', linestyles='dashed', label='Baseline Model')
plt.ylim(0,1)
sns.boxplot(data=data, palette='Set2')
locs, labels = plt.xticks()
plt.xticks(locs, pipelines, rotation=45, ha='right')
for loc in locs:
    plt.axvline(x=loc, color='grey', linestyle='--', lw=0.5) 
plt.tight_layout()
plt.legend(handles=[baseline_line], loc='lower right')
plt.title("Pipelines' performances on Rob Luke's dataset")
plt.ylabel('Mean accuracies')
plt.show()







#%%
# For Rigets data
author_id = 7
paradigm = 'DoC'
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

#plotting mean response to imagery task
print('Mean response to imagery task')
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

#%%
#plotting standard fNIRS response image
evoked_dict = {
    "Imagery/HbO": epochs["1"].average(picks="hbo"),
    "Imagery/HbR": epochs["1"].average(picks="hbr"),
    "Control/HbO": epochs["2"].average(picks="hbo"),
    "Control/HbR": epochs["2"].average(picks="hbr"),
}

# Rename channels until the encoding of frequency in ch_name is fixed
for condition in evoked_dict:
    evoked_dict[condition].rename_channels(lambda x: x[:-4])

color_dict = dict(HbO="#AA3377", HbR="b")
styles_dict = dict(Control=dict(linestyle="dashed"), Imagery=dict(linestyle="dotted"))

mne.viz.plot_compare_evokeds(
    evoked_dict, combine="mean", ci=0.95, colors=color_dict, styles=styles_dict)

#%%
#plotting topomaps of right tapping, left tapping and control
topomap_args = dict(extrapolate="local")

evoked_imagery_hbo = epochs["1"].average(picks="hbo")
evoked_imagery_hbr = epochs["1"].average(picks="hbr")
evoked_control_hbo = epochs["2"].average(picks="hbo")
evoked_control_hbr = epochs["2"].average(picks="hbr")
evoked_diff_hbo = mne.combine_evoked([evoked_control_hbo, evoked_imagery_hbo], weights=[1, -1])
evoked_diff_hbr = mne.combine_evoked([evoked_control_hbr, evoked_imagery_hbr], weights=[1, -1])

ts = 9
vlim = (-15, 15)

print('Topomap of HbO and HbR for control')
evoked_control_hbo.plot_topomap(times=ts, vlim=vlim, colorbar=False, **topomap_args)
evoked_control_hbr.plot_topomap(times=ts, vlim=vlim, colorbar=False, **topomap_args)
print('Topomap of HbO and HbR for imagery')
evoked_imagery_hbo.plot_topomap(times=ts, vlim=vlim, colorbar=False, **topomap_args)
evoked_imagery_hbr.plot_topomap(times=ts, vlim=vlim, colorbar=False, **topomap_args)
print('Topomap of HbO and HbR for difference between control and imagery')
evoked_diff_hbo.plot_topomap(times=ts, vlim=vlim, colorbar=False, **topomap_args)
evoked_diff_hbr.plot_topomap(times=ts, vlim=vlim, colorbar=False, **topomap_args)
#%%

times = np.arange(0, 12, 2)
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
print('Response across all channels for imagery')
epochs["1"].average().plot_image(clim=clims)
print('Response across all channels for control')
epochs["2"].average().plot_image(clim=clims)

# print channel names
cuh_data.info['ch_names']

#%%
#violinplot for pipelines results
df = pd.read_csv('Results/results_DoC_healthy.csv', sep=',', header=None)
none = df.iloc[0, 1::6]
bandpass = df.iloc[1, 1::6]
ica = df.iloc[2, 1::6]
bpca = df.iloc[3, 1::2]
regression = df.iloc[4, 1::6]
tddr = df.iloc[5, 1::6]
wiener = df.iloc[6, 1::6]
spline = df.iloc[7, 1::6]
ica_tddr = df.iloc[8, 1::6]
ica_wiener = df.iloc[9, 1::6]
ica_spline = df.iloc[10, 1::6]
bpca_tddr = df.iloc[11, 1::6]
bpca_wiener = df.iloc[12, 1::6]
bpca_spline = df.iloc[13, 1::6]
regression_tddr = df.iloc[14, 1::6]
regression_wiener = df.iloc[15, 1::6]
regression_spline = df.iloc[16, 1::6]
baseline = 0.5

data = [none, bandpass, ica, bpca, regression, tddr, wiener, spline, ica_tddr, ica_wiener, ica_spline, bpca_tddr, bpca_wiener, bpca_spline, regression_tddr, regression_wiener, regression_spline]

sns.violinplot(data=data, scale='width', inner='point', linewidth=1.5, cut=0, bw=0.5, palette='Set2')
plt.show()

#%%
#versus boxplot 
pipelines = ['Control', 'Band pass', 'ICA', 'bPCA', 'SCR', 'TDDR', 'Wiener', 'Spline', 'ICA-TDDR', 'ICA-Wiener', 'ICA-Spline', 'bPCA-TDDR', 'bPCA-Wiener', 'bPCA-Spline', 'SCR-TDDR', 'SCR-Wiener', 'SCR-Spline']
plt.figure(figsize=(10,6))
baseline_line = plt.hlines(baseline, -1, len(data), color='r', linestyles='dashed', label='Baseline Model')
plt.ylim(0,1)
sns.boxplot(data=data, palette='Set2')
locs, labels = plt.xticks()
plt.xticks(locs, pipelines, rotation=45, ha='right')
for loc in locs:
    plt.axvline(x=loc, color='grey', linestyle='--', lw=0.5) 
plt.tight_layout()
plt.legend(handles=[baseline_line], loc='upper right')
plt.title("Pipelines' performances on CUH's dataset (control)")
plt.ylabel('Mean accuracies')
plt.show()

#%%
# boxplot for the results of the patients
Patient1 = df.iloc[:, 1]
Patient2 = df.iloc[:, 7]
Patient3 = df.iloc[:, 13]
Patient4 = df.iloc[:, 19]
Patient5 = df.iloc[:, 25]
Patient6 = df.iloc[:, 31]
Patient7 = df.iloc[:, 37]
Patient8 = df.iloc[:, 43]
Patient9 = df.iloc[:, 49]
Patient10 = df.iloc[:, 55]
Patient11 = df.iloc[:, 61]
Patient12 = df.iloc[:, 67]
Patient13 = df.iloc[:, 73]
Patient14 = df.iloc[:, 79]
Patient15 = df.iloc[:, 85]
Patient16 = df.iloc[:, 91]
Patient17 = df.iloc[:, 97]
Patient18 = df.iloc[:, 103]
Patient19 = df.iloc[:, 109]
Patient20 = df.iloc[:, 115]
Patient21 = df.iloc[:, 121]
Patient22 = df.iloc[:, 127]
Patient23 = df.iloc[:, 133]
Patient24 = df.iloc[:, 139]
Patient25 = df.iloc[:, 145]
Patient26 = df.iloc[:, 151]
Patient27 = df.iloc[:, 157]
Patient28 = df.iloc[:, 163]
Patient29 = df.iloc[:, 169]
Patient30 = df.iloc[:, 175]
Patient31 = df.iloc[:, 181]
Patient32 = df.iloc[:, 187]
Patient33 = df.iloc[:, 193]
Patient34 = df.iloc[:, 199]
Patient35 = df.iloc[:, 205]
Patient36 = df.iloc[:, 211]

data = [Patient1, Patient2, Patient3, Patient4, Patient5, Patient6, Patient7, Patient8, Patient9, Patient10, Patient11, Patient12, Patient13, Patient14, Patient15, Patient16, Patient17, Patient18, Patient19, Patient20, Patient21, Patient22, Patient23, Patient24, Patient25, Patient26, Patient27, Patient28, Patient29, Patient30, Patient31, Patient32, Patient33, Patient34, Patient35, Patient36]

patients = ['Patient 1', 'Patient', 'Patient 3', 'Patient 4', 'Patient 5', 'Patient 6', 'Patient 7', 'Patient 8', 'Patient 9', 'Patient 10', 'Patient 11', 'Patient 12', 'Patient 13', 'Patient 14', 'Patient 15', 'Patient 16', 'Patient 17', 'Patient 18', 'Patient 19', 'Patient 20', 'Patient 21', 'Patient 22', 'Patient 23', 'Patient 24', 'Patient 25', 'Patient 26', 'Patient 27', 'Patient 28', 'Patient 29', 'Patient 30', 'Patient 31', 'Patient 32', 'Patient 33', 'Patient 34', 'Patient 35', 'Patient 36']
plt.figure(figsize=(10,6))
baseline_line = plt.hlines(baseline, -1, len(data), color='r', linestyles='dashed', label='Baseline Model')
plt.ylim(0,1)
sns.boxplot(data=data, palette='Set2')
locs, labels = plt.xticks()
plt.xticks(locs, patients, rotation=45, ha='right')
for loc in locs:
    plt.axvline(x=loc, color='grey', linestyle='--', lw=0.5) 
plt.tight_layout()
plt.legend(handles=[baseline_line], loc='upper right')
plt.title("Patients' mean accuracies (CUH's dataset)")
plt.ylabel('Mean accuracies')
plt.show()

#%%
# boxplot for the results of the participants
Participant1 = df.iloc[:, 1]
Participant2 = df.iloc[:, 7]
Participant3 = df.iloc[:, 13]
Participant4 = df.iloc[:, 19]
Participant5 = df.iloc[:, 25]

data = [Participant1, Participant2, Participant3, Participant4, Participant5]

participants = ['Participant 1', 'Participant 2', 'Participant 3', 'Participant 4', 'Participant 5']
plt.figure(figsize=(10,6))
baseline_line = plt.hlines(baseline, -1, len(data), color='r', linestyles='dashed', label='Baseline Model')
plt.ylim(0,1)
sns.boxplot(data=data, palette='Set2')
locs, labels = plt.xticks()
plt.xticks(locs, participants, rotation=45, ha='right')
for loc in locs:
    plt.axvline(x=loc, color='grey', linestyle='--', lw=0.5)
plt.tight_layout()
plt.legend(handles=[baseline_line], loc='upper right')
plt.title("Participants' mean accuracies (CUH's dataset with healthy partcipants)")
plt.ylabel('Mean accuracies')
plt.show()
