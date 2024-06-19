# The annoying and yet necessary ANNOTATIONS RE-ADDER

import mne
from mne_nirs.io import write_raw_snirf
from Tools.data_loaders import load_CUH_data
datas = [load_CUH_data(i, 'DoC') for i in range(1, 8)]
for data in datas:
    data.annotations.set_durations(15)
    data.annotations.append([onset + 15 for onset in data.annotations.onset], 14.0, 2)

for i in range(len(datas)):
    write_raw_snirf(datas[i], f'./Rigshospitalet_dataset/DoC/CORRECTED_2024-04-29_0{i+1}.snirf')