from mne_bids import (BIDSPath,read_raw_bids,print_dir_tree,make_report,find_matching_paths,get_entity_vals)
import mne
import numpy as np
import os

def load_individual(id):
    sessions = get_entity_vals("./Rob Luke Tapping dataset", "session")
    datatype = "nirs"
    extension = [".snirf"]
    bids_paths = find_matching_paths("./Rob Luke Tapping dataset", datatypes=datatype, extensions=extension)
    data = [read_raw_bids(bids_path, verbose=False) for bids_path in bids_paths]
    data = data[id]
    data = data.pick(picks='all')
    data.annotations.set_durations(5)
    data.annotations.delete(np.nonzero(data.annotations.description == "15.0"))
    return data

def load_author_data(author_id:int):
    '''
    values for author_id
    1:gunnar
    2:signe
    3:viktor
    4:Adam
    '''
    path_name = 'Author_dataset/rec' + f'/2024-05-01_00{author_id}.snirf'
    data = mne.io.read_raw_snirf(path_name)
    return data

def concatenate_data(data_list:list, labels_list:list) -> np.ndarray:
    data = data_list[0]
    labels = labels_list[0]
    for i in range(1, len(data_list)):
        data = np.concatenate((data, data_list[i]), axis=0)
        labels = np.concatenate((labels, labels_list[i]), axis=0)
    return data, labels

def load_CUH_data(author_id:int,paradigm:str):
    '''
    values for author_id
    the numbers between 1-7
    
    values for paradigm
    DoC or Healthy
    '''
    path_name = 'Rigshospitalet_dataset/'+ paradigm + f'/_2024-04-29_0{author_id}.snirf'
    data = mne.io.read_raw_snirf(path_name)
    return data