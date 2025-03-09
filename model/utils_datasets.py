import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from dataset import DiTauDataset
from pathlib import Path
from config import get_config
import glob



def load_data(file, tauprod_included):
    """
    Load data from a list of files and apply necessary transformations.
    """

    data = np.load(file)
    taus = data['x_taus']
    tar = np.log(data['target'])
    jets = data['x_jets']
    met = data['x_met']
    inv_mass_reco = data['inv_mass_reco']
    inv_mass_reco = np.reshape(inv_mass_reco, (inv_mass_reco.shape[0], 1, 1))
    inv_mass_tar = data['inv_mass_tar']
    tauprod = data['x_tauprod'] if tauprod_included else None

    return taus, tauprod, tar, jets, met, inv_mass_reco, inv_mass_tar



def split_data(indices, val_size=0.2):
    indices_train, indices_val = train_test_split(indices, test_size=val_size, random_state=42)
    return indices_train, indices_val



def append_to_split(split, indices, data):
    """
    Add data to the corresponding split (train, valid).
    Handle 'Dataset' and 'pairType' as lists, and other keys like 'tau' and 'tauprod' by appending the slices.
    """
    for key in data:
        if key in ['dataset', 'pairType']:
            # If it's a list type, extend it with the indexed data
            split[key].extend([data[key][i] for i in indices])
        else:
            # If it's a NumPy array, append the data (concatenation will be handled later)
            split[key].append(data[key][indices])



def prepare_splits(config, tauprod_included):

    input_files = config['Data']
    
    splits = {'train': {}, 'valid': {}}
    datasets = ['train', 'valid']
    keys = ['tau', 'tar', 'jets', 'met', 'inv_mass_reco', 'inv_mass_tar', 'dataset', 'pairType']
    if tauprod_included:keys.append('tauprod')

    for ds in datasets:
        for key in keys:
            splits[ds][key] = []
    
    for num_file, file in enumerate(input_files): 
        if 'test' in file:
            continue
    
        taus, tauprod, tar, jets, met, inv_mass_reco, inv_mass_tar = load_data(file, tauprod_included)

        # Identify dataset and pairtype
        d = file.split('/')[-3]
        if "SUSY" in file:
            d = d[-4:]
        dataset = [d] * taus.shape[0]
        pairtype = ['tau_tau' if 'tau_tau' in file else 'ele_tau' if 'ele_tau' in file else 'mu_tau'] * taus.shape[0]

        assert all(
            len(arr) == taus.shape[0] for arr in [tar, jets, met, inv_mass_reco, inv_mass_tar]
        ) and (tauprod is None or len(tauprod) == taus.shape[0]), "Length mismatch"

        num_events = taus.shape[0]
        indices_train, indices_val = split_data(np.arange(num_events))

        data_dict = {'tau': taus, 'tar': tar, 'jets': jets, 'met': met, 
                     'inv_mass_reco': inv_mass_reco, 'inv_mass_tar': inv_mass_tar,  
                     'dataset': dataset, 'pairType': pairtype}
        if tauprod is not None:
            data_dict['tauprod'] = tauprod

        append_to_split(splits['train'], indices_train, data_dict)
        append_to_split(splits['valid'], indices_val, data_dict)

    for ds in datasets:
        for key in splits[ds].keys():
            to_exlude = ['dataset', 'pairType']
            if key not in to_exlude:
                splits[ds][key] = torch.from_numpy(np.concatenate(splits[ds][key], axis=0))
    
    return splits['train'], splits['valid']



def occurrences(dataset_pair, dataset_sample):

    unique_datasets = set(dataset_sample)
    unique_pairs = set(dataset_pair)

    counts = {dataset: {'total': 0} for dataset in unique_datasets}
    for dataset in counts:
        for pair in unique_pairs:
            counts[dataset][pair] = 0

    for pair, dataset in zip(dataset_pair, dataset_sample):
        counts[dataset]['total'] += 1  
        counts[dataset][pair] += 1  
    return counts



def print_statistics(data, dataset_name, file_output=None):
    counts = occurrences(data['pairType'], data['dataset'])
    
    output = f"{dataset_name}\n \t N. Events: {len(data['tau'])}\n"  
    
    for dataset, pair_types in counts.items():
        output += f"{dataset}: {pair_types['total']}\n"
        for pair_type, count in pair_types.items():
            if pair_type != 'total':  
                output += f"\t {pair_type}: {count}\n"
    if file_output:
        file_output.write(output)
    else:
        print(output)





def equalize_datasets_inplace(train, tauprod_included, dyh_ratio):
    dataset_sample = train['dataset']
    pair_sample = train['pairType']
    
    # Count occurrences for each dataset and pairType
    counts = occurrences(pair_sample, dataset_sample)
    
    # Find the minimum number of events between all the data sets (excluding DY)
    min_count = min([counts[dataset]['total'] for dataset in counts if dataset != "DY"])
    
    # Set the maximum number of events for DY as 3/5 of the min count
    dy_max_count = int(min_count*dyh_ratio)

    indices_to_keep = []

    for dataset in counts:
        dataset_mask = np.array(dataset_sample) == dataset  
        dataset_indices = np.where(dataset_mask)[0]  
        
        # Max event to consider
        if dataset == "DYJetsToLL_M-50-madgraphMLM":
            max_count = min(len(dataset_indices), dy_max_count)  # For "DY" use dy_max_count as maximum 
        else:
            max_count = min(len(dataset_indices), min_count)  # For the other datasets use min_count as maximum
        
        # Select max_count events randomly (without replacing)
        if len(dataset_indices) > max_count:
            sampled_indices = np.random.choice(dataset_indices, max_count, replace=False)
        else:
            sampled_indices = dataset_indices
        
        indices_to_keep.extend(sampled_indices)
    
    indices_to_keep = np.array(indices_to_keep)
    train['tau'] = train['tau'][indices_to_keep]
    train['tar'] = train['tar'][indices_to_keep]
    if tauprod_included:
        train['tauprod'] = train['tauprod'][indices_to_keep]
    train['inv_mass_reco'] = train['inv_mass_reco'][indices_to_keep]
    train['inv_mass_tar'] = train['inv_mass_tar'][indices_to_keep]
    train['dataset'] = np.array(train['dataset'])[indices_to_keep].tolist()  
    train['pairType'] = np.array(train['pairType'])[indices_to_keep].tolist()

    return train





def get_ds(config):
    tauprod_included = config["use_tauprod"]
    train, valid = prepare_splits(config, tauprod_included)
    train_data = equalize_datasets_inplace(train, tauprod_included, config['DYH_ratio'])
    valid_data = equalize_datasets_inplace(valid, tauprod_included, config['DYH_ratio'])

    output_file = config['model_folder'] + '/output_data_size.txt'
    with open(output_file, 'w') as file_output:
        print_statistics(train_data, 'Train', file_output)
        print_statistics(valid_data, 'Validation', file_output)
        
    train_ds = DiTauDataset(train_data)
    valid_ds = DiTauDataset(valid_data)

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    valid_dataloader = DataLoader(valid_ds, batch_size=config['batch_size'], shuffle=False)

    return train_dataloader, valid_dataloader



# FOR TEST

def get_test_ds(base_folder):
    config = get_config(base_folder)
    tauprod_included = config['use_tauprod']
    dataset = config['test_sample']
    pairType = config['test_pairType']
    file = glob.glob(f"{base_folder}{dataset}/DATA/test/{pairType}*.npz")[0]
    print(f"Test dataset: {file}")

    # Load the datasets
    taus, tauprod, tar, jets, met, inv_mass_reco, inv_mass_tar = load_data(file, tauprod_included)

    dataset = [dataset] * taus.shape[0]
    pairtype = [pairType] * taus.shape[0]

    data_dict = {'tau': taus, 'tar': tar, 'jets': jets, 'met': met, 
                 'inv_mass_reco': inv_mass_reco, 'inv_mass_tar': inv_mass_tar,  
                 'dataset': dataset, 'pairType': pairtype}
    if tauprod is not None:
        data_dict['tauprod'] = tauprod

    splits = {'test': {}}

    for key in data_dict.keys():
        splits['test'][key] = []

    append_to_split(splits['test'], np.arange(taus.shape[0]), data_dict)

    for key in splits['test'].keys():
        if key not in ['dataset', 'pairType']:
            splits['test'][key] = torch.from_numpy(np.concatenate(splits['test'][key], axis=0))
    
    print_statistics(splits['test'], 'Test')

    ds = DiTauDataset(splits['test'])
    test_dataloader = DataLoader(ds, batch_size=1, shuffle=False)

    return test_dataloader, dataset

