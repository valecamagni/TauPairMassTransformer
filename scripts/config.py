from pathlib import Path

def get_config(base_folder):
    """
    Return the model configuration for the training/inference setup.

    Args:
        base_folder (str): base folder to data sets.

    Returns:
        dict: Model configuration.
    """

    data_folders = [
        Path(base_folder) / "GluGluHToTauTau_M125" / "DATA",
        Path(base_folder) / "DYJetsToLL_M-50-madgraphMLM" / "DATA",
        #Path(base_folder) / "SUSYBBH2Tau_M140" / "DATA",
        #Path(base_folder) / "SUSYBBH2Tau_M160" / "DATA",
        #Path(base_folder) / "SUSYBBH2Tau_M180" / "DATA",
        #Path(base_folder) / "SUSYBBH2Tau_M200" / "DATA",
        #Path(base_folder) / "SUSYBBH2Tau_M250" / "DATA",
        #Path(base_folder) / "SUSYBBH2Tau_M300" / "DATA",
        Path(base_folder) / "SUSYBBH2Tau_M350" / "DATA",
        #Path(base_folder) / "SUSYGGH2Tau_2HDM_M140" / "DATA",
        #Path(base_folder) / "SUSYGGH2Tau_2HDM_M160" / "DATA",
        #Path(base_folder) / "SUSYGGH2Tau_2HDM_M180" / "DATA",
        #Path(base_folder) / "SUSYGGH2Tau_2HDM_M200" / "DATA",
        #Path(base_folder) / "SUSYGGH2Tau_2HDM_M250" / "DATA",
        #Path(base_folder) / "SUSYGGH2Tau_2HDM_M300" / "DATA",
        Path(base_folder) / "SUSYGGH2Tau_2HDM_M350" / "DATA",
        Path(base_folder) / "TTToHadronic" / "DATA",
        Path(base_folder) / "TTToSemiLeptonic" / "DATA",
    ]
    pair_types = ["mu_tau"]
    use_tauprod = True
    num_categories = 5 if use_tauprod else 4 # tau, tauprod, met, mass, jet
    num_objects = 17 if use_tauprod else 7  # 2 taus + 3 jets + 1 met + 1 mass + 10 tauprods 

    return {
        "batch_size": 256,
        "Data": extract_files_for_train(pair_types, data_folders),
        "num_epochs": 300,
       "lr": 1e-4,
        "tau_loss_weight": 1.0,
        "mass_loss_weight": 1e-2,
        "DYH_ratio": 5/9,  
        "early_stopping_patience": 15,
        "patience_RLR": 10,  
        "use_tauprod": use_tauprod, 
        "num_categories": num_categories,
        "num_objects": num_objects, 
        "input_features_tauprod": 10, 
        "input_features_taus": 18,
        "input_features_jets": 4,
        "input_features_met": 8,
        "input_features_mass": 1,    
        "out_features": 2,  # (logpt_gen) x 2
        "embed_dim": [128, 512, 128], 
        # for NFM: "embed_dim": [128, 512, 128 - num_categories],
        "embed_int_dim": [64, 64, 64, 8],
        "weight_epoch": 76,  
        "test_sample": "GluGluHToTauTau_M125",  
        "test_pairType": "tau_tau",
        "model_folder": "allinclusive",
        "model_basename": "tmodel_",
        "preload": None,
   }


def get_weights_file_path(config, epoch: str): 
    """
    Returns the full path to the model weights file for a given epoch.

    Args:
        config (dict): Configuration dictionary containing 'model_folder' and 'model_basename'.
        epoch (str): String representing the current epoch.

    Returns:
        str: Full path to the weights file.
    """

    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt" 
    return str(Path(__file__).parent / model_folder / "weights" / model_filename)



def extract_files_for_train(pairType, folders):
    """
    Extracts the list of file paths to load for training.

    Args:
        pairType (list): List of pair types to be considered. Default is None.
        folders (list): List of directories to search in. Default is None.

    Returns:
        list: List of sorted file paths.
    """

    filepath_list = []
    for dataset in folders:
        for type in pairType:
            pattern = f"{type}*.npz"
            filepath_list.extend([str(p) for p in Path(dataset).glob(pattern)])

    return filepath_list

