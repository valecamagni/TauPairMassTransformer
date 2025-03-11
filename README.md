# TauPairMassTransformer (TPMT)

## Overview
TauPairMassTransformer (TPMT) is a deep learning model based on transformer architecture, designed to reconstruct the invariant mass of a pair of tau leptons in high-energy physics experiments.

## Repository Structure
The repository is organized as follows:

```
TauPairMassTransformer/

├── scripts/           
│   ├── config.py            # Configuration file for training and inference
│   ├── dataset.py           # Custom PyTorch dataset loader
│   ├── model_TPMT.py        # Transformer model definition (TPMT)
│   ├── model_NF.py          # Normalizing Flow model definition (NF)
│   ├── run_TPMT.py          # Training script for the TPMT model
│   ├── run_NF.py            # Training script for the NF model
│   ├── inference_TPMT.py    # Inference script for the TPMT model
│   ├── inference_NF.py      # Inference script for the NF model
│   ├── export.py            # Export trained model to ONNX format
│   ├── onnx_inference.py    # Run inference using ONNX Runtime
│   ├── utils_train.py       # Utility functions for model training
│   ├── utils_dataset.py     # Utility functions for dataset preparation (train/test/validation splits)
│   ├── utils_model.py       # Layer classes used for model construction
│   ├── utils_results.py     # Utility functions for generating final plots
│   ├── plot_results.py      # Script to generate and save validation plots per sample
│   ├── plot_comparisons.py  # Script to compare model results with FastMTT algorithm

├── preprocessing/           
│   ├── fully_hadronic.py    # Preprocessing pipeline for fully hadronic tau pair decays  
│   ├── semi_leptonic.py     # Preprocessing pipeline for semi-leptonic tau pair decays  
│   ├── submit.py            # Script to submit preprocessing jobs (e.g., batch processing on clusters)  
│   ├── utils.py             # Utility functions for data preprocessing (e.g., data formatting, filtering)  

├── README.md                # Project documentation
└── requirements.txt         # Dependencies
```

## Installation
To set up the environment:

```bash
git clone https://github.com/valecamagni/TauPairMassTransformer.git
cd TauPairMassTransformer
pip install -r requirements.txt
```

## Data sets preparation

The `preprocessing` directory contains scripts to preprocess fully hadronic and semi-leptonic tau decay events from NanoAOD ROOT files, preparing them for training and inference with the TPMT. The preprocessing extracts relevant physics features, applies selection criteria, and formats the data for the model.
Both `fully_hadronic.py` and `semi_leptonic.py` accept several command-line arguments to control the preprocessing behavior:

| Parameter    | Type   | Default  | Description |
|-------------|--------|----------|-------------|
| `-i, --input`  | `str`   | Path to ROOT file | Input NanoAOD file to process. |
| `-f, --flat`   | `int`   | `0`  | Set to `1` for flat mass samples. |
| `-p, --pairType` | `str`  | `"tau_tau"/"ele_tau"` | Decay channel: `tau_tau`, `mu_tau`, or `ele_tau`. |
| `-t, --test`   | `bool`  | `True` | If `True`, saves additional dataset for SVFit/FastMTT algorithm. |
| `-pT, --pTcut` | `float` | `20.0` | pT threshold for tau selection. |


#### Example usage:
```
python fully_hadronic.py -i /path/to/nanoAOD.root -p tau_tau -pT 25.0 -t True
python semi_leptonic.py -i /path/to/nanoAOD.root -p mu_tau -pT 20.0 -t False
```

#### Outputs:

- `{sample}/DATA/{pairType}_{root_file}.npz` processed input NanoAOD
- `{sample}/DATA/SVFit_{pairType}_{root_file}.root` SVFit/FastMTT-compatible inputs as a `.root` file
- `{sample}/LOGS/Log_{pairType}_{root_file}.txt` log file
- `{sample}/PLOTS/gen_reco_invmass_{pairType}_{root_file}.png` invariant mass plot







## Configuration
The model/config.py script defines the training and inference setup for the TPMT. 
It specifies the dataset locations, model hyperparameters (i.e, `batch_size`, `lr`, `embed_dim`, etc...), and file management settings.
It takes in input:
- `base_folder`: path to the directory containing dataset folders
- `data_folders`: list with paths to datasets used for training and validation
- `pair_types`: defines the decay channels considered ($\tau\tau$, $\mu\tau$, $e\tau$)
- `use_tauprod`: if `True` the decay products are passed as input to the model, otherwise not (model structure changes)

It returns a dictionary containing the model configuration, saved in the `model_folder` directory.




## Training
To train the TPMT model, modify the `config.py` file according to your dataset and run:

```bash
python3 run_TPMT.py #run_NF.py to train the Normalizing Flow model
```

The training script will:
- Load datasets from the `DATA/` folders
- Train the transformer model with specified hyperparameters
- Save the model weights and loss plots in the `model_folder` (and `model_folder/weights/`) directory  and a `.txt` with an overview of the number of events used for training and validation divided in subcategory (e.g., `tau_tau`)

## Inference
To perform inference using a trained model, specify the `model_folder`, `weight_epoch`, `test_sample` and `test_pairtype` in the config.py and run:

```bash
python inference_TPMT.py #inference_NF.py to inference with the Normalizing Flow
```

The results, including reco and predicted masses, pt, eta, and phi values, are saved as a CSV file for further analysis in `model_folder/results/sample`.


! Training and Inference scripts takes as optional command-line argument the
`--gpu` parameter that point to the GPU ID to use for running the script. The default value is 0, meaning the first GPU will be used. If a GPU is not available, the script will fall back to CPU.


## Final Plot Generation
The repository provides two scripts for generating plots that analyze model performance and compare different reconstruction methods for $\tau\tau$ invariant mass estimation.

1. `plot_results.py`
This script generates plots to evaluate the regression performance of the model, focusing on the pT of the two tau leptons and on the $m_{\tau\tau}$ reconstruction.
To run the script, specify the following parameters in the `config.py`:
    - `model_folder:` Path to the trained model directory
    - `test_sample`: Dataset used for evaluation
    - `test_pairType`: Type of tau pair used in the analysis.
2. `plot_comparisons.py`
This script produces final comparison plots between different samples and invariant mass reconstruction methods. For each test sample, it generates a comparative plot of TPMT vs. FastMTT distributions, including Gaussian fits for further analysis.


