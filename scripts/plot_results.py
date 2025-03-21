import pandas as pd
import glob
import awkward as ak
import os
from config import get_config
from utils_results import validation_plots_pt_mass, plot_response_vs_pt

base_folder = "/gwpool/users/camagni/Di-tau/pre_processing/"
config = get_config(base_folder)
dataset = config['test_sample']
model_folder = config['model_folder']
pairtype = config["test_pairType"]

results = pd.read_csv(glob.glob(f"{model_folder}/results/{dataset}/results_inference_{pairtype}*.csv")[0])
fastmtt_file = glob.glob(f'/gwpool/users/camagni/Di-tau/fastmtt/TIDAL/Tools/FastMTT/DATA/{dataset}/fastmtt/{pairtype}.parquet')
fastmtt = ak.from_parquet(fastmtt_file)
os.makedirs(config['model_folder']+ '/results/'+ dataset + '/plots', exist_ok=True)
plots_folders = config['model_folder']+ '/results/'+ dataset + '/plots'
validation_plots_pt_mass(results, fastmtt, plots_folders, pairtype)
plot_response_vs_pt(results, config['model_folder']+ '/results/'+ dataset + '/plots/' + f'response_vs_pt_{pairtype}.png')


