import pandas as pd
import glob
import os
from config import get_config
from utils_results import validation_plots_pt_mass, plot_response_vs_pt

base_folder = "/gwpool/users/camagni/Di-tau/pre_processing/"
config = get_config(base_folder)
dataset = config['test_sample']
model_folder = config['model_folder']
pairtype = config["test_pairType"]

results = pd.read_csv(glob.glob(f"{model_folder}/RESULTS/{dataset}/results_inference_{pairtype}*.csv")[0])
os.makedirs(config['model_folder']+ '/RESULTS/'+ dataset + '/PLOTS', exist_ok=True)
plots_folders = config['model_folder']+ '/RESULTS/'+ dataset + '/PLOTS'
validation_plots_pt_mass(results, plots_folders, config['test_pairType'])
plot_response_vs_pt(results, config['model_folder']+ '/RESULTS/'+ dataset + '/PLOTS/' + 'response_vs_pt_plot.png')


