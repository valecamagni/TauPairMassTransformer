import pandas as pd
import glob
import awkward as ak
import numpy as np
import os
import matplotlib.pyplot as plt
from config import get_config
from utils_results import validation_plots_pt_mass, plot_response_vs_pt

base_folder = "/gwpool/users/camagni/Di-tau/pre_processing/"
config = get_config(base_folder)
dataset = config['test_sample']
model_folder = config['model_folder']
pairtype = config["test_pairType"]

#results = pd.read_csv(glob.glob(f"{model_folder}/results/{dataset}/results_inference_{pairtype}*.csv")[0])
results = pd.read_csv('/gwpool/users/camagni/TauPairMassTransformer/preprocessing/DYto2L_M_50_amcatnloFXFX/testDY.csv')
#fastmtt_file = glob.glob(f'/gwpool/users/camagni/Di-tau/fastmtt/TIDAL/Tools/FastMTT/DATA/{dataset}/fastmtt/{pairtype}.parquet')
#fastmtt = ak.from_parquet(fastmtt_file)
#os.makedirs(config['model_folder']+ '/results/'+ dataset + '/plots', exist_ok=True)
#plots_folders = config['model_folder']+ '/results/'+ dataset + '/plots'
#validation_plots_pt_mass(results, fastmtt, plots_folders, pairtype)
#plot_response_vs_pt(results, config['model_folder']+ '/results/'+ dataset + '/plots/' + f'response_vs_pt_{pairtype}.png')


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Dati
data = results['pred_mass']

# Parametri del fit gaussiano
mu, std = norm.fit(data)

# Istogramma
plt.figure(figsize=(10, 8), dpi=150)
n, bins, _ = plt.hist(data, histtype='step', linewidth=2.0, bins=100, range=(0, 250))

# Calcolo del fit gaussiano (non normalizzato)
x = np.linspace(0, 250, 1000)
bin_width = bins[1] - bins[0]
scaling = len(data) * bin_width  # scala per la stessa area dell’istogramma
p = norm.pdf(x, mu, std) * scaling

# Plot della gaussiana
plt.plot(x, p, 'r--', linewidth=2.0, label=f'Guassian Fit\n$\mu={mu:.1f}$, $\sigma={std:.1f}$')

# Etichette e stile
plt.xlabel(r'$m_{\tau\tau}$', fontsize=18, loc='right')
plt.ylabel('Frequency', fontsize=18)
plt.title(r'DYto2L_M_50_amcatnloFXFX - $e\tau$', loc='right', fontsize=20)
plt.xticks(np.arange(0, 251, 20))
plt.grid(True)
plt.legend(fontsize = 14)

# Salvataggio
plt.savefig('/gwpool/users/camagni/TauPairMassTransformer/preprocessing/DYto2L_M_50_amcatnloFXFX/testmass_DY.png')



