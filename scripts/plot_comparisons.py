import pandas as pd
import matplotlib.pyplot as plt
import os
import awkward as ak
import numpy as np
import glob
from config import get_config
from utils_results import plot_roc_curve

base_folder="/gwpool/users/camagni/Di-tau/pre_processing/"
config = get_config(base_folder)
model_folder = config['model_folder']
pairType = config['test_pairType']

tpmt_masses = []
reco_masses = []
fastmtt_masses = []
mc_masses = []
folders = []

for folder in os.listdir(model_folder+'/RESULTS/'):
    if not folder.endswith('.png'):
        if (pairType == "tau_tau") & (folder == "TTToSemiLeptonic"): continue
        if (pairType == "ele_tau") & (folder == "TTToHadronic"): continue
        if (pairType == "mu_tau") & (folder == "TTToHadronic"): continue
        folders.append(folder)
        print(folder)
        df = pd.read_csv(glob.glob(f"{model_folder}/RESULTS/{folder}/results_inference_{pairType}*.csv")[0])
        fastmtt_file = glob.glob(f'/gwpool/users/camagni/Di-tau/fastmtt/TIDAL/Tools/FastMTT/DATA/{folder}/fastmtt/{pairType}.parquet')
        fastmtt = ak.from_parquet(fastmtt_file)
        fastmtt_masses.append(fastmtt['FastMTT_mass'])
        tpmt_masses.append(df['pred_mass'])
        reco_masses.append(df['reco_mass'])
        if "SUSY" in folder:
            mc_masses.append([int(folder[-3:])]*df['tar_mass'].shape[0])
        elif "GluGluH" in folder:
            mc_masses.append([125.0]*df['tar_mass'].shape[0])
        elif "DY" in folder:
            mc_masses.append([90.0]*df['tar_mass'].shape[0])
        elif "TTT" in folder:
            mc_masses.append([50.0]*df['tar_mass'].shape[0])

fig = plt.figure(figsize = (6,6))
xmin, xmax, bins = 0, 400, 200
colors = ["blue", "red", "green", "violet", "grey"]
for num, ds in enumerate(tpmt_masses):
    plt.hist(tpmt_masses[num], bins=bins, alpha=0.7, range = (xmin, xmax),  weights=np.ones_like(tpmt_masses[num]) / len(tpmt_masses[num]), histtype='step', label=f'{folders[num]}', color=colors[num], )

plt.xticks(np.linspace(xmin, xmax, 11))
plt.title(r"$m_{\tau\tau}^{TPMT}$ histograms" + f" - {pairType}", fontsize=16)
plt.xlabel(r'$m_{\tau\tau}^{TPMT}$')
plt.ylabel('Frquency')
plt.xlim(xmin, xmax)
plt.ylim(0, 0.2)
plt.legend(loc = "upper left", fontsize = 8)
plt.grid(True)
plt.tight_layout() 
plt.savefig(f'{model_folder}/RESULTS/overall_masses_{pairType}.png')


## ROC curve
plot_roc_curve(np.array(mc_masses[0] + mc_masses[1]),  
               np.array(pd.concat([tpmt_masses[0], tpmt_masses[1]], axis = 0)), 
               np.array(pd.concat([pd.Series(ak.to_numpy(fastmtt_masses[0])), pd.Series(ak.to_numpy(fastmtt_masses[1]))], axis = 0)), f'{model_folder}/RESULTS/', pairType)


#fig = plt.figure(figsize = (6,6))
#xmin, xmax, bins = 0, 400, 200
#colors = ["blue", "red", "green", "violet", "grey"]
#for num, ds in enumerate(tpmt_masses):
#    plt.hist(ak.to_numpy(fastmtt_masses[num]), bins=bins, alpha=0.7, range = (xmin, xmax),  weights=np.ones_like(ak.to_numpy(fastmtt_masses[num])) / len(ak.to_numpy(fastmtt_masses[num])), histtype='step', label=f'{folders[num]}', color=colors[num], )
#
#plt.xticks(np.linspace(xmin, xmax, 11))
#plt.title(r"$m_{\tau\tau}^{FastMTT}$ histograms", fontsize=16)
#plt.xlabel(r'$m_{\tau\tau}^{FastMTT}$')
#plt.ylabel('Frquency')
#plt.xlim(xmin, xmax)
#plt.ylim(0, 0.2)
#plt.legend(loc = "upper left", fontsize = 8)
#plt.grid(True)
#plt.tight_layout() 
#plt.savefig(f'{model_folder}/RESULTS/FastMTT_masses.png')






