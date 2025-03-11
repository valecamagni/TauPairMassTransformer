import pandas as pd
import matplotlib.pyplot as plt
import os
import awkward as ak
import numpy as np
import glob
from config import get_config
from utils_results import plot_roc_curve
from scipy.optimize import curve_fit
from scipy.stats import ks_2samp

base_folder="/gwpool/users/camagni/Di-tau/pre_processing/"
config = get_config(base_folder)
model_folder = config['model_folder']
pairType = config['test_pairType']


if pairType == "ele_tau":
    label = r"$e\tau$"
elif pairType == "mu_tau":
    label = r"$\mu\tau$"
elif pairType == "tau_tau":
    label = r"$\tau\tau$"
    

tar_masses = []
tpmt_masses = []
fastmtt_masses = []
reco_masses = []
mc_masses = []
folders = []

for folder in os.listdir(model_folder+'/results/'):
    if not folder.endswith('.png'):
        if (pairType == "tau_tau") & (folder == "TTToSemiLeptonic"): continue
        if (pairType == "ele_tau") & (folder == "TTToHadronic"): continue
        if (pairType == "mu_tau") & (folder == "TTToHadronic"): continue
        folders.append(folder)
        print(folder)
        df = pd.read_csv(glob.glob(f"{model_folder}/results/{folder}/results_inference_{pairType}*.csv")[0])
        fastmtt_file = glob.glob(f'/gwpool/users/camagni/Di-tau/fastmtt/TIDAL/Tools/FastMTT/DATA/{folder}/fastmtt/{pairType}.parquet')
        fastmtt = ak.from_parquet(fastmtt_file)
        fastmtt_masses.append(fastmtt['FastMTT_mass'])
        tpmt_masses.append(df['pred_mass'])
        reco_masses.append(df['reco_mass'])
        if "SUSY" in folder:
            mc_masses.append([int(folder[-3:])]*df['tar_mass'].shape[0])
            tar_masses.append(df['tar_mass'])
        elif "GluGluH" in folder:
            mc_masses.append([125.0]*df['tar_mass'].shape[0])
            tar_masses.append(df['tar_mass'])
        elif "DY" in folder:
            mc_masses.append([90.0]*df['tar_mass'].shape[0])
            tar_masses.append(df['tar_mass'])
        elif "TTT" in folder:
            mc_masses.append([50.0]*df['tar_mass'].shape[0])
            tar_masses.append(df['tar_mass'])


if pairType == "tau_tau":
    ylim = 0.3
else:
    ylim = 0.2


# TPMT
fig = plt.figure(figsize=(6,6))
xmin, xmax, bins = 0, 400, 200
colors = ["blue", "red", "green", "violet", "grey"]

for num, ds in enumerate(tpmt_masses):
    plt.hist(tpmt_masses[num], bins=bins, alpha=0.7, range=(xmin, xmax),  
             weights=np.ones_like(tpmt_masses[num]) / len(tpmt_masses[num]),  
             histtype='step', label=f'{folders[num]}', color=colors[num])

plt.xticks(np.linspace(xmin, xmax, 11))
plt.title(r"$m_{\tau\tau}^{TPMT}$ histograms" + f" - {label}", fontsize=16, loc = 'right')
plt.xlabel(r'$m_{\tau\tau}^{TPMT}$', fontsize = 14)
plt.ylabel('Frequency', fontsize = 14)
plt.xlim(xmin, xmax)
plt.ylim(0, ylim)
plt.legend(loc="upper center", fontsize=8)
plt.grid(True)

plt.text(0.02, 1.00, r'$\mathbf{CMS}$' + r' $\mathit{Simulation}$', 
         transform=plt.gca().transAxes, fontsize=14, verticalalignment='bottom', 
         horizontalalignment='left')

plt.text(0.02, 0.95, r'Private Work', 
         transform=plt.gca().transAxes, fontsize=14, verticalalignment='bottom', 
         horizontalalignment='left')

plt.tight_layout()  
plt.savefig(f'{model_folder}/results/TPMT_masses_{pairType}.png', dpi=300)


# FastMTT
fig = plt.figure(figsize=(6,6))
xmin, xmax, bins = 0, 400, 200
colors = ["blue", "red", "green", "violet", "grey"]

for num, ds in enumerate(tpmt_masses):
    plt.hist(ak.to_numpy(fastmtt_masses[num]), bins=bins, alpha=0.7, range=(xmin, xmax),  
             weights=np.ones_like(ak.to_numpy(fastmtt_masses[num])) / len(ak.to_numpy(fastmtt_masses[num])),  
             histtype='step', label=f'{folders[num]}', color=colors[num])

plt.xticks(np.linspace(xmin, xmax, 11))
plt.title(r"$m_{\tau\tau}^{FastMTT}$ histograms" + f" - {label}", fontsize=16, loc = 'right')
plt.xlabel(r'$m_{\tau\tau}^{FastMTT}$', fontsize = 14)
plt.ylabel('Frequency', fontsize = 14)
plt.xlim(xmin, xmax)
plt.ylim(0, ylim)
plt.legend(loc="upper right", fontsize=8)
plt.grid(True)

plt.text(0.02, 1.00, r'$\mathbf{CMS}$' + r' $\mathit{Simulation}$', 
         transform=plt.gca().transAxes, fontsize=14, verticalalignment='bottom', 
         horizontalalignment='left')

plt.text(0.02, 0.95, r'Private Work', 
         transform=plt.gca().transAxes, fontsize=14, verticalalignment='bottom', 
         horizontalalignment='left')

plt.tight_layout()  
plt.savefig(f'{model_folder}/results/FastMTT_masses_{pairType}.png', dpi=300)



# MonteCarlo
fig = plt.figure(figsize=(6,6))
xmin, xmax, bins = 0, 400, 200
colors = ["blue", "red", "green", "violet", "grey"]

for num, ds in enumerate(tar_masses):
    plt.hist(ak.to_numpy(tar_masses[num]), bins=bins, alpha=0.7, range=(xmin, xmax),  
             weights=np.ones_like(ak.to_numpy(tar_masses[num])) / len(ak.to_numpy(tar_masses[num])),  
             histtype='step', label=f'{folders[num]}', color=colors[num])

plt.xticks(np.linspace(xmin, xmax, 11))
plt.title(r"$m_{\tau\tau}^{MC}$ histograms" + f" - {label}", fontsize=16, loc = 'right')
plt.xlabel(r'$m_{\tau\tau}^{MC}$', fontsize = 14)
plt.ylabel('Frequency', fontsize = 14)
plt.xlim(xmin, xmax)
plt.legend(loc="upper right", fontsize=8)
plt.grid(True)

plt.text(0.02, 1.00, r'$\mathbf{CMS}$' + r' $\mathit{Simulation}$', 
         transform=plt.gca().transAxes, fontsize=14, verticalalignment='bottom', 
         horizontalalignment='left')

plt.text(0.02, 0.95, r'Private Work', 
         transform=plt.gca().transAxes, fontsize=14, verticalalignment='bottom', 
         horizontalalignment='left')

plt.tight_layout()  
plt.savefig(f'{model_folder}/results/MC_masses_{pairType}.png', dpi=300)




# ROC curve
plot_roc_curve(np.array(mc_masses[0] + mc_masses[1]),  
               np.array(pd.concat([tpmt_masses[0], tpmt_masses[1]], axis = 0)), 
               np.array(pd.concat([pd.Series(ak.to_numpy(fastmtt_masses[0])), pd.Series(ak.to_numpy(fastmtt_masses[1]))], axis = 0)), f'{model_folder}/results/', pairType)




def gaussian(x, A, mu, sigma):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

xmin, xmax, bins = 0, 400, 120
colors = ["blue", "red"]

for i, (tpmt, fastmtt) in enumerate(zip(tpmt_masses, fastmtt_masses)):

    if "M350" in folders[i]:
        where = "center left"
    else:
        where = "upper right"

    fig = plt.figure(figsize=(6,6))

    tpmt = np.array(tpmt)
    fastmtt = np.array(fastmtt)

    hist_tpmt, bin_edges_tpmt = np.histogram(tpmt, bins=bins, range=(xmin, xmax))
    hist_fastmtt, bin_edges_fastmtt = np.histogram(fastmtt, bins=bins, range=(xmin, xmax))

    bin_centers_tpmt = (bin_edges_tpmt[:-1] + bin_edges_tpmt[1:]) / 2
    bin_centers_fastmtt = (bin_edges_fastmtt[:-1] + bin_edges_fastmtt[1:]) / 2

    bin_centers_tpmt = np.array(bin_centers_tpmt)
    hist_tpmt = np.array(hist_tpmt)

    bin_centers_fastmtt = np.array(bin_centers_fastmtt)
    hist_fastmtt = np.array(hist_fastmtt)

    initial_guess_tpmt = [max(hist_tpmt), np.mean(tpmt), np.std(tpmt)]
    initial_guess_fastmtt = [max(hist_fastmtt), np.mean(fastmtt), np.std(fastmtt)]

    popt_tpmt, _ = curve_fit(gaussian, bin_centers_tpmt, hist_tpmt, p0=initial_guess_tpmt)
    popt_fastmtt, _ = curve_fit(gaussian, bin_centers_fastmtt, hist_fastmtt, p0=initial_guess_fastmtt)

    x_fit = np.linspace(xmin, xmax, 300)

    plt.hist(tpmt, bins=bins, alpha=0.7, range=(xmin, xmax), 
             histtype='step', label=r'$m_{\tau\tau}^{TPMT}$', color=colors[0])

    plt.hist(fastmtt, bins=bins, alpha=0.7, range=(xmin, xmax),  
             histtype='step', label=r'$m_{\tau\tau}^{FastMTT}$', color=colors[1])

    plt.plot(x_fit, gaussian(x_fit, *popt_tpmt), linestyle='dashed', color=colors[0], label=f'TPMT Fit (μ={popt_tpmt[1]:.1f}, σ={popt_tpmt[2]:.1f})')
    plt.plot(x_fit, gaussian(x_fit, *popt_fastmtt), linestyle='dashed', color=colors[1], label=f'FastMTT Fit (μ={popt_fastmtt[1]:.1f}, σ={popt_fastmtt[2]:.1f})')

    plt.xticks(np.linspace(xmin, xmax, 11))
    plt.title(f"{folders[i]}" + f" - {label}", fontsize=12, loc='right')
    plt.xlabel(r'$m_{\tau\tau}$', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.xlim(xmin, xmax)
    plt.legend(loc=where, fontsize=10)
    plt.grid(True)

    plt.text(0.02, 1.00, r'$\mathbf{CMS}$' + r' $\mathit{Simulation}$', 
             transform=plt.gca().transAxes, fontsize=12, verticalalignment='bottom', 
             horizontalalignment='left')

    plt.text(0.02, 0.95, r'Private Work', 
             transform=plt.gca().transAxes, fontsize=12, verticalalignment='bottom', 
             horizontalalignment='left')

    plt.tight_layout()
    
    plt.savefig(f'{model_folder}/results/masses_pair_{folders[i]}_{pairType}.png', dpi=300)
    plt.close()



# more deepen analysis
for i, (tpmt, fastmtt) in enumerate(zip(tpmt_masses, fastmtt_masses)):

    if "M350" in folders[i]:
        where = "center left"
    else:
        where = "upper right"

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # Creazione di 3 subplot
    ax1, ax2 = axes

    tpmt = np.array(tpmt)
    fastmtt = np.array(fastmtt)

    # Istogrammi per mass distribution
    hist_tpmt, bin_edges_tpmt = np.histogram(tpmt, bins=bins, range=(xmin, xmax))
    hist_fastmtt, bin_edges_fastmtt = np.histogram(fastmtt, bins=bins, range=(xmin, xmax))

    bin_centers_tpmt = (bin_edges_tpmt[:-1] + bin_edges_tpmt[1:]) / 2
    bin_centers_fastmtt = (bin_edges_fastmtt[:-1] + bin_edges_fastmtt[1:]) / 2

    initial_guess_tpmt = [max(hist_tpmt), np.mean(tpmt), np.std(tpmt)]
    initial_guess_fastmtt = [max(hist_fastmtt), np.mean(fastmtt), np.std(fastmtt)]

    popt_tpmt, _ = curve_fit(gaussian, bin_centers_tpmt, hist_tpmt, p0=initial_guess_tpmt)
    popt_fastmtt, _ = curve_fit(gaussian, bin_centers_fastmtt, hist_fastmtt, p0=initial_guess_fastmtt)

    x_fit = np.linspace(xmin, xmax, 300)

    # Plot 1: Distribuzione delle masse
    ax1.hist(tpmt, bins=bins, alpha=0.7, range=(xmin, xmax), histtype='step', label=r'$m_{\tau\tau}^{TPMT}$', color=colors[0])
    ax1.hist(fastmtt, bins=bins, alpha=0.7, range=(xmin, xmax), histtype='step', label=r'$m_{\tau\tau}^{FastMTT}$', color=colors[1])

    ax1.plot(x_fit, gaussian(x_fit, *popt_tpmt), linestyle='dashed', color=colors[0], label=f'TPMT Fit (μ={popt_tpmt[1]:.1f}, σ={popt_tpmt[2]:.1f})')
    ax1.plot(x_fit, gaussian(x_fit, *popt_fastmtt), linestyle='dashed', color=colors[1], label=f'FastMTT Fit (μ={popt_fastmtt[1]:.1f}, σ={popt_fastmtt[2]:.1f})')

    ax1.set_xlabel(r'$m_{\tau\tau}$', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_xlim(xmin, xmax)
    ax1.legend(loc=where, fontsize=10)
    ax1.grid(True)
    ax1.set_title(f"{folders[i]}" + r" - $m_{\tau\tau}$" + f" - {label}", loc = 'right')

    # KS Test e CDF
    ks_stat, p_value = ks_2samp(tpmt, fastmtt)
    ax2.plot(np.sort(tpmt), np.linspace(0, 1, len(tpmt)), label="TPMT CDF", color=colors[0])
    ax2.plot(np.sort(fastmtt), np.linspace(0, 1, len(fastmtt)), label="FastMTT CDF", color=colors[1])
    ax2.set_xlabel(r'$m_{\tau\tau}$', fontsize=12)
    ax2.set_xlim(0, 400)
    ax2.legend(loc = "center right", fontsize = 10)
    ax2.set_ylabel('CDF', fontsize=12)
    ax2.grid(True)
    ax2.set_title(f"CDF Comparison (KS={ks_stat:.3f}, p={p_value:.3f})", loc = 'right')

    for ax in axes:
        ax.text(0.02, 1.00, r'$\mathbf{CMS}$' + r' $\mathit{Simulation}$', 
                 transform=ax.transAxes, fontsize=12, verticalalignment='bottom', horizontalalignment='left')

        ax.text(0.02, 0.95, r'Private Work', 
                 transform=ax.transAxes, fontsize=12, verticalalignment='bottom', horizontalalignment='left')

    plt.tight_layout()
    
    plt.savefig(f'{model_folder}/results/analysis_{folders[i]}_{pairType}.png', dpi=300)
    plt.close()
