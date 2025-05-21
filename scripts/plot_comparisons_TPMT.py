import pandas as pd
import matplotlib.pyplot as plt
import os
import awkward as ak
import numpy as np
import glob
from config import get_config
from scipy.optimize import curve_fit


def gaussian(x, A, mu, sigma):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


base_folder="/gwpool/users/camagni/DiTau/TauPairMassTransformer/pre_processing/"
config = get_config(base_folder)
model_folder = config['model_folder']
pairType = config['test_pairType']

if pairType == "ele_tau":
    label = r"$e\tau$"
elif pairType == "mu_tau":
    label = r"$\mu\tau$"
elif pairType == "tau_tau":
    label = r"$\tau\tau$"

tpmt_masses = []
reco_masses = []
folders = []

for folder in os.listdir(model_folder+'/results/'):
        if folder.endswith(".png"): continue
        folders.append(folder)
        print(folder)
        df = pd.read_csv(glob.glob(f"{model_folder}/results/{folder}/results_inference_{pairType}*.csv")[0])
        tpmt_masses.append(df['pred_mass'])
        reco_masses.append(df['reco_mass'])



xmin, xmax, bins = 0, 400, 120

for i, tpmt in enumerate(tpmt_masses):

    fig = plt.figure(figsize=(6,6))

    tpmt = np.array(tpmt)

    hist_tpmt, bin_edges_tpmt = np.histogram(tpmt, bins=bins, range=(xmin, xmax))
    bin_centers_tpmt = (bin_edges_tpmt[:-1] + bin_edges_tpmt[1:]) / 2
    bin_centers_tpmt = np.array(bin_centers_tpmt)
    hist_tpmt = np.array(hist_tpmt)
    initial_guess_tpmt = [max(hist_tpmt), np.mean(tpmt), np.std(tpmt)]
    popt_tpmt, _ = curve_fit(gaussian, bin_centers_tpmt, hist_tpmt, p0=initial_guess_tpmt)
    x_fit = np.linspace(xmin, xmax, 400)

    plt.hist(tpmt, bins=bins, alpha=0.7, range=(xmin, xmax), 
             histtype='step', label=r'$m_{\tau\tau}^{TPMT}$', color = 'blue',)

    plt.plot(x_fit, gaussian(x_fit, *popt_tpmt), color = 'blue', linestyle='dashed', label=f'TPMT Fit (μ={popt_tpmt[1]:.1f}, σ={popt_tpmt[2]:.1f})')

    plt.xticks(np.linspace(xmin, xmax, 11))
    plt.legend()
    plt.title(f"{folders[i]}" + f" - {label}", fontsize=12, loc='right')
    plt.xlabel(r'$m_{\tau\tau}$', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.xlim(xmin, xmax)
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



# PREVIOUS PLOTS + CDFs
for i, tpmt in enumerate(tpmt_masses):

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    ax1, ax2 = axes

    tpmt = np.array(tpmt)
    hist_tpmt, bin_edges_tpmt = np.histogram(tpmt, bins=bins, range=(xmin, xmax))
    bin_centers_tpmt = (bin_edges_tpmt[:-1] + bin_edges_tpmt[1:]) / 2
    initial_guess_tpmt = [max(hist_tpmt), np.mean(tpmt), np.std(tpmt)]
    popt_tpmt, _ = curve_fit(gaussian, bin_centers_tpmt, hist_tpmt, p0=initial_guess_tpmt)
    x_fit = np.linspace(xmin, xmax, 400)

    ax1.hist(tpmt, bins=bins, alpha=0.7, range=(xmin, xmax), histtype='step', label=r'$m_{\tau\tau}^{TPMT}$', color = 'blue')
    ax1.plot(x_fit, gaussian(x_fit, *popt_tpmt), color = 'blue', linestyle='dashed', label=f'TPMT Fit (μ={popt_tpmt[1]:.1f}, σ={popt_tpmt[2]:.1f})')
    ax1.set_xlabel(r'$m_{\tau\tau}$', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_xlim(xmin, xmax)
    ax1.legend()
    ax1.grid(True)
    ax1.set_title(f"{folders[i]}" + r" - $m_{\tau\tau}$" + f" - {label}", loc = 'right')

    ax2.plot(np.sort(tpmt), np.linspace(0, 1, len(tpmt)), label="TPMT CDF", color = 'blue',)
    ax2.set_xlabel(r'$m_{\tau\tau}$', fontsize=12)
    ax2.set_xlim(0, 400)
    ax2.legend(loc = "center right", fontsize = 10)
    ax2.set_ylabel('CDF', fontsize=12)
    ax2.grid(True)

    for ax in axes:
        ax.text(0.02, 1.00, r'$\mathbf{CMS}$' + r' $\mathit{Simulation}$', 
                 transform=ax.transAxes, fontsize=12, verticalalignment='bottom', horizontalalignment='left')

        ax.text(0.02, 0.95, r'Private Work', 
                 transform=ax.transAxes, fontsize=12, verticalalignment='bottom', horizontalalignment='left')

    plt.tight_layout()
    plt.legend()
    
    plt.savefig(f'{model_folder}/results/analysis_{folders[i]}_{pairType}.png', dpi=300)
    plt.close()




def compute_efficiency_asymmetric(signal, background, lower_percentile=5, upper_percentile=95):
    """
    Computes background efficiency using a dynamically chosen signal mass window
    based on given percentiles (default: 5th-95th).
    
    Args:
    - signal : Signal mass distribution
    - background : Background mass distribution
    - lower_percentile : Lower bound percentile
    - upper_percentile : Upper bound percentile

    Returns:
    - background_eff : Background efficiency within the signal mass window
    - signal_eff : Signal efficiency within the mass window 
    - cut_low : Computed lower mass cut
    - cut_high : Computed upper mass cut
    - median_signal : Median of the signal distribution (to check asymmetry)
    """

    cut_low, cut_high = np.percentile(signal, [lower_percentile, upper_percentile]) # compute dynamic percentiles
    
    median_signal = np.median(signal) # median of signal (helps check for asymmetry)

    # total number of events
    total_signal = len(signal)
    total_background = len(background)

    # compute efficiency as the fraction of events within the interval
    signal_eff = np.sum((signal >= cut_low) & (signal <= cut_high)) / total_signal
    background_eff = np.sum((background >= cut_low) & (background <= cut_high)) / total_background

    return background_eff, signal_eff, cut_low, cut_high, median_signal

background_eff_tpmt, signal_eff_tpmt, tpmt_low, tpmt_high, tpmt_median = compute_efficiency_asymmetric(tpmt_masses[1], tpmt_masses[0])

print(f"TPMT: Background Efficiency = {background_eff_tpmt:.4f}, Signal Efficiency = {signal_eff_tpmt:.4f}, Range = [{tpmt_low:.1f}, {tpmt_high:.1f}], Median = {tpmt_median:.1f}")



