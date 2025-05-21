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


def plot_mass_distribution(tpmt, folder, label, model_folder, bins, xmin, xmax):
    fig = plt.figure(figsize=(6, 6))

    tpmt = np.array(tpmt)
    hist, bin_edges = np.histogram(tpmt, bins=bins, range=(xmin, xmax))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    popt, _ = curve_fit(gaussian, bin_centers, hist, p0=[max(hist), np.mean(tpmt), np.std(tpmt)])
    x_fit = np.linspace(xmin, xmax, 400)

    plt.hist(tpmt, bins=bins, alpha=0.7, range=(xmin, xmax), histtype='step', label=r'$m_{\tau\tau}^{TPMT}$', color='blue')
    plt.plot(x_fit, gaussian(x_fit, *popt), 'b--', label=f'TPMT Fit (μ={popt[1]:.1f}, σ={popt[2]:.1f})')

    plt.xticks(np.linspace(xmin, xmax, 11))
    plt.legend()
    plt.title(f"{folder} - {label}", fontsize=12, loc='right')
    plt.xlabel(r'$m_{\tau\tau}$')
    plt.ylabel('Frequency')
    plt.xlim(xmin, xmax)
    plt.grid(True)

    # CMS label
    plt.text(0.02, 1.00, r'$\mathbf{CMS}$' + r' $\mathit{Simulation}$', transform=plt.gca().transAxes, fontsize=12, verticalalignment='bottom')
    plt.text(0.02, 0.95, r'Private Work', transform=plt.gca().transAxes, fontsize=12, verticalalignment='bottom')

    plt.tight_layout()
    plt.savefig(f'{model_folder}/results/masses_pair_{folder}_{pairType}.png', dpi=300)
    plt.close()



def plot_mass_and_cdf(tpmt, folder, label, model_folder, bins, xmin, xmax):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    tpmt = np.array(tpmt)
    hist, bin_edges = np.histogram(tpmt, bins=bins, range=(xmin, xmax))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    popt, _ = curve_fit(gaussian, bin_centers, hist, p0=[max(hist), np.mean(tpmt), np.std(tpmt)])
    x_fit = np.linspace(xmin, xmax, 400)

    ax1.hist(tpmt, bins=bins, alpha=0.7, range=(xmin, xmax), histtype='step', label=r'$m_{\tau\tau}^{TPMT}$', color='blue')
    ax1.plot(x_fit, gaussian(x_fit, *popt), 'b--', label=f'TPMT Fit (μ={popt[1]:.1f}, σ={popt[2]:.1f})')
    ax1.set_xlim(xmin, xmax)
    ax1.set_xlabel(r'$m_{\tau\tau}$')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f"{folder} - $m_{{\tau\tau}}$ - {label}", loc='right')
    ax1.legend()
    ax1.grid(True)

    # CDF
    ax2.plot(np.sort(tpmt), np.linspace(0, 1, len(tpmt)), label="TPMT CDF", color='blue')
    ax2.set_xlim(xmin, xmax)
    ax2.set_xlabel(r'$m_{\tau\tau}$')
    ax2.set_ylabel('CDF')
    ax2.legend(loc='center right')
    ax2.grid(True)

    for ax in (ax1, ax2):
        ax.text(0.02, 1.00, r'$\mathbf{CMS}$' + r' $\mathit{Simulation}$', transform=ax.transAxes, fontsize=12, verticalalignment='bottom')
        ax.text(0.02, 0.95, r'Private Work', transform=ax.transAxes, fontsize=12, verticalalignment='bottom')

    plt.tight_layout()
    plt.savefig(f'{model_folder}/results/analysis_{folder}_{pairType}.png', dpi=300)
    plt.close()


# Efficiency computation
def compute_efficiency_asymmetric(signal, background, lower_percentile=5, upper_percentile=95):
    cut_low, cut_high = np.percentile(signal, [lower_percentile, upper_percentile])
    median_signal = np.median(signal)
    total_signal = len(signal)
    total_background = len(background)

    signal_eff = np.sum((signal >= cut_low) & (signal <= cut_high)) / total_signal
    background_eff = np.sum((background >= cut_low) & (background <= cut_high)) / total_background

    return background_eff, signal_eff, cut_low, cut_high, median_signal




base_folder = "/gwpool/users/camagni/DiTau/TauPairMassTransformer/pre_processing/"
config = get_config(base_folder)
model_folder = config['model_folder']
pairType = config['test_pairType']
label_map = {"ele_tau": r"$e\tau$", "mu_tau": r"$\mu\tau$", "tau_tau": r"$\tau\tau$"}
label = label_map.get(pairType, pairType)

tpmt_masses = []
reco_masses = []
folders = []

# Load data
for folder in os.listdir(os.path.join(model_folder, 'results')):
    if folder.endswith(".png"):
        continue
    folders.append(folder)
    print(f"Processing: {folder}")
    csv_path = glob.glob(f"{model_folder}/results/{folder}/results_inference_{pairType}*.csv")[0]
    df = pd.read_csv(csv_path)
    tpmt_masses.append(df['pred_mass'])
    reco_masses.append(df['reco_mass'])

# Plotting
xmin, xmax, bins = 0, 400, 120

for i, folder in enumerate(folders):
    plot_mass_distribution(tpmt_masses[i], folder, label, model_folder, bins, xmin, xmax)
    plot_mass_and_cdf(tpmt_masses[i], folder, label, model_folder, bins, xmin, xmax)

# Efficiency
background_eff_tpmt, signal_eff_tpmt, tpmt_low, tpmt_high, tpmt_median = compute_efficiency_asymmetric(
    signal=tpmt_masses[1], background=tpmt_masses[0]
)

print(f"TPMT: Background Efficiency = {background_eff_tpmt:.4f}, Signal Efficiency = {signal_eff_tpmt:.4f}, "
      f"Range = [{tpmt_low:.1f}, {tpmt_high:.1f}], Median = {tpmt_median:.1f}")
