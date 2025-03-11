import matplotlib.pyplot as plt
import numpy as np
import mplhep as hep
from matplotlib.colors import LogNorm
import os
from scipy.optimize import curve_fit
from sklearn.metrics import roc_curve, roc_auc_score


# Define the Gaussian function
def gaussian(x, a, b, c):
    return a * np.exp(-((x - b) ** 2) / (2 * c ** 2))


def validation_plots_pt_mass(results, fastmtt, plots_folder, pairtype):

    target_pt1 = results['T_pt_1'].to_numpy()
    target_pt2 = results['T_pt_2'].to_numpy()
    regressed_pt1 = results['P_pt_1'].to_numpy()
    regressed_pt2 = results['P_pt_2'].to_numpy()
    target_mass = results['tar_mass'].to_numpy()
    regressed_mass = results['pred_mass'].to_numpy()
    fastmtt_pt1 = fastmtt['FastMTT_pt_1'].to_numpy()
    fastmtt_pt2 = fastmtt['FastMTT_pt_2'].to_numpy()
    fastmtt_mass = fastmtt['FastMTT_mass'].to_numpy()

    xmin = 0
    xmax = 250
    bins = 100

    if "TTTo" in plots_folder:
        mass_xmin = 0
        mass_xmax = 250
    elif "M350" in plots_folder:
        mass_xmin = 200
        mass_xmax = 500
    else:
        mass_xmin = 50
        mass_xmax = 200    

    ##### 2D Histogram plots for pt (separate for tau1 and tau2) and mass (single plot)
    f1, axs = plt.subplots(1, 3, figsize=(18, 6))
    # Plot for pt (Tau1)
    im1 = axs[0].hist2d(target_pt1, regressed_pt1, bins=bins, range = [[xmin, xmax], [xmin, xmax]] , norm=LogNorm())
    axs[0].set_xlabel(r"$p_T^{MC}$ - $\tau_1$", fontsize = 14)
    axs[0].set_ylabel(r"$p_T^{TPMT}$ - $\tau_1$", fontsize = 14)
    axs[0].set_xlim(xmin, xmax)
    axs[0].set_ylim(xmin, xmax)
    axs[0].set_title(r"TPMT vs MC $p_T$ $\tau_1$", loc = 'right', fontsize = 14)
    axs[0].grid(True)
    axs[0].set_xticks(np.linspace(xmin, xmax, 11))  
    axs[0].set_yticks(np.linspace(xmin, xmax, 11))  
    axs[0].text(0.02, 1.00, r'$\mathbf{CMS}$' + r' $\mathit{Simulation}$', 
            transform=axs[0].transAxes, fontsize=14, verticalalignment='bottom', 
            horizontalalignment='left')

    axs[0].text(0.02, 0.95, r'Private Work', 
            transform=axs[0].transAxes, fontsize=14, verticalalignment='bottom', 
            horizontalalignment='left')
    f1.colorbar(im1[3], ax=axs[0])
    # Plot for pt (Tau2)
    im2 = axs[1].hist2d(target_pt2, regressed_pt2, bins=bins, range = [[xmin, xmax], [xmin, xmax]], norm=LogNorm())
    axs[1].set_xlabel(r"$p_T^{MC}$ - $\tau_2$", fontsize = 14)
    axs[1].set_ylabel(r"$p_T^{TPMT}$ - $\tau_2$", fontsize = 14)
    axs[1].set_xlim(xmin, xmax)
    axs[1].set_ylim(xmin, xmax)
    axs[1].set_title(r"TPMT vs MC $p_T$ $\tau_2$", loc = 'right', fontsize = 14)
    axs[1].grid(True)
    axs[1].set_xticks(np.linspace(xmin, xmax, 11))  
    axs[1].set_yticks(np.linspace(xmin, xmax, 11))  
    axs[1].text(1.46, 1.00, r'$\mathbf{CMS}$' + r' $\mathit{Simulation}$', 
            transform=axs[0].transAxes, fontsize=14, verticalalignment='bottom', 
            horizontalalignment='left')

    axs[1].text(1.46, 0.95, r'Private Work', 
            transform=axs[0].transAxes, fontsize=14, verticalalignment='bottom', 
            horizontalalignment='left')
    f1.colorbar(im2[3], ax=axs[1])
    # Plot for mass (unified)
    im3 = axs[2].hist2d(target_mass, regressed_mass, bins=bins, range = [[mass_xmin, mass_xmax], [mass_xmin, mass_xmax]], norm=LogNorm())
    axs[2].set_xlabel(r"$m_{\tau\tau}^{MC}$", fontsize = 14)
    axs[2].set_ylabel(r"$m_{\tau\tau}^{TPMT}$", fontsize = 14)
    axs[2].set_xlim(mass_xmin, mass_xmax)
    axs[2].set_ylim(mass_xmin, mass_xmax)
    axs[2].set_title(r"TPMT vs MC $m_{\tau\tau}$", loc = 'right', fontsize = 14)
    axs[2].grid(True)
    axs[2].set_xticks(np.linspace(mass_xmin, mass_xmax, 11))  
    axs[2].set_yticks(np.linspace(mass_xmin, mass_xmax, 11))  
    axs[2].text(2.9, 1.00, r'$\mathbf{CMS}$' + r' $\mathit{Simulation}$', 
            transform=axs[0].transAxes, fontsize=14, verticalalignment='bottom', 
            horizontalalignment='left')

    axs[2].text(2.9, 0.95, r'Private Work', 
            transform=axs[0].transAxes, fontsize=14, verticalalignment='bottom', 
            horizontalalignment='left')
    f1.colorbar(im3[3], ax=axs[2])

    plt.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.1, wspace=0.15, hspace=0.1)
    f1.savefig(os.path.join(plots_folder, "2D_histograms_pt_mass_{pairtype}.png".format(pairtype=pairtype)), dpi = 300)

    ##### 1D Histograms for pt (separate for tau1 and tau2) and mass (single plot)
    f2, axs = plt.subplots(1, 3, figsize=(18, 6))
    # Hist for pt (Tau1)
    axs[0].hist(target_pt1, bins=bins, range = (xmin, xmax), label=r"$p_T^{MC}$ - $\tau_1$", histtype="step", color = "green", linewidth=2)
    axs[0].hist(regressed_pt1, bins=bins, range = (xmin, xmax), label=r"$p_T^{TPMT}$ - $\tau_1$", histtype="step", color = "red", linewidth=2)
    axs[0].hist(fastmtt_pt1, bins=bins, range = (xmin, xmax), label=r"$p_T^{FastMTT}$ - $\tau_1$", histtype="step", color = "deepskyblue", linewidth=2)
    axs[0].set_xlabel(r"$p_T$", fontsize = 14)
    axs[0].set_ylabel(r"Frequency", fontsize = 14)
    axs[0].set_xlim(xmin, xmax)
    axs[0].legend()
    axs[0].set_title(r"$p_T^{TPMT}$ vs $p_T^{MC}$ vs $p_T^{FastMTT}$ - $\tau_1$", loc = 'right', fontsize = 14)
    axs[0].grid(True)
    axs[0].set_xticks(np.linspace(xmin, xmax, 11)) 
    axs[0].text(0.02, 1.00, r'$\mathbf{CMS}$' + r' $\mathit{Simulation}$', 
            transform=axs[0].transAxes, fontsize=14, verticalalignment='bottom', 
            horizontalalignment='left')

    axs[0].text(0.02, 0.95, r'Private Work', 
            transform=axs[0].transAxes, fontsize=14, verticalalignment='bottom', 
            horizontalalignment='left') 
    # Hist for pt (Tau2)
    axs[1].hist(target_pt2, bins=bins, range = (xmin, xmax),label=r"$p_T^{MC}$ - $\tau_2$", histtype="step", color= "green", linewidth=2)
    axs[1].hist(regressed_pt2, bins=bins, range = (xmin, xmax),label=r"$p_T^{TPMT}$ - $\tau_2$", histtype="step", color = "red", linewidth=2)
    axs[1].hist(fastmtt_pt2, bins=bins, range = (xmin, xmax), label=r"$p_T^{FastMTT}$ - $\tau_2$", histtype="step", color = "deepskyblue", linewidth=2)
    axs[1].set_xlabel(r"$p_T$", fontsize = 14)
    axs[1].set_xlim(xmin, xmax)
    axs[1].legend()
    axs[1].set_title(r"$p_T^{TPMT}$ vs $p_T^{MC}$ vs $p_T^{FastMTT}$ - $\tau_2$", loc = 'right', fontsize = 14)
    axs[1].grid(True)
    axs[1].set_xticks(np.linspace(xmin, xmax, 11)) 
    axs[1].text(1.22, 1.00, r'$\mathbf{CMS}$' + r' $\mathit{Simulation}$', 
            transform=axs[0].transAxes, fontsize=14, verticalalignment='bottom', 
            horizontalalignment='left')

    axs[1].text(1.22, 0.95, r'Private Work', 
            transform=axs[0].transAxes, fontsize=14, verticalalignment='bottom', 
            horizontalalignment='left')  
    

    # Hist for mass (unified)    
    axs[2].hist(regressed_mass, bins=bins, range = (mass_xmin, mass_xmax), label=r"TPMT $m_{\tau\tau}$", histtype="step", color = "red", linewidth=2)
    axs[2].hist(target_mass, bins=bins, range = (mass_xmin, mass_xmax), label=r"MC $m_{\tau\tau}$", histtype="step", color = "green", linewidth=2)
    axs[2].hist(fastmtt_mass, bins=bins, range = (mass_xmin, mass_xmax), label=r"FastMTT $m_{\tau\tau}$", histtype="step", color = "deepskyblue", linewidth=2)
    axs[2].set_xlabel(r"$m_{\tau\tau}$", fontsize = 14)
    axs[2].legend()
    axs[2].set_title(r"$m_{\tau\tau}$ histogram", loc = 'right', fontsize = 14)
    axs[2].grid(True)
    axs[2].set_xticks(np.linspace(mass_xmin, mass_xmax, 11))  # More ticks
    axs[2].set_xlim(mass_xmin, mass_xmax)

    axs[2].text(2.42, 1.00, r'$\mathbf{CMS}$' + r' $\mathit{Simulation}$', 
            transform=axs[0].transAxes, fontsize=14, verticalalignment='bottom', 
            horizontalalignment='left')

    axs[2].text(2.42, 0.95, r'Private Work', 
            transform=axs[0].transAxes, fontsize=14, verticalalignment='bottom', 
            horizontalalignment='left')  
    
    plt.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.1, wspace=0.2, hspace=0.1)
    f2.savefig(os.path.join(plots_folder, "1D_histograms_pt_mass_{pairtype}.png".format(pairtype=pairtype)), dpi = 300)

    #### Delta plots for pt (separate for tau1 and tau2) and mass (single plot)
    f3, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Delta pt (difference between regressed and target pt for tau1)
    im1 = axs[0].hist2d(regressed_pt1 - target_pt1, regressed_pt2 - target_pt2, range = [[-100, 100], [-100, 100]], 
                  bins=bins, norm=LogNorm())
    axs[0].set_xlabel(r"$p_T^{TPMT}$ - $p_T^{MC}$ ($\tau_1$)", fontsize = 14)
    axs[0].set_ylabel(r"$p_T^{TPMT}$ - $p_T^{MC}$ ($\tau_2$)", fontsize = 14)
    axs[0].set_title(r"$\Delta p_T$ $\tau_1$ vs $\tau_2$", loc = 'right', fontsize = 14)
    axs[0].set_xlim(-100, 100)
    axs[0].set_ylim(-100, 100)
    axs[0].grid(True)
    axs[0].set_xticks(np.linspace(-100, 100, 11))  
    axs[0].set_yticks(np.linspace(-100, 100, 11))  
    axs[0].text(0.02, 1.00, r'$\mathbf{CMS}$' + r' $\mathit{Simulation}$', 
            transform=axs[0].transAxes, fontsize=14, verticalalignment='bottom', 
            horizontalalignment='left')

    axs[0].text(0.02, 0.95, r'Private Work', 
            transform=axs[0].transAxes, fontsize=14, verticalalignment='bottom', 
            horizontalalignment='left')
    f3.colorbar(im1[3], ax=axs[0])
    # Delta pt (difference between regressed and target pt for tau2)
    im2 = axs[1].hist2d(target_pt1, target_pt2, bins=bins, range = [[0, 400], [0, 400]], norm=LogNorm())
    axs[1].set_xlabel(r"$p_T^{MC}$ $\tau_1$", fontsize = 14)
    axs[1].set_ylabel(r"$p_T^{MC}$ $\tau_2$", fontsize = 14)
    axs[1].set_title(r"$p_T^{MC}$ $\tau_1$ vs $\tau_2$", loc = 'right', fontsize = 14)
    axs[1].set_xlim(0, 400)
    axs[1].set_ylim(0, 400)
    axs[1].grid(True)
    axs[1].set_xticks(np.linspace(0, 400, 11))  
    axs[1].set_yticks(np.linspace(0, 400, 11))  
    axs[1].text(1.46, 1.00, r'$\mathbf{CMS}$' + r' $\mathit{Simulation}$', 
            transform=axs[0].transAxes, fontsize=14, verticalalignment='bottom', 
            horizontalalignment='left')

    axs[1].text(1.46, 0.95, r'Private Work', 
            transform=axs[0].transAxes, fontsize=14, verticalalignment='bottom', 
            horizontalalignment='left')
    f3.colorbar(im2[3], ax=axs[1])
    # Delta mass (single plot)
    im3 = axs[2].hist2d(regressed_pt1, regressed_pt2, bins=bins, range = [[0, 400], [0, 400]], norm=LogNorm())
    axs[2].set_xlabel(r"$p_T^{TPMT}$ $\tau_1$", fontsize = 14)
    axs[2].set_ylabel(r"$p_T^{TPMT}$ $\tau_2$", fontsize = 14)
    axs[2].set_title(r"$p_T^{TPMT}$ $\tau_1$ vs $\tau_2$", loc = 'right', fontsize = 14)
    axs[2].set_xlim(0, 400)
    axs[2].set_ylim(0, 400)
    axs[2].grid(True)
    axs[2].set_xticks(np.linspace(0, 400, 11))  
    axs[2].set_yticks(np.linspace(0, 400, 11))  
    axs[2].text(2.9, 1.00, r'$\mathbf{CMS}$' + r' $\mathit{Simulation}$', 
            transform=axs[0].transAxes, fontsize=14, verticalalignment='bottom', 
            horizontalalignment='left')

    axs[2].text(2.9, 0.95, r'Private Work', 
            transform=axs[0].transAxes, fontsize=14, verticalalignment='bottom', 
            horizontalalignment='left')
    f3.colorbar(im3[3], ax=axs[2])
        
    plt.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.1, wspace=0.15, hspace=0.1)
    f3.savefig(os.path.join(plots_folder, "Delta_plots_pt_mass_{pairtype}.png".format(pairtype=pairtype)), dpi = 300)

    plt.close(f1)
    plt.close(f2)
    plt.close(f3)

    return f1, f2, f3



def plot_response_vs_pt(results, save_path):
    bins = np.linspace(0, 200, 51)
    
    target_pt1 = results['T_pt_1'].to_numpy()
    target_pt2 = results['T_pt_2'].to_numpy()
    regressed_pt1 = results['P_pt_1'].to_numpy()
    regressed_pt2 = results['P_pt_2'].to_numpy()
    input_pt1 = results['R_pt_1'].to_numpy()
    input_pt2 = results['R_pt_2'].to_numpy()
    
    def calculate_stats(true_values, predicted_values, bins):
        bin_means, bin_rmse, bin_centers = [], [], []
        for bin_start, bin_end in zip(bins[:-1], bins[1:]):
            mask = (true_values >= bin_start) & (true_values < bin_end)
            bin_values = predicted_values[mask]
            if len(bin_values) > 0:
                mean = np.mean(bin_values)
                #std = np.std(bin_values)
                rmse = np.sqrt(np.mean((bin_values - mean)**2))
                bin_means.append(mean)
                bin_rmse.append(rmse)
                bin_centers.append((bin_start + bin_end) / 2)
        return np.array(bin_centers), np.array(bin_means), np.array(bin_rmse)
    
    bin_centers_reco_tau1, means_reco_tau1, rmse_reco_tau1 = calculate_stats(input_pt1, regressed_pt1 / target_pt1, bins)
    bin_centers_reco_tau2, means_reco_tau2, rmse_reco_tau2 = calculate_stats(input_pt2, regressed_pt2 / target_pt2, bins)
    
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    
    axs[0].errorbar(bin_centers_reco_tau1, means_reco_tau1, yerr=rmse_reco_tau1, fmt='o-', label=r'$\tau_1$', color='red')
    axs[0].set_xlabel(r'$p_T^{RECO}$', fontsize = 14)
    axs[0].set_ylabel(r'$p_T^{TPMT}$/$p_T^{MC}$', fontsize = 14)
    axs[0].set_title(r'$\tau_1$: $p_T^{RECO}$ vs $p_T^{TPMT}$/$p_T^{MC}$', loc = 'right', fontsize = 14)
    axs[0].set_xlim(0, 200)
    axs[0].set_ylim(0.5, 1.5)
    axs[0].grid(True)
    axs[0].yaxis.set_major_locator(plt.MultipleLocator(0.1))
    axs[0].legend()
    
    axs[0].text(0.02, 1.00, r'$\mathbf{CMS}$' + r' $\mathit{Simulation}$', 
            transform=axs[0].transAxes, fontsize=14, verticalalignment='bottom', 
            horizontalalignment='left')

    axs[0].text(0.02, 0.95, r'Private Work', 
            transform=axs[0].transAxes, fontsize=14, verticalalignment='bottom', 
            horizontalalignment='left')
    
    axs[1].errorbar(bin_centers_reco_tau2, means_reco_tau2, yerr=rmse_reco_tau2, fmt='o-', label=r'$\tau_2$', color='blue')
    axs[1].set_xlabel(r'$p_T^{RECO}$', fontsize = 14)
    axs[1].set_ylabel(r'$p_T^{TPMT}$/$p_T^{MC}$', fontsize = 14)
    axs[1].set_title(r'$\tau_2$: $p_T^{RECO}$ vs $p_T^{TPMT}$/$p_T^{MC}$', loc = 'right', fontsize = 14)
    axs[1].set_xlim(0, 200)
    axs[1].set_ylim(0.4, 1.4)
    axs[1].grid(True)
    axs[1].yaxis.set_major_locator(plt.MultipleLocator(0.1))
    axs[1].legend()
    

    axs[1].text(1.12, 1.00, r'$\mathbf{CMS}$' + r' $\mathit{Simulation}$', 
            transform=axs[0].transAxes, fontsize=14, verticalalignment='bottom', 
            horizontalalignment='left')

    axs[1].text(1.12, 0.95, r'Private Work', 
            transform=axs[0].transAxes, fontsize=14, verticalalignment='bottom', 
            horizontalalignment='left')
        
    plt.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.1, wspace=0.1, hspace=0.1)

    plt.savefig(save_path, dpi = 300)
    plt.close()



# Plot ROC curve function
def plot_roc_curve(result_tar_mass, result_pred_mass, result_fastmtt_mass, base_folder, pairtype):
    
    if pairtype == "ele_tau":
        label = r"$e\tau$"
    elif pairtype == "mu_tau":
        label = r"$\mu\tau$"
    elif pairtype == "tau_tau":
        label = r"$\tau\tau$"

    fpr_T, tpr_T, _ = roc_curve(result_tar_mass, result_pred_mass, pos_label=125.0)
    fpr_R, tpr_R, _ = roc_curve(result_tar_mass, result_fastmtt_mass, pos_label=125.0)

    # Compute AUC
    roc_auc_T = roc_auc_score(result_tar_mass, result_pred_mass)
    roc_auc_R = roc_auc_score(result_tar_mass, result_fastmtt_mass)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr_T, tpr_T, color='green', lw=2, label=f'AUC TPMT = {roc_auc_T:.2f}')
    plt.plot(fpr_R, tpr_R, color='red', lw=2, label=f'AUC FastMTT = {roc_auc_R:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(r'$m_{\tau\tau}^{TPMT}$ and $m_{\tau\tau}^{FastMTT}$ ROC curves' + f" - {label}")
    plt.legend(loc='lower right')
    plt.savefig(f"{base_folder}/roc_TPMT_FastMTT_{pairtype}.png", dpi = 300)
    plt.close()



# Plot mass comparison function
def plot_mass_comparison(resultsH, resultsH_fastmtt, resultsDY, resultsDY_fastmtt, base_folder, pairtype):
    plt.figure(figsize=(6, 4), dpi=400)
    plt.style.use(hep.style.CMS)
    plt.xticks(np.arange(0, 250, 20))
    x = np.linspace(0, 250, 150)

    # Define a function to create histograms
    def create_histogram(data):
        hist, bin_edges = np.histogram(data, weights=np.ones(len(data)) / len(data), density=False, bins=100, range=(0, 250))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        return hist, bin_centers

    def fit_histogram(bin_centers, hist, initial_guess):
        popt, _ = curve_fit(gaussian, bin_centers, hist, p0=initial_guess)
        return popt

    hist_s_H, bin_centers_s_H = create_histogram(np.array(resultsH_fastmtt['svfit_mass']))
    hist_p_H, bin_centers_p_H = create_histogram(resultsH['pred_mass'])

    hist_s_DY, bin_centers_s_DY = create_histogram(np.array(resultsDY_fastmtt['svfit_mass']))
    hist_p_DY, bin_centers_p_DY = create_histogram(resultsDY['pred_mass'])

    initial_guess_H = [0.035, 125, 10]
    initial_guess_DY = [0.035, 90, 10]

    popt_H_svfit = fit_histogram(bin_centers_s_H, hist_s_H, initial_guess_H)
    popt_H_tpm = fit_histogram(bin_centers_p_H, hist_p_H, initial_guess_H)
    
    popt_DY_svfit = fit_histogram(bin_centers_s_DY, hist_s_DY, initial_guess_DY)
    popt_DY_tpm = fit_histogram(bin_centers_p_DY, hist_p_DY, initial_guess_DY)

    # Plot the distributions
    plt.hist(resultsH['pred_mass'], weights=np.ones(len(resultsH['pred_mass'])) / len(resultsH['pred_mass']),
             density=False, bins=100, range=(0, 250), histtype='step', linewidth=1.7, label=r'$m_{\tau\tau}$ TPMT - H', color='blue')
    plt.hist(resultsDY['pred_mass'], weights=np.ones(len(resultsDY['pred_mass'])) / len(resultsDY['pred_mass']),
             density=False, bins=100, range=(0, 250), histtype='step', linewidth=1.7, label=r'$m_{\tau\tau}$ TPMT - Z', color='red')
    plt.hist(resultsH_fastmtt['svfit_mass'], weights=np.ones(len(resultsH_fastmtt['svfit_mass'])) / len(resultsH_fastmtt['svfit_mass']),
             density=False, bins=100, range=(0, 250), histtype='step', linestyle="dashed", linewidth=1.2, label=r'$m_{\tau\tau}$ FastMTT - H', color='blue')
    plt.hist(resultsDY_fastmtt['svfit_mass'], weights=np.ones(len(resultsDY_fastmtt['svfit_mass'])) / len(resultsDY_fastmtt['svfit_mass']),
             density=False, bins=100, range=(0, 250), histtype='step', linestyle="dashed", linewidth=1.2, label=r'$m_{\tau\tau}$ FastMTT - Z', color='red')

    plt.plot(bin_centers_s_H, gaussian(bin_centers_s_H, *popt_H_svfit), linestyle='--', linewidth=0.8, color="blue")
    plt.plot(bin_centers_p_H, gaussian(bin_centers_p_H, *popt_H_tpm), linestyle='--', linewidth=0.8, color="blue")

    plt.plot(bin_centers_s_DY, gaussian(bin_centers_s_DY, *popt_DY_svfit), linestyle='--', linewidth=0.8, color="red")
    plt.plot(bin_centers_p_DY, gaussian(bin_centers_p_DY, *popt_DY_tpm), linestyle='--', linewidth=0.8, color="red")

    plt.xlim(0, 250)
    plt.xlabel('Invariant Mass [GeV]', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(r'$m_{\tau\tau}^{TPMT}$ vs $m_{\tau\tau}^{FastMTT}$ for H and Z samples' + f" - {pairtype}", fontsize=16)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(loc='upper right', fontsize=10)
    plt.savefig(f"{base_folder}/mZH_{pairtype}_TPMT_FASTMTT.png")
    plt.close()

    return popt_H_svfit, popt_H_tpm, popt_DY_svfit, popt_DY_tpm

