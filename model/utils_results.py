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


def validation_plots_pt_mass(results, plots_folder, pairtype):

    # Extracting data from the DataFrame
    target_pt1 = results['T_pt_1'].to_numpy()
    target_pt2 = results['T_pt_2'].to_numpy()
    regressed_pt1 = results['P_pt_1'].to_numpy()
    regressed_pt2 = results['P_pt_2'].to_numpy()
    target_mass = results['tar_mass'].to_numpy()
    regressed_mass = results['pred_mass'].to_numpy()

    xmin = 0
    xmax = 200
    bins = 100

    ##### 2D Histogram plots for pt (separate for tau1 and tau2) and mass (single plot)
    f1, axs = plt.subplots(1, 3, figsize=(18, 6))
    # Plot for pt (Tau1)
    im1 = axs[0].hist2d(target_pt1, regressed_pt1, bins=bins, range = [[xmin, xmax], [xmin, xmax]] , norm=LogNorm())
    axs[0].set_xlabel(r"MC pt - $\tau_1$")
    axs[0].set_ylabel(r"TPMT pt - $\tau_1")
    axs[0].set_xlim(xmin, xmax)
    axs[0].set_ylim(xmin, xmax)
    axs[0].set_title(r"2D Histogram for $p_T$ $\tau_1$")
    axs[0].grid(True)
    axs[0].set_xticks(np.linspace(xmin, xmax, 11))  
    axs[0].set_yticks(np.linspace(xmin, xmax, 11))  
    f1.colorbar(im1[3], ax=axs[0])
    # Plot for pt (Tau2)
    im2 = axs[1].hist2d(target_pt2, regressed_pt2, bins=bins, range = [[xmin, xmax], [xmin, xmax]], norm=LogNorm())
    axs[1].set_xlabel(r"MC $p_T$ - $\tau_2$")
    axs[1].set_ylabel(r"TPMT $p_T$ - $\tau_2")
    axs[1].set_xlim(xmin, xmax)
    axs[1].set_ylim(xmin, xmax)
    axs[1].set_title(r"2D Histogram for $p_T$ $\tau_2$")
    axs[1].grid(True)
    axs[1].set_xticks(np.linspace(xmin, xmax, 11))  
    axs[1].set_yticks(np.linspace(xmin, xmax, 11))  
    f1.colorbar(im2[3], ax=axs[1])
    # Plot for mass (unified)
    im3 = axs[2].hist2d(target_mass, regressed_mass, bins=bins, range = [[xmin, xmax], [xmin, xmax]], norm=LogNorm())
    axs[2].set_xlabel("MC mass")
    axs[2].set_ylabel("TPMT mass")
    axs[2].set_xlim(xmin, xmax)
    axs[2].set_ylim(xmin, xmax)
    axs[2].set_title(r"2D Histogram $m_{\tau\tau}$")
    axs[2].grid(True)
    axs[2].set_xticks(np.linspace(xmin, xmax, 11))  
    axs[2].set_yticks(np.linspace(xmin, xmax, 11))  
    f1.colorbar(im3[3], ax=axs[2])

    plt.tight_layout()
    f1.savefig(os.path.join(plots_folder, "2D_histograms_pt_mass_{pairtype}.png".format(pairtype=pairtype)))

    ##### 1D Histograms for pt (separate for tau1 and tau2) and mass (single plot)
    f2, axs = plt.subplots(1, 3, figsize=(18, 6))
    # Hist for pt (Tau1)
    axs[0].hist(target_pt1, bins=bins, range = (xmin, xmax), label=r"MC $p_T$ - $\tau_1$", histtype="step", color = "green", linewidth=2)
    axs[0].hist(regressed_pt1, bins=bins, range = (xmin, xmax), label=r"TPMT $p_T$ - $\tau_1$", histtype="step", color = "red", linewidth=2)
    axs[0].set_xlabel(r"$p_T$")
    axs[0].set_xlim(xmin, xmax)
    axs[0].legend()
    axs[0].set_title(r"1D Histogram for $p_T$ - $\tau_1$")
    axs[0].grid(True)
    axs[0].set_xticks(np.linspace(xmin, xmax, 11))  
    # Hist for pt (Tau2)
    axs[1].hist(target_pt2, bins=bins, range = (xmin, xmax),label=r"MC $p_T$ - $\tau_2$", histtype="step", color= "green", linewidth=2)
    axs[1].hist(regressed_pt2, bins=bins, range = (xmin, xmax),label=r"TPMT $p_T$ - $\tau_2$", histtype="step", color = "red", linewidth=2)
    axs[1].set_xlabel(r"$p_T$")
    axs[1].set_xlim(xmin, xmax)
    axs[1].legend()
    axs[1].set_title(r"1D Histogram for $p_T$ - $\tau_2$")
    axs[1].grid(True)
    axs[1].set_xticks(np.linspace(xmin, xmax, 11))  
    # Hist for mass (unified)
    axs[2].hist(regressed_mass, bins=bins, range = (50, 400), label=r"TPMT $m_{\tau\tau}$", histtype="step", color = "red", linewidth=2)
    axs[2].set_xlabel(r"$m_{\tau\tau}$")
    axs[2].legend()
    axs[2].set_title(r"1D Histogram for $m_{\tau\tau}$")
    axs[2].grid(True)
    axs[2].set_xticks(np.linspace(50, 400, 11))  # More ticks
    axs[2].set_xlim(50, 400)

    plt.tight_layout()
    f2.savefig(os.path.join(plots_folder, "1D_histograms_pt_mass_{pairtype}.png".format(pairtype=pairtype)))

    #### Delta plots for pt (separate for tau1 and tau2) and mass (single plot)
    f3, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Delta pt (difference between regressed and target pt for tau1)
    im1 = axs[0].hist2d(regressed_pt1 - target_pt1, regressed_pt2 - target_pt2, range = [[-100, 100], [-100, 100]], 
                  bins=bins, norm=LogNorm())
    axs[0].set_xlabel(r"Delta $p_T$ $\tau_1$ (TPMT - MC)")
    axs[0].set_ylabel(r"Delta $p_T$ $\tau_2$ (TPMT - MC)")
    axs[0].set_title(r"Delta $p_T$ $\tau_1$ vs. $\tau_2$")
    axs[0].set_xlim(-100, 100)
    axs[0].set_ylim(-100, 100)
    axs[0].grid(True)
    axs[0].set_xticks(np.linspace(-100, 100, 11))  
    axs[0].set_yticks(np.linspace(-100, 100, 11))  
    f3.colorbar(im1[3], ax=axs[0])
    # Delta pt (difference between regressed and target pt for tau2)
    im2 = axs[1].hist2d(target_pt1, target_pt2, bins=bins, range = [[0, 400], [0, 400]], norm=LogNorm())
    axs[1].set_xlabel(r"MC $p_T$ $\tau_1$")
    axs[1].set_ylabel(r"MC $p_T$ $\tau_2$")
    axs[1].set_title(r"MC $p_T$ $\tau_1$ vs. $\tau_2$")
    axs[1].set_xlim(0, 400)
    axs[1].set_ylim(0, 400)
    axs[1].grid(True)
    axs[1].set_xticks(np.linspace(0, 400, 11))  
    axs[1].set_yticks(np.linspace(0, 400, 11))  
    f3.colorbar(im2[3], ax=axs[1])
    # Delta mass (single plot)
    im3 = axs[2].hist2d(regressed_pt1, regressed_pt2, bins=bins, range = [[0, 400], [0, 400]], norm=LogNorm())
    axs[2].set_xlabel(r"TPMT $p_T$ $\tau_1$")
    axs[2].set_ylabel(r"TPMT $p_T$ $\tau_2$")
    axs[2].set_title(r"TPMT $p_T$ $\tau_1$ vs. $\tau_2$")
    axs[2].set_xlim(0, 400)
    axs[2].set_ylim(0, 400)
    axs[2].grid(True)
    axs[2].set_xticks(np.linspace(0, 400, 11))  
    axs[2].set_yticks(np.linspace(0, 400, 11))  
    f3.colorbar(im3[3], ax=axs[2])

    plt.tight_layout()
    f3.savefig(os.path.join(plots_folder, "Delta_plots_pt_mass_{pairtype}.png".format(pairtype=pairtype)))

    plt.close(f1)
    plt.close(f2)
    plt.close(f3)

    #plot_mass_histogram(regressed_mass, plots_folder, pairtype)

    return f1, f2, f3


def plot_mass_histogram(regressed_mass, plots_folder, pairtype):

    fig, ax = plt.subplots(figsize=(8, 6))
    xmin, xmax = 70, 170

    ax.hist(regressed_mass, bins=100, range = (xmin, xmax), label="Regressed mass", histtype="step", color='b', linewidth=1.5)
    ax.set_xlabel("Mass", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_title("1D Histogram for mass", fontsize=14)
    ax.set_xticks(np.linspace(xmin, xmax, 21))  
    ax.set_xlim(xmin, xmax)
    ax.xaxis.set_tick_params(rotation=45)
    ax.grid(True, which='both', linestyle='--', linewidth=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_folder, "Regressed_Mass_{pairtype}.png".format(pairtype = pairtype)))
    plt.close(fig)  




def plot_response_vs_pt(results, save_path):
    """
    Questa funzione calcola e traccia la response (pt_regressed / pt_target) per ciascun valore di pt
    (reco, target, regressed) per tau1 e tau2, e salva il grafico.

    Args:
    - results (dict): Un dizionario contenente i dati per le variabili target_pt1, target_pt2,
                       regressed_pt1, regressed_pt2, input_pt1, input_pt2.
    - save_path (str): Il percorso in cui salvare il grafico.

    Returns:
    - None: Il grafico verrà salvato nel percorso fornito.
    """
    bins = np.linspace(0, 200, 51)
    # Estrai i dati dai risultati
    target_pt1 = results['T_pt_1'].to_numpy()
    target_pt2 = results['T_pt_2'].to_numpy()
    regressed_pt1 = results['P_pt_1'].to_numpy()
    regressed_pt2 = results['P_pt_2'].to_numpy()
    input_pt1 = results['R_pt_1'].to_numpy()  # pt reco per tau1
    input_pt2 = results['R_pt_2'].to_numpy()  # pt reco per tau2

    # Funzione per calcolare media e RMSE
    def calculate_stats(true_values, predicted_values, bins):
        bin_means = []
        bin_rmse = []
        bin_centers = []

        for bin_start, bin_end in zip(bins[:-1], bins[1:]):
            mask = (true_values >= bin_start) & (true_values < bin_end)
            bin_values = predicted_values[mask]

            if len(bin_values) > 0:
                mean = np.mean(bin_values)
                rmse = np.sqrt(np.mean((bin_values - mean)**2))
                bin_means.append(mean)
                bin_rmse.append(rmse)
                bin_centers.append((bin_start + bin_end) / 2)

        return np.array(bin_centers), np.array(bin_means), np.array(bin_rmse)

    # Calcola la media e RMSE per ciascun bin di pt (sia per tau1 che per tau2)
    bin_centers_reco_tau1, means_reco_tau1, rmse_reco_tau1 = calculate_stats(input_pt1, regressed_pt1 / target_pt1, bins)
    bin_centers_target_tau1, means_target_tau1, rmse_target_tau1 = calculate_stats(target_pt1, regressed_pt1 / target_pt1, bins)
    bin_centers_regressed_tau1, means_regressed_tau1, rmse_regressed_tau1 = calculate_stats(regressed_pt1, regressed_pt1 / target_pt1, bins)

    bin_centers_reco_tau2, means_reco_tau2, rmse_reco_tau2 = calculate_stats(input_pt2, regressed_pt2 / target_pt2, bins)
    bin_centers_target_tau2, means_target_tau2, rmse_target_tau2 = calculate_stats(target_pt2, regressed_pt2 / target_pt2, bins)
    bin_centers_regressed_tau2, means_regressed_tau2, rmse_regressed_tau2 = calculate_stats(regressed_pt2, regressed_pt2 / target_pt2, bins)

    # Crea i grafici
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))  # 2 righe, 3 colonne

    # Primo grafico: Tau1 - pt_reco vs pt_regressed / pt_target
    axs[0, 0].errorbar(bin_centers_reco_tau1, means_reco_tau1, yerr=rmse_reco_tau1, fmt='o-', label=r'$\tau_1$', color='red')
    axs[0, 0].set_xlabel(r'$p_T^{RECO}$')
    axs[0, 0].set_ylabel(r'$p_T^{TPMT}$/$p_T^{MC}$')
    axs[0, 0].set_title(r'$\tau_1$: $p_T^{RECO}$ vs $p_T^{TPMT}$/$p_T^{MC}$')
    axs[0, 0].set_xlim(0, 200)  # Imposta il range fisso dell'asse x
    axs[0, 0].set_ylim(0.4, 1.4)  # Imposta il range fisso dell'asse y
    axs[0, 0].grid(True)
    axs[0, 0].yaxis.set_major_locator(plt.MultipleLocator(0.1))  # Imposta griglia ogni 0.1
    axs[0, 0].legend()

    # Secondo grafico: Tau1 - pt_target vs pt_regressed / pt_target
    axs[0, 1].errorbar(bin_centers_target_tau1, means_target_tau1, yerr=rmse_target_tau1, fmt='o-', label=r'$\tau_1$', color='green')
    axs[0, 1].set_xlabel(r'$p_T^{MC}$')
    axs[0, 1].set_ylabel(r'$p_T^{TPMT}$/$p_T^{MC}$')
    axs[0, 1].set_title(r'$\tau_1$: $p_T^{MC}$ vs $p_T^{TPMT}$/$p_T^{MC}$')
    axs[0, 1].set_xlim(0, 200)  # Imposta il range fisso dell'asse x
    axs[0, 1].set_ylim(0.4, 1.4)  # Imposta il range fisso dell'asse y
    axs[0, 1].grid(True)
    axs[0, 1].yaxis.set_major_locator(plt.MultipleLocator(0.1))  # Imposta griglia ogni 0.1
    axs[0, 1].legend()

    # Terzo grafico: Tau1 - pt_regressed vs pt_regressed / pt_target
    axs[0, 2].errorbar(bin_centers_regressed_tau1, means_regressed_tau1, yerr=rmse_regressed_tau1, fmt='o-', label=r'$\tau_1$', color='blue')
    axs[0, 2].set_xlabel(r'$p_T^{TPMT}$')
    axs[0, 2].set_ylabel(r'$p_T^{TPMT}$/$p_T^{MC}$')
    axs[0, 2].set_title(r'$\tau_1$: $p_T^{TPMT}$ vs $p_T^{TPMT}$/$p_T^{MC}$')
    axs[0, 2].set_xlim(0, 200)  # Imposta il range fisso dell'asse x
    axs[0, 2].set_ylim(0.4, 1.4)  # Imposta il range fisso dell'asse y
    axs[0, 2].grid(True)
    axs[0, 2].yaxis.set_major_locator(plt.MultipleLocator(0.1))  # Imposta griglia ogni 0.1
    axs[0, 2].legend()

    # Quarto grafico: Tau2 - pt_reco vs pt_regressed / pt_target
    axs[1, 0].errorbar(bin_centers_reco_tau2, means_reco_tau2, yerr=rmse_reco_tau2, fmt='o-', label=r'$\tau_2$', color='red')
    axs[1, 0].set_xlabel(r'$p_T^{RECO}$')
    axs[1, 0].set_ylabel(r'$p_T^{TPMT}$/$p_T^{MC}$')
    axs[1, 0].set_title(r'$\tau_2$: $p_T^{RECO}$ vs $p_T^{TPMT}$/$p_T^{MC}$')
    axs[1, 0].set_xlim(0, 200)  # Imposta il range fisso dell'asse x
    axs[1, 0].set_ylim(0.4, 1.4)  # Imposta il range fisso dell'asse y
    axs[1, 0].grid(True)
    axs[1, 0].yaxis.set_major_locator(plt.MultipleLocator(0.1))  # Imposta griglia ogni 0.1
    axs[1, 0].legend()

    # Quinto grafico: Tau2 - pt_target vs pt_regressed / pt_target
    axs[1, 1].errorbar(bin_centers_target_tau2, means_target_tau2, yerr=rmse_target_tau2, fmt='o-', label=r'$\tau_2$', color='green')
    axs[1, 1].set_xlabel(r'$p_T^{MC}$')
    axs[1, 1].set_ylabel(r'$p_T^{TPMT}$/$p_T^{MC}$')
    axs[1, 1].set_title(r'$\tau_2$: $p_T^{MC}$ vs $p_T^{TPMT}$/$p_T^{MC}$')
    axs[1, 1].set_xlim(0, 200)  # Imposta il range fisso dell'asse x
    axs[1, 1].set_ylim(0.4, 1.4)  # Imposta il range fisso dell'asse y
    axs[1, 1].grid(True)
    axs[1, 1].yaxis.set_major_locator(plt.MultipleLocator(0.1))  # Imposta griglia ogni 0.1
    axs[1, 1].legend()

    # Sesto grafico: Tau2 - pt_regressed vs pt_regressed / pt_target
    axs[1, 2].errorbar(bin_centers_regressed_tau2, means_regressed_tau2, yerr=rmse_regressed_tau2, fmt='o-', label=r'$\tau_2$', color='blue')
    axs[1, 2].set_xlabel(r'$p_T^{TPMT}$')
    axs[1, 2].set_ylabel(r'$p_T^{TPMT}$/$p_T^{MC}$')
    axs[1, 2].set_title(r'$\tau_2$: $p_T^{TPMT}$ vs $p_T^{TPMT}$/$p_T^{MC}$')
    axs[1, 2].set_xlim(0, 200)  # Imposta il range fisso dell'asse x
    axs[1, 2].set_ylim(0.4, 1.4)  # Imposta il range fisso dell'asse y
    axs[1, 2].grid(True)
    axs[1, 2].yaxis.set_major_locator(plt.MultipleLocator(0.1))  # Imposta griglia ogni 0.1
    axs[1, 2].legend()

    # Salva il grafico nel percorso specificato
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()





# Plot ROC curve function
def plot_roc_curve(result_tar_mass, result_pred_mass, result_reco_mass, base_folder, pairtype):
    fpr_T, tpr_T, _ = roc_curve(result_tar_mass, result_pred_mass, pos_label=125.0)
    fpr_R, tpr_R, _ = roc_curve(result_tar_mass, result_reco_mass, pos_label=125.0)
    #fpr_S, tpr_S, _ = roc_curve(result_tar_mass, result_svfit, pos_label=125.0)

    # Compute AUC
    roc_auc_T = roc_auc_score(result_tar_mass, result_pred_mass)
    roc_auc_R = roc_auc_score(result_tar_mass, result_reco_mass)
    #roc_auc_S = roc_auc_score(result_tar_mass, result_svfit)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr_T, tpr_T, color='green', lw=2, label=f'ROC curve TPMT (area = {roc_auc_T:.2f})')
    plt.plot(fpr_R, tpr_R, color='red', lw=2, label=f'ROC curve RECO (area = {roc_auc_R:.2f})')
    #plt.plot(fpr_S, tpr_S, color='orange', lw=2, label=f'ROC curve SVFIT (area = {roc_auc_S:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.savefig(f"{base_folder}/ROC_TPMT_RECO_{pairtype}.png")
    plt.close()



# Plot mass comparison function
def plot_mass_comparison(resultsH, resultsH_svfit, resultsDY, resultsDY_svfit, base_folder, pairtype):
    plt.figure(figsize=(6, 4), dpi=400)
    plt.style.use(hep.style.CMS)
    plt.xticks(np.arange(0, 250, 20))
    x = np.linspace(0, 250, 150)

    # Define a function to create histograms
    def create_histogram(data, color, label, linestyle='solid'):
        hist, bin_edges = np.histogram(data, weights=np.ones(len(data)) / len(data), density=False, bins=100, range=(0, 250))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        return hist, bin_centers

    def fit_histogram(bin_centers, hist, initial_guess):
        popt, _ = curve_fit(gaussian, bin_centers, hist, p0=initial_guess)
        return popt

    hist_s_H, bin_centers_s_H = create_histogram(np.array(resultsH_svfit['svfit_mass']), 'blue', r'$m_{\tau\tau}$ SVFIT - H')
    hist_p_H, bin_centers_p_H = create_histogram(resultsH['pred_mass'], 'blue', r'$m_{\tau\tau}$ TPMT - H')

    hist_s_DY, bin_centers_s_DY = create_histogram(np.array(resultsDY_svfit['svfit_mass']), 'red', r'$m_{\tau\tau}$ SVFIT - Z')
    hist_p_DY, bin_centers_p_DY = create_histogram(resultsDY['pred_mass'], 'red', r'$m_{\tau\tau}$ TPMT - Z')

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
    plt.hist(resultsH_svfit['svfit_mass'], weights=np.ones(len(resultsH_svfit['svfit_mass'])) / len(resultsH_svfit['svfit_mass']),
             density=False, bins=100, range=(0, 250), histtype='step', linestyle="dashed", linewidth=1.2, label=r'$m_{\tau\tau}$ SVFIT - H', color='blue')
    plt.hist(resultsDY_svfit['svfit_mass'], weights=np.ones(len(resultsDY_svfit['svfit_mass'])) / len(resultsDY_svfit['svfit_mass']),
             density=False, bins=100, range=(0, 250), histtype='step', linestyle="dashed", linewidth=1.2, label=r'$m_{\tau\tau}$ SVFIT - Z', color='red')

    plt.plot(bin_centers_s_H, gaussian(bin_centers_s_H, *popt_H_svfit), linestyle='--', linewidth=0.8, color="blue")
    plt.plot(bin_centers_p_H, gaussian(bin_centers_p_H, *popt_H_tpm), linestyle='--', linewidth=0.8, color="blue")

    plt.plot(bin_centers_s_DY, gaussian(bin_centers_s_DY, *popt_DY_svfit), linestyle='--', linewidth=0.8, color="red")
    plt.plot(bin_centers_p_DY, gaussian(bin_centers_p_DY, *popt_DY_tpm), linestyle='--', linewidth=0.8, color="red")

    plt.xlim(0, 250)
    plt.xlabel('Invariant Mass [GeV]', fontsize=12)
    plt.ylabel('Events', fontsize=12)
    plt.title('TPMT Invariant Mass Distribution for H and Z samples', fontsize=16)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(loc='upper right', fontsize=10)
    plt.savefig(f"{base_folder}/RESULTS/DY_H_invariant_mass_{pairtype}.png")
    plt.close()

    return popt_H_svfit, popt_H_tpm, popt_DY_svfit, popt_DY_tpm

